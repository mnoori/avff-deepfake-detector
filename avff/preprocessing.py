import cv2
import torch
import numpy as np
from moviepy.editor import VideoFileClip
import librosa
from torchvision import transforms
import torchaudio
import torchvision.transforms as T
from torch.utils.data import Dataset
import os
import json
from typing import Dict, List, Tuple, Optional
import random
from PIL import Image
import logging

class VideoProcessor:
    def __init__(self, frame_count: int = 8, target_size: Tuple[int, int] = (224, 224)):
        self.frame_count = frame_count
        self.target_size = target_size
        
        # Simplified video augmentation pipeline
        self.video_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_frames(self, video_path: str) -> torch.Tensor:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Failed to open video: {video_path}")
                return torch.zeros((self.frame_count, 3, *self.target_size))
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < self.frame_count:
                logging.warning(f"Video has fewer frames ({total_frames}) than required ({self.frame_count})")
                return torch.zeros((self.frame_count, 3, *self.target_size))
            
            # Select evenly spaced frames
            frame_indices = np.linspace(0, total_frames-1, self.frame_count, dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    logging.error(f"Failed to read frame at index {idx}")
                    return torch.zeros((self.frame_count, 3, *self.target_size))
                
                # Convert BGR to RGB and process on CPU
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.video_transform(frame)
                frames.append(frame)
            
            cap.release()
            
            # Stack frames and move to GPU in one operation
            return torch.stack(frames)
            
        except Exception as e:
            logging.error(f"Error processing video {video_path}: {str(e)}")
            return torch.zeros((self.frame_count, 3, *self.target_size))

class AudioProcessor:
    def __init__(self, target_length: int = 16000):
        self.target_length = target_length
        self.resampler = None
    
    def process_audio(self, video_path: str) -> torch.Tensor:
        try:
            # Extract audio from video using moviepy
            video = VideoFileClip(video_path)
            audio = video.audio
            
            if audio is None:
                logging.warning(f"No audio found in video: {video_path}")
                return torch.zeros(self.target_length)
            
            # Extract audio as numpy array
            audio_array = audio.to_soundarray()
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Convert to torch tensor and process on CPU
            waveform = torch.from_numpy(audio_array).float()
            
            # Resample if necessary (moviepy audio is typically 44100Hz)
            if self.resampler is None:
                self.resampler = torchaudio.transforms.Resample(44100, 16000)
            waveform = self.resampler(waveform.unsqueeze(0))
            
            # Trim or pad to target length
            if waveform.shape[1] > self.target_length:
                waveform = waveform[:, :self.target_length]
            else:
                pad_length = self.target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
            # Clean up
            video.close()
            
            return waveform.squeeze(0)
            
        except Exception as e:
            logging.error(f"Error processing audio {video_path}: {str(e)}")
            return torch.zeros(self.target_length)

class DFDCDataset(Dataset):
    def __init__(self, root_dir: str, metadata_path: str, split: str = 'train'):
        self.root_dir = root_dir
        self.split = split
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        
        # Load metadata
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logging.info(f"Loaded metadata with {len(self.metadata)} entries")
        except Exception as e:
            logging.error(f"Failed to load metadata from {metadata_path}: {str(e)}")
            raise
        
        # Validate video files exist
        self.videos: List[Tuple[str, dict]] = []
        for filename, info in self.metadata.items():
            video_path = os.path.join(root_dir, filename)
            if os.path.exists(video_path):
                self.videos.append((filename, info))
            else:
                logging.warning(f"Video file not found: {video_path}")
        
        # Split dataset (80% train, 20% val)
        if split == 'train':
            self.videos = self.videos[:int(0.8 * len(self.videos))]
        elif split == 'val':
            self.videos = self.videos[int(0.8 * len(self.videos)):]
        
        logging.info(f"Initialized {split} dataset with {len(self.videos)} videos")
        
    def __len__(self) -> int:
        return len(self.videos)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        filename, video_info = self.videos[idx]
        video_path = os.path.join(self.root_dir, filename)
        
        try:
            # Process video and audio with error handling
            video_frames = self.video_processor.extract_frames(video_path)
            audio_features = self.audio_processor.process_audio(video_path)
            
            # Get label (0 for REAL, 1 for FAKE)
            label = 1 if video_info['label'] == 'FAKE' else 0
            
            # Move tensors to GPU in one operation
            return {
                'video': video_frames,
                'audio': audio_features,
                'label': torch.tensor(label, dtype=torch.long),
                'filename': filename
            }
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
            # Return zero tensors as fallback
            return {
                'video': torch.zeros((8, 3, 224, 224)),
                'audio': torch.zeros(16000),
                'label': torch.tensor(0, dtype=torch.long),
                'filename': filename
            } 