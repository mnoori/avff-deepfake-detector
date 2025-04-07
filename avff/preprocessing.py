import cv2
import torch
import numpy as np
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
            # Load audio using torchaudio
            waveform, sample_rate = torchaudio.load(video_path)
            
            # Convert to mono if stereo
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != 16000:
                if self.resampler is None:
                    self.resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = self.resampler(waveform)
            
            # Trim or pad to target length
            if waveform.size(1) > self.target_length:
                waveform = waveform[:, :self.target_length]
            else:
                pad_length = self.target_length - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
            return waveform.squeeze(0)
            
        except Exception as e:
            logging.error(f"Error processing audio {video_path}: {str(e)}")
            # Try using librosa as fallback
            try:
                y, sr = librosa.load(video_path, sr=16000, duration=self.target_length/16000)
                waveform = torch.FloatTensor(y)
                
                if len(waveform) < self.target_length:
                    pad_length = self.target_length - len(waveform)
                    waveform = torch.nn.functional.pad(waveform, (0, pad_length))
                else:
                    waveform = waveform[:self.target_length]
                    
                return waveform
            except:
                return torch.zeros(self.target_length)

# Rest of the code remains the same...