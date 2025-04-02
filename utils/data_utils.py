import torch
import torchaudio
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from typing import Tuple, Optional

class VideoFrameDataset(Dataset):
    def __init__(self, 
                 video_paths: list,
                 labels: list,
                 frame_count: int = 16,
                 image_size: Tuple[int, int] = (224, 224),
                 audio_length: float = 1.0,
                 sample_rate: int = 16000):
        """
        Dataset for loading video frames and audio segments.
        
        Args:
            video_paths: List of paths to video files
            labels: List of labels (0 for real, 1 for fake)
            frame_count: Number of frames to sample from each video
            image_size: Size to resize frames to
            audio_length: Length of audio segment in seconds
            sample_rate: Audio sample rate
        """
        self.video_paths = video_paths
        self.labels = labels
        self.frame_count = frame_count
        self.image_size = image_size
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.video_paths)
    
    def _extract_frames(self, video_path: str) -> torch.Tensor:
        """Extract frames from video file."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames-1, self.frame_count, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.image_size)
                frame = self.transform(frame)
                frames.append(frame)
        
        cap.release()
        return torch.stack(frames)
    
    def _extract_audio(self, video_path: str) -> torch.Tensor:
        """Extract audio segment from video file."""
        waveform, sample_rate = torchaudio.load(video_path)
        
        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract segment of specified length
        target_length = int(self.audio_length * self.sample_rate)
        if waveform.shape[1] > target_length:
            start = torch.randint(0, waveform.shape[1] - target_length, (1,))
            waveform = waveform[:, start:start + target_length]
        else:
            # Pad if audio is shorter than target length
            pad_length = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        
        return waveform
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self._extract_frames(video_path)
        audio = self._extract_audio(video_path)
        
        return frames, audio, torch.tensor(label, dtype=torch.float32)

def create_data_loaders(
    train_videos: list,
    train_labels: list,
    val_videos: list,
    val_labels: list,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_videos: List of training video paths
        train_labels: List of training labels
        val_videos: List of validation video paths
        val_labels: List of validation labels
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        **dataset_kwargs: Additional arguments for VideoFrameDataset
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = VideoFrameDataset(train_videos, train_labels, **dataset_kwargs)
    val_dataset = VideoFrameDataset(val_videos, val_labels, **dataset_kwargs)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 