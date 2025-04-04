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
        Dataset for loading preprocessed video frames and audio segments.
        
        Args:
            video_paths: List of paths to preprocessed video directories
            labels: List of labels (0 for real, 1 for fake)
            frame_count: Number of frames to sample from each video
            image_size: Size of frames
            audio_length: Length of audio segment in seconds
            sample_rate: Audio sample rate
        """
        self.video_paths = video_paths
        self.labels = labels
        self.frame_count = frame_count
        self.image_size = image_size
        self.audio_length = audio_length
        self.sample_rate = sample_rate
    
    def __len__(self):
        return len(self.video_paths)
    
    def _load_frames(self, video_dir: str) -> torch.Tensor:
        """Load preprocessed frames from .pt file."""
        try:
            frames_path = os.path.join(video_dir, 'frames.pt')
            frames = torch.load(frames_path)
            
            # Ensure frames have the correct shape (num_frames, channels, height, width)
            if len(frames.shape) != 4:
                raise ValueError(f"Expected frames to have 4 dimensions, got {len(frames.shape)}")
            
            # Sample frames if we have more than we need
            if frames.shape[0] > self.frame_count:
                indices = torch.linspace(0, frames.shape[0]-1, self.frame_count, dtype=torch.long)
                frames = frames[indices]
            
            # Select a random frame from the sequence
            frame_idx = torch.randint(0, frames.shape[0], (1,))
            frame = frames[frame_idx].squeeze(0)
            
            return frame
        except Exception as e:
            print(f"Error loading frames from {video_dir}: {str(e)}")
            return torch.zeros((3, *self.image_size))
    
    def _load_audio(self, video_dir: str) -> torch.Tensor:
        """Load preprocessed audio from .pt file."""
        try:
            audio_path = os.path.join(video_dir, 'audio.pt')
            audio = torch.load(audio_path)
            
            # Ensure audio is 2D (channels, samples)
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)
            elif len(audio.shape) > 2:
                raise ValueError(f"Expected audio to have 1 or 2 dimensions, got {len(audio.shape)}")
            
            # Extract segment of specified length if needed
            target_length = int(self.audio_length * self.sample_rate)
            if audio.shape[1] > target_length:
                start = torch.randint(0, audio.shape[1] - target_length, (1,))
                audio = audio[:, start:start + target_length]
            else:
                # Pad if audio is shorter than target length
                pad_length = target_length - audio.shape[1]
                audio = torch.nn.functional.pad(audio, (0, pad_length))
            
            return audio
        except Exception as e:
            print(f"Error loading audio from {video_dir}: {str(e)}")
            return torch.zeros((1, int(self.audio_length * self.sample_rate)))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_dir = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self._load_frames(video_dir)
        audio = self._load_audio(video_dir)
        
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