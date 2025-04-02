import torch
import torchaudio
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import Tuple, Optional

def preprocess_video_frames(frames: np.ndarray, 
                          target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Preprocess video frames."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    processed_frames = []
    for frame in frames:
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        processed_frames.append(frame)
    
    return torch.stack(processed_frames)

def preprocess_audio(audio: torch.Tensor,
                    target_length: int,
                    sample_rate: int = 16000) -> torch.Tensor:
    """Preprocess audio waveform."""
    # Resample if necessary
    if audio.shape[1] > target_length:
        start = torch.randint(0, audio.shape[1] - target_length, (1,))
        audio = audio[:, start:start + target_length]
    else:
        # Pad if audio is shorter than target length
        pad_length = target_length - audio.shape[1]
        audio = torch.nn.functional.pad(audio, (0, pad_length))
    
    return audio 