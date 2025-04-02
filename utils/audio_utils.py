import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional

def extract_audio_features(audio: torch.Tensor,
                         sample_rate: int = 16000,
                         n_mels: int = 80,
                         n_fft: int = 400,
                         hop_length: int = 160) -> torch.Tensor:
    """Extract mel spectrogram features from audio."""
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to log scale
    mel_spec = mel_spec(audio)
    mel_spec = torch.log(mel_spec + 1e-9)
    
    return mel_spec

def load_audio(file_path: str,
               target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """Load and resample audio file."""
    waveform, sample_rate = torchaudio.load(file_path)
    
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform, target_sr 