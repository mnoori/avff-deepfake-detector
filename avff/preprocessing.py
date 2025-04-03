import cv2
import torch
import numpy as np
from moviepy.editor import VideoFileClip
import librosa
from torchvision import transforms
from transformers import Wav2Vec2FeatureExtractor

class VideoProcessor:
    def __init__(self, target_size=(224, 224), num_frames=16):
        self.target_size = target_size
        self.num_frames = num_frames
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Calculate frame interval
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // self.num_frames)
        
        frame_idx = 0
        while len(frames) < self.num_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                # Resize and preprocess frame
                frame = cv2.resize(frame, self.target_size)
                frame = self.transform(frame)
                frames.append(frame)
                
            frame_idx += 1
            
        cap.release()
        
        # Pad if we don't have enough frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
            
        return torch.stack(frames)

class AudioProcessor:
    def __init__(self, sample_rate=16000, duration=5):
        self.sample_rate = sample_rate
        self.duration = duration
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        
    def extract_audio(self, video_path):
        # Extract audio using moviepy
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Convert to numpy array and normalize
        audio_array = audio.to_soundarray()
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
            
        # Resample if necessary
        if audio.fps != self.sample_rate:
            audio_array = librosa.resample(audio_array, orig_sr=audio.fps, target_sr=self.sample_rate)
            
        # Trim or pad to desired duration
        target_length = self.sample_rate * self.duration
        if len(audio_array) > target_length:
            audio_array = audio_array[:target_length]
        else:
            audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
            
        # Extract features using Wav2Vec2 feature extractor
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        return inputs.input_values 