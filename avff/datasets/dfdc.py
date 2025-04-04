import os
import json
import torch
from torch.utils.data import Dataset
from ..preprocessing import VideoProcessor, AudioProcessor

class DFDCDataset(Dataset):
    def __init__(self, root_dir, metadata_path, split='train', transform=None):
        """
        DFDC Dataset loader
        
        Args:
            root_dir (str): Directory containing the video files
            metadata_path (str): Path to metadata.json
            split (str): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on frames
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Initialize processors
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Convert metadata dict to list of (filename, info) tuples
        self.videos = [(filename, info) for filename, info in self.metadata.items()]
        
        # Split dataset (80% train, 20% val)
        if split == 'train':
            self.videos = self.videos[:int(0.8 * len(self.videos))]
        elif split == 'val':
            self.videos = self.videos[int(0.8 * len(self.videos)):]
        
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        filename, video_info = self.videos[idx]
        video_path = os.path.join(self.root_dir, filename)
        
        # Process video and audio
        video_frames = self.video_processor.extract_frames(video_path)
        audio_features = self.audio_processor.extract_audio(video_path)
        
        # Get label (0 for REAL, 1 for FAKE)
        label = 1 if video_info['label'] == 'FAKE' else 0
        
        return {
            'video': video_frames,
            'audio': audio_features,
            'label': torch.tensor(label, dtype=torch.long),
            'filename': filename
        } 