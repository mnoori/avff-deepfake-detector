import os
import json
import torch
import logging
from torch.utils.data import Dataset
from ..preprocessing import VideoProcessor, AudioProcessor
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

class DFDCDataset(Dataset):
    def __init__(self, root_dir: str, metadata_path: str, split: str = 'train'):
        """
        DFDC Dataset loader with improved error handling and logging
        
        Args:
            root_dir (str): Directory containing the video files
            metadata_path (str): Path to metadata.json
            split (str): 'train' or 'val'
        """
        self.root_dir = root_dir
        self.split = split
        
        # Load metadata
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logging.info(f"Loaded metadata with {len(self.metadata)} entries")
        except Exception as e:
            logging.error(f"Failed to load metadata from {metadata_path}: {str(e)}")
            raise
        
        # Setup processed data directory
        self.processed_dir = os.path.join(root_dir, 'processed')
        if not os.path.exists(self.processed_dir):
            raise RuntimeError(
                f"Processed data directory not found at {self.processed_dir}. "
                "Please run preprocess_dfdc.py first."
            )
        
        # Load processing metadata
        processing_metadata_path = os.path.join(self.processed_dir, 'processing_metadata.json')
        if not os.path.exists(processing_metadata_path):
            raise RuntimeError(
                f"Processing metadata not found at {processing_metadata_path}. "
                "Please run preprocess_dfdc.py first."
            )
        
        with open(processing_metadata_path, 'r') as f:
            processing_metadata = json.load(f)
            logging.info(f"Loaded processing metadata: {processing_metadata}")
        
        # Validate video files exist in processed directory
        self.videos: List[Tuple[str, dict]] = []
        for filename, info in self.metadata.items():
            video_name = os.path.splitext(filename)[0]
            processed_path = os.path.join(self.processed_dir, video_name)
            
            if os.path.exists(processed_path):
                frames_path = os.path.join(processed_path, 'frames.pt')
                audio_path = os.path.join(processed_path, 'audio.pt')
                
                if os.path.exists(frames_path) and os.path.exists(audio_path):
                    self.videos.append((filename, info))
                else:
                    logging.warning(f"Missing processed files for {video_name}")
            else:
                logging.warning(f"Missing processed directory for {video_name}")
        
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
        video_name = os.path.splitext(filename)[0]
        processed_path = os.path.join(self.processed_dir, video_name)
        
        try:
            # Load processed frames and audio
            frames = torch.load(os.path.join(processed_path, 'frames.pt'))
            audio = torch.load(os.path.join(processed_path, 'audio.pt'))
            
            # Get label (0 for REAL, 1 for FAKE)
            label = 1 if video_info['label'] == 'FAKE' else 0
            
            return {
                'video': frames,
                'audio': audio,
                'label': torch.tensor(label, dtype=torch.long),
                'filename': filename
            }
        except Exception as e:
            logging.error(f"Error loading processed data for {filename}: {str(e)}")
            # Return zero tensors as fallback
            return {
                'video': torch.zeros((8, 3, 224, 224)),
                'audio': torch.zeros(16000),
                'label': torch.tensor(0, dtype=torch.long),
                'filename': filename
            } 