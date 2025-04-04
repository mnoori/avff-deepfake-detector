import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, ViTModel, Wav2Vec2Model
from typing import Dict, List, Tuple, Optional
import logging

class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.feature_dim = self.encoder.config.hidden_size
        
    def to(self, device):
        super().to(device)
        self.encoder = self.encoder.to(device)
        return self
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length]
        outputs = self.encoder(x)
        # Use mean pooling over sequence dimension
        features = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
        return features

class VisualEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super().__init__()
        self.encoder = ViTModel.from_pretrained(model_name)
        self.feature_dim = self.encoder.config.hidden_size
        
    def to(self, device):
        super().to(device)
        self.encoder = self.encoder.to(device)
        return self
        
    def forward(self, x):
        # x shape: [batch_size, num_frames, channels, height, width]
        batch_size, num_frames = x.shape[:2]
        # Reshape to process all frames
        x = x.view(batch_size * num_frames, *x.shape[2:])  # [batch_size * num_frames, channels, height, width]
        outputs = self.encoder(x)
        # Get CLS token output for each frame
        features = outputs.last_hidden_state[:, 0]  # [batch_size * num_frames, hidden_size]
        # Reshape back to batch and average over frames
        features = features.view(batch_size, num_frames, -1).mean(dim=1)  # [batch_size, hidden_size]
        return features

class AVFFModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        audio_backbone: str = "facebook/wav2vec2-base-960h",
        visual_backbone: str = "google/vit-base-patch16-224",
        fusion_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Initialize encoders
        self.audio_encoder = AudioEncoder(audio_backbone)
        self.visual_encoder = VisualEncoder(visual_backbone)
        
        # Feature dimensions
        self.audio_dim = self.audio_encoder.feature_dim
        self.visual_dim = self.visual_encoder.feature_dim
        self.fusion_dim = fusion_dim
        
        # Feature projection layers
        self.audio_proj = nn.Linear(self.audio_dim, fusion_dim)
        self.visual_proj = nn.Linear(self.visual_dim, fusion_dim)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
    def to(self, device):
        super().to(device)
        self.audio_encoder = self.audio_encoder.to(device)
        self.visual_encoder = self.visual_encoder.to(device)
        return self
        
    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features
        audio_features = self.audio_encoder(audio)  # [batch_size, audio_dim]
        visual_features = self.visual_encoder(video)  # [batch_size, visual_dim]
        
        # Project features
        audio_features = self.audio_proj(audio_features)  # [batch_size, fusion_dim]
        visual_features = self.visual_proj(visual_features)  # [batch_size, fusion_dim]
        
        # Concatenate features
        fused_features = torch.cat([audio_features, visual_features], dim=1)  # [batch_size, fusion_dim * 2]
        
        # Fusion
        fused_features = self.fusion(fused_features)  # [batch_size, fusion_dim]
        
        # Classification
        logits = self.classifier(fused_features)  # [batch_size, num_classes]
        
        return {
            'logits': logits,
            'features': fused_features
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Main classification loss
        main_loss = F.cross_entropy(outputs['logits'], labels)
        
        return {
            'total_loss': main_loss,
            'main_loss': main_loss
        } 