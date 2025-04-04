import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel

class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.feature_dim = self.encoder.config.hidden_size
        
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
    def __init__(self, num_classes=2, audio_backbone="facebook/wav2vec2-base-960h", visual_backbone="google/vit-base-patch16-224"):
        super().__init__()
        self.audio_encoder = AudioEncoder(audio_backbone)
        self.visual_encoder = VisualEncoder(visual_backbone)
        
        # Fusion and classification
        total_feature_dim = self.audio_encoder.feature_dim + self.visual_encoder.feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, video_input, audio_input):
        # Extract features
        audio_features = self.audio_encoder(audio_input)
        visual_features = self.visual_encoder(video_input)
        
        # Concatenate features
        combined_features = torch.cat([audio_features, visual_features], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        return logits 