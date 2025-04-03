import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, ViTModel

class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
    def forward(self, audio_input):
        outputs = self.wav2vec2(audio_input)
        return outputs.last_hidden_state

class VisualEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        
    def forward(self, visual_input):
        outputs = self.vit(visual_input)
        return outputs.last_hidden_state

class AVFFModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.visual_encoder = VisualEncoder()
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, audio_input, visual_input):
        # Encode audio and visual features
        audio_features = self.audio_encoder(audio_input)
        visual_features = self.visual_encoder(visual_input)
        
        # Average pooling over time dimension
        audio_features = torch.mean(audio_features, dim=1)
        visual_features = torch.mean(visual_features, dim=1)
        
        # Concatenate features
        combined_features = torch.cat([audio_features, visual_features], dim=1)
        
        # Classification
        logits = self.fusion(combined_features)
        return logits 