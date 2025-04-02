import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VisualFeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(VisualFeatureExtractor, self).__init__()
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
    def forward(self, x):
        # x shape: (batch_size, 3, height, width)
        features = self.backbone(x)
        features = features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
        return features

class AudioFeatureExtractor(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512):
        super(AudioFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, hidden_size)
        
    def forward(self, x):
        # x shape: (batch_size, 1, time_steps)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

class FeatureFusion(nn.Module):
    def __init__(self, visual_dim=2048, audio_dim=512, fusion_dim=1024):
        super(FeatureFusion, self).__init__()
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.attention = nn.MultiheadAttention(fusion_dim, num_heads=8)
        self.fusion_fc = nn.Linear(fusion_dim * 2, fusion_dim)
        
    def forward(self, visual_features, audio_features):
        # Project features to same dimension
        v_features = self.visual_proj(visual_features)
        a_features = self.audio_proj(audio_features)
        
        # Apply attention mechanism
        v_features = v_features.unsqueeze(0)  # Add sequence dimension
        a_features = a_features.unsqueeze(0)
        
        attn_output, _ = self.attention(v_features, a_features, a_features)
        attn_output = attn_output.squeeze(0)
        
        # Concatenate and fuse features
        fused_features = torch.cat([v_features.squeeze(0), attn_output], dim=1)
        fused_features = self.fusion_fc(fused_features)
        
        return fused_features

class AVFFModel(nn.Module):
    def __init__(self, num_classes=1):
        super(AVFFModel, self).__init__()
        self.visual_extractor = VisualFeatureExtractor()
        self.audio_extractor = AudioFeatureExtractor()
        self.fusion = FeatureFusion()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, visual_input, audio_input):
        # Extract features
        visual_features = self.visual_extractor(visual_input)
        audio_features = self.audio_extractor(audio_input)
        
        # Fuse features
        fused_features = self.fusion(visual_features, audio_features)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output

def get_model(num_classes=1):
    model = AVFFModel(num_classes=num_classes)
    return model 