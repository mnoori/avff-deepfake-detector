import torch
import torch.nn as nn
import torchvision.models as models

class VisualEncoder(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
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