import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_classes=1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x) 