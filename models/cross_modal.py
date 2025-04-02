import torch
import torch.nn as nn

class CrossModalFusion(nn.Module):
    def __init__(self, visual_dim=2048, audio_dim=512, fusion_dim=1024):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.cross_attention = nn.MultiheadAttention(fusion_dim, num_heads=8)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(self, visual_features, audio_features):
        # Project features to same dimension
        v_proj = self.visual_proj(visual_features)
        a_proj = self.audio_proj(audio_features)
        
        # Cross-attention
        attn_out, _ = self.cross_attention(v_proj.unsqueeze(0), 
                                         a_proj.unsqueeze(0),
                                         a_proj.unsqueeze(0))
        
        # Fusion
        fused = self.fusion_mlp(torch.cat([v_proj, attn_out.squeeze(0)], dim=1))
        return fused 