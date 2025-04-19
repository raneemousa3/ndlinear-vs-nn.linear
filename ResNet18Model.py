import torch
import torch.nn as nn
import torchvision.models as models
from ndLinearVideo import NdLinear

class ResNetBaselineVideoModel(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        # 1) Load ResNet‑18 and remove its head
        backbone = models.resnet18(pretrained=True)
        self.feature_dim = backbone.fc.in_features  # should be 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        # 2) Freeze if desired
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        # 3) Per‑frame projection
        self.frame_linear = nn.Linear(self.feature_dim, hidden_dim)
        # 4) Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, num_frames, H, W, C)
        b, t, H, W, C = x.shape
        # 1) Merge batch & time, permute to CNN format
        x = x.view(b * t, H, W, C).permute(0, 3, 1, 2)  # → (b*t, C, H, W)
        # 2) Extract features
        feats = self.backbone(x)                         # → (b*t, 512)
        # 3) Project per frame
        feats = self.frame_linear(feats)                 # → (b*t, hidden_dim)
        # 4) Un‑flatten back to (batch, time, hidden_dim)
        feats = feats.view(b, t, -1)                     # → (b, t, hidden_dim)
        # 5) Temporal pooling
        clip_feat = feats.mean(dim=1)                    # → (b, hidden_dim)
        # 6) Classify
        logits = self.classifier(clip_feat)               # → (b, num_classes)
        return logits


class ResNetNdVideoModel(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        # NdLinear instead of nn.Linear
        self.frame_ndlinear = NdLinear([self.feature_dim], [hidden_dim])
        self.classifier     = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        b, t, H, W, C = x.shape
        x = x.view(b * t, H, W, C).permute(0, 3, 1, 2)
        feats = self.backbone(x)                # (b*t, 512)
        feats = feats.view(b, t, self.feature_dim)  # → (b, t, 512)
        
        # NdLinear can handle the (b, t, feat) tensor directly:
        feats = self.frame_ndlinear(feats)      # → (b, t, hidden_dim)
        
        clip_feat = feats.mean(dim=1)           # → (b, hidden_dim)
        logits    = self.classifier(clip_feat)  # → (b, num_classes)
        return logits
