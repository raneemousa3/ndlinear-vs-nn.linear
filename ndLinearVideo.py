
try:
    from NdLinear.ndlinear import NdLinear
except ImportError:
    import sys
    sys.path.append("/Users/raneemmousa/Desktop/NdLinear/NdLinear")
    from NdLinear import NdLinear

import torch.nn as nn
class NdVideoModel(nn.Module):
    def __init__(self, channels, hid_dim, num_classes):
        super().__init__()
        self.ndlinear   = NdLinear([channels], [hid_dim])
        self.classifier = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        # x: (batch, frames, H, W, channels)
        x = self.ndlinear(x)             # → (batch, frames, H, W, hid_dim)
        x = x.mean(dim=(1,2,3))          # → (batch, hid_dim)
        return self.classifier(x)        # → (batch, num_classes)
class BaselineVideoModel(nn.Module):
    def __init__(self, channels, hidden_dim, num_classes):
        super(BaselineVideoModel, self).__init__()
        # Linear layer that maps each pixel’s channel vector into hidden_dim
        self.linear     = nn.Linear(channels, hidden_dim)
        # Final classifier from pooled features to class logits
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, frames, height, width, channels)
        batch, frames, H, W, C = x.shape

        # 1) Flatten every “pixel” across all frames & spatial dims into rows
        #    New shape: (batch * frames * H * W, C)
        x = x.view(-1, C)

        # 2) Apply the linear transformation to each channel vector
        #    Now shape: (batch * frames * H * W, hidden_dim)
        x = self.linear(x)

        # 3) Reshape back to 5D so we can pool over frames + spatial dims:
        #    (batch, frames, H, W, hidden_dim)
        x = x.view(batch, frames, H, W, -1)

        # 4) Average‐pool over frames, height, and width:
        #    x.mean(dim=(1,2,3)) → (batch, hidden_dim)
        x = x.mean(dim=(1, 2, 3))

        # 5) Final classification layer to get logits per video:
        #    (batch, num_classes)
        logits = self.classifier(x)
        return logits
