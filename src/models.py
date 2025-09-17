import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SmallResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = BasicConv3d(channels, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm2(self.conv2(out))
        out = F.relu(out + x)
        return out

class Encoder3D(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, embed_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            BasicConv3d(in_ch, base_ch, ks=5, stride=1, padding=2),
            BasicConv3d(base_ch, base_ch),
        )
        self.down1 = nn.Sequential(
            nn.MaxPool3d(2),
            BasicConv3d(base_ch, base_ch*2),
            SmallResBlock(base_ch*2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d(2),
            BasicConv3d(base_ch*2, base_ch*4),
            SmallResBlock(base_ch*4)
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(base_ch*4, embed_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x

# projector MLP used during pretraining (SimCLR)
class Projector(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)