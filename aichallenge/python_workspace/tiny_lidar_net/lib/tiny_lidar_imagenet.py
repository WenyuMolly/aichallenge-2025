import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional
from torch import Tensor
from jaxtyping import Float

from .model import TinyLidarNet, TinyLidarNetSmall


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class ResNet15(nn.Module):
    """Small ResNet-like backbone producing a compact global feature vector."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * BasicBlock.expansion, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride))
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TinyLidarImageNet(nn.Module):
    """
    Multi-modal model combining TinyLidarNet (or TinyLidarNetSmall) for 1D LiDAR
    and ResNet-50 for images. The ResNet's final `fc` is replaced with
    `nn.Identity()` so standard ResNet checkpoints can be loaded.

    Args:
      use_small_lidar: use the smaller LiDAR branch
      input_dim: LiDAR input length
      output_dim: final output dimension (e.g., 2 for steer/accel)
      resnet_pretrained: if True, use torchvision's builtin pretrained weights
      resnet_checkpoint: path to a checkpoint to load into the resnet backbone
      freeze_resnet: if True, freeze resnet parameters
      proj_dim: projection dimension for each modality before fusion
    """

    def __init__(
        self,
        use_small_lidar: bool = False,
        input_dim: int = 1080,
        output_dim: int = 2,
        resnet_pretrained: bool = False,
        resnet_checkpoint: Optional[str] = None,
        freeze_resnet: bool = False,
        resnet_type: str = "resnet15",
        resnet_out_dim: int = 256,
    ):
        super().__init__()

        if use_small_lidar:
            self.lidar_net = TinyLidarNetSmall(input_dim=input_dim, output_dim=output_dim)
        else:
            self.lidar_net = TinyLidarNet(input_dim=input_dim, output_dim=output_dim)

        # Instantiate ResNet backbone (small resnet15 or torchvision resnet50)
        if resnet_type == "resnet50":
            try:
                if resnet_pretrained:
                    self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                else:
                    self.resnet = models.resnet50(weights=None)
            except Exception:
                # Fallback for older torchvision versions
                self.resnet = models.resnet50(pretrained=resnet_pretrained)
            self.resnet.fc = nn.Identity()
            self.resnet_out_dim = 2048
        elif resnet_type == "resnet15":
            # small, custom ResNet variant
            self.resnet = ResNet15(out_dim=resnet_out_dim)
            self.resnet_out_dim = resnet_out_dim
        else:
            raise ValueError(f"Unsupported resnet_type: {resnet_type}")

        # Early fusion: project image feature to the same number of channels as the
        # last LiDAR conv output and fuse before flattening.
        # Find the last conv layer on the LiDAR branch to determine channels and temporal length
        conv_candidates = [getattr(self.lidar_net, f"conv{i}", None) for i in range(5, 0, -1)]
        last_conv = next((c for c in conv_candidates if c is not None), None)
        if last_conv is None:
            raise RuntimeError("LiDAR network has no conv layers to fuse with")
        conv_channels = last_conv.out_channels

        # Map image features to conv_channels and fuse along channel dim
        self.img_to_channels = nn.Linear(self.resnet_out_dim, conv_channels)
        self.fusion_conv = nn.Conv1d(conv_channels * 2, conv_channels, kernel_size=1)

        # Ensure the LiDAR FC head expects the original flattened dimensionality
        # (we perform early fusion on conv-level and keep flattened size unchanged)
        self.lidar_net.fc1 = nn.Linear(self.lidar_net.flatten_dim, 100)
        nn.init.kaiming_normal_(self.lidar_net.fc1.weight, nonlinearity="relu")
        if self.lidar_net.fc1.bias is not None:
            nn.init.constant_(self.lidar_net.fc1.bias, 0.0)

        if resnet_checkpoint:
            self.load_resnet_checkpoint(resnet_checkpoint, map_location="cpu")

        if freeze_resnet:
            for p in self.resnet.parameters():
                p.requires_grad = False

    # Note: fusion uses concatenation into TinyLidarNet's fc1, so no separate
    # projection/fusion heads are required.

    def load_resnet_checkpoint(self, path: str, map_location=None, strict: bool = False):
        sd = torch.load(path, map_location=map_location)
        # unbox common checkpoint wrappers
        if isinstance(sd, dict) and "state_dict" in sd and not any(k.startswith("conv1") for k in sd):
            sd = sd["state_dict"]

        def _strip_module(k: str) -> str:
            return k[7:] if k.startswith("module.") else k

        new_sd = { _strip_module(k): v for k, v in sd.items() }
        # remove fc keys (we replaced fc with Identity)
        filtered = {k: v for k, v in new_sd.items() if not k.startswith("fc.")}

        self.resnet.load_state_dict(filtered, strict=strict)

    def forward(self, lidar_x: Float[Tensor, "batch 1 1080"], img_x: Tensor) -> Float[Tensor, "batch 2"]:
        # LiDAR conv pipeline (reuse layers from existing LiDAR net)
        x = F.relu(self.lidar_net.conv1(lidar_x))
        x = F.relu(self.lidar_net.conv2(x))
        if hasattr(self.lidar_net, "conv3"):
            x = F.relu(self.lidar_net.conv3(x))
        if hasattr(self.lidar_net, "conv4"):
            x = F.relu(self.lidar_net.conv4(x))
        if hasattr(self.lidar_net, "conv5"):
            x = F.relu(self.lidar_net.conv5(x))

        # Early fusion: use conv-level features
        # x shape: [B, C, L]
        img_feat = self.resnet(img_x)
        if img_feat.ndim > 2:
            img_feat = torch.flatten(img_feat, 1)

        # Project image features into conv channel space and expand across temporal dim
        img_proj = F.relu(self.img_to_channels(img_feat))  # [B, C]
        img_proj = img_proj.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # [B, C, L]

        # Concatenate along channel dim and fuse back to conv_channels
        fused = torch.cat([x, img_proj], dim=1)  # [B, 2C, L]
        fused = F.relu(self.fusion_conv(fused))  # [B, C, L]

        x = torch.flatten(fused, start_dim=1)

        # Pass through TinyLidarNet's FC head
        x = F.relu(self.lidar_net.fc1(x))
        x = F.relu(self.lidar_net.fc2(x))
        x = F.relu(self.lidar_net.fc3(x))
        out = torch.tanh(self.lidar_net.fc4(x))
        return out
