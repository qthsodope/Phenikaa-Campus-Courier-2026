"""Các mạng nơ-ron của khối đọc ảnh.

- KeypointNet: dò điểm (tâm giao lộ, 3 ô mẫu chú giải, ô thời tiết) dạng bản đồ nhiệt.
  Bộ mã hóa = ResNet18 học sẵn trên ImageNet (torchvision), cắt tới layer3; bộ giải mã nối tắt về stride 4.
- PatchNet: CNN nhỏ phân loại mảnh ảnh, nhiều "đầu ra" (head) tùy nhiệm vụ.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .common import EDGE_CLASSES, NODE_CLASSES, HEADINGS, KP_CHANNELS


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class KeypointNet(nn.Module):
    def __init__(self, pretrained=True, backbone="resnet18"):
        super().__init__()
        if backbone == "resnet34":
            w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            r = torchvision.models.resnet34(weights=w)
        else:
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            r = torchvision.models.resnet18(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)        # stride 2, 64
        self.pool = r.maxpool                                    # stride 4
        self.l1, self.l2, self.l3 = r.layer1, r.layer2, r.layer3  # stride 4/8/16: 64/128/256
        self.up3 = _block(256 + 128, 128)
        self.up2 = _block(128 + 64, 64)
        self.head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, len(KP_CHANNELS), 1))
        self.head[-1].bias.data.fill_(-4.0)                      # khởi đầu: hầu như không có điểm

    def forward(self, x):
        x = self.pool(self.stem(x))
        f1 = self.l1(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        y = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up3(torch.cat([y, f2], 1))
        y = F.interpolate(y, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up2(torch.cat([y, f1], 1))
        return self.head(y)                                      # (B,5,H/4,W/4) logits


class PatchNet(nn.Module):
    """CNN nhỏ: tích chập bước 2 + 4 tầng tích chập + gộp trung bình, rồi các đầu ra."""

    def __init__(self, heads, width=48):
        super().__init__()
        c = width
        self.body = nn.Sequential(
            nn.Conv2d(3, c, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            _block(c, c), nn.MaxPool2d(2),
            _block(c, 2 * c), nn.MaxPool2d(2),
            _block(2 * c, 4 * c), nn.MaxPool2d(2),
            _block(4 * c, 4 * c), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2))
        self.heads = nn.ModuleDict({k: nn.Linear(4 * c, n) for k, n in heads.items()})

    def forward(self, x):
        h = self.body(x)
        return {k: m(h) for k, m in self.heads.items()}


def edge_net():
    return PatchNet({"look": len(EDGE_CLASSES), "stairs": 2, "oneway": 3}, width=32)


def node_net():
    return PatchNet({"content": len(NODE_CLASSES), "heading": len(HEADINGS)}, width=32)


def misc_net():
    return PatchNet({"legend": 3, "weather": 2}, width=24)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
