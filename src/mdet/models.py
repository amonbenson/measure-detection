"""Networks. Only ONNX-friendly operators: conv, batch norm, ReLU, pooling, softmax, concat."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def conv_bn(cin: int, cout: int, k=3, stride=1, dilation=1, groups=1) -> nn.Sequential:
    if isinstance(k, int):
        k = (k, k)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    pad = ((k[0] - 1) // 2 * dilation[0], (k[1] - 1) // 2 * dilation[1])
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, pad, dilation=dilation, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SepBlock(nn.Module):
    """Depthwise-separable residual block (MobileNet style)."""

    def __init__(self, c: int, dilation=(1, 1)):
        super().__init__()
        self.dw = conv_bn(c, c, 3, dilation=dilation, groups=c)
        self.pw = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return F.relu(x + self.pw(self.dw(x)))


class MeasureNet(nn.Module):
    """Boundary column model (guideline section 4.5, head 1).

    Input: (B, 2, H, W) strip at s = 8: channel 0 ink (1 = black), channel 1 staff-band mask.
    Output: (B, 4, W / 2) per column pair: logits of `start_edge` and `start_header` (the start
    boundary under the two annotation conventions, docs/CONVENTIONS.md), logit of `barline`, and the
    sub-column offset in [0, 1) (units of 2 px).
    """

    def __init__(self, width: int = 64, head: int = 96, heads: int = 4):
        super().__init__()
        self.heads = heads
        c1, c2, c3 = 32, 48, width
        self.stem = nn.Sequential(conv_bn(2, c1, 3, stride=(2, 2)), conv_bn(c1, c1, 3))
        self.down2 = nn.Sequential(conv_bn(c1, c2, 3, stride=(2, 1)), SepBlock(c2))
        self.down3 = nn.Sequential(conv_bn(c2, c3, 3, stride=(2, 1)), SepBlock(c3), SepBlock(c3, (2, 1)), SepBlock(c3, (4, 1)))
        self.att = nn.Conv2d(c3, heads, 1)
        self.proj = nn.Sequential(nn.Conv1d(c3 * heads + c3, head, 1, bias=False), nn.BatchNorm1d(head), nn.ReLU(inplace=True))
        self.tcn = nn.ModuleList()
        for d in (1, 2, 4, 8, 16, 32):
            self.tcn.append(nn.Sequential(nn.Conv1d(head, head, 3, padding=d, dilation=d, bias=False), nn.BatchNorm1d(head), nn.ReLU(inplace=True)))
        self.out = nn.Conv1d(head, 4, 1)
        nn.init.constant_(self.out.bias[:3], -4.0)  # focal-loss prior

    def forward(self, x):
        f = self.down3(self.down2(self.stem(x)))  # (B, C, H/8, W/2)
        b, c, h, w = f.shape
        a = torch.softmax(self.att(f), dim=2)  # attention over height, per head
        pooled = [(f * a[:, k : k + 1]).sum(dim=2) for k in range(self.heads)]
        pooled.append(f.amax(dim=2))
        z = self.proj(torch.cat(pooled, dim=1))
        for layer in self.tcn:
            z = z + layer(z)
        y = self.out(z)
        return torch.cat([y[:, :3], torch.sigmoid(y[:, 3:])], dim=1)


class LabelOCR(nn.Module):
    """Convolution-only CTC reader (guideline section 4.7).

    Input: (B, 1, 32, 128) crop, ink = 1. Output: (B, T = 32, n_classes) log-probabilities;
    class 0 is the CTC blank.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.body = nn.Sequential(
            conv_bn(1, 32, 3),
            conv_bn(32, 32, 3),
            nn.MaxPool2d(2),  # 16 x 64
            conv_bn(32, 64, 3),
            conv_bn(64, 64, 3),
            nn.MaxPool2d(2),  # 8 x 32
            conv_bn(64, 128, 3),
            conv_bn(128, 128, 3),
            nn.MaxPool2d((2, 1)),  # 4 x 32
            conv_bn(128, 160, 3),
            nn.MaxPool2d((2, 1)),  # 2 x 32
            conv_bn(160, 192, (2, 1), stride=1),  # collapses height with padding 0
        )
        self.seq = nn.Sequential(
            nn.Conv1d(192, 192, 3, padding=1, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, 192, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, n_classes, 1),
        )

    def forward(self, x):
        f = self.body(x)  # (B, 192, 1, 32)
        f = f.amax(dim=2)
        return torch.log_softmax(self.seq(f).transpose(1, 2), dim=2)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
