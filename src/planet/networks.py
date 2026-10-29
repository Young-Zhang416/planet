"""卷积观测模型，保留任意前导 batch/time 轴。"""

from __future__ import annotations

import torch
from torch import nn


def preprocess(images: torch.Tensor, dequantize: bool = True) -> torch.Tensor:
    """uint8 [...,3,64,64] -> float 同形状；5-bit 量化并中心化。

    训练添加 U[0,1/32) 去量化噪声；交互/评估关闭噪声。
    """
    x = torch.floor(images.float() / 8) / 32 - 0.5
    return x + torch.rand_like(x) / 32 if dequantize else x


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers = []
        # [3,64,64] -> [32,31,31] -> [64,14,14]
        # -> [128,6,6] -> [256,2,2] -> [1024]
        for cin, cout in zip((3, 32, 64, 128), (32, 64, 128, 256)):
            layers.extend((nn.Conv2d(cin, cout, 4, 2), nn.ReLU()))
        self.net = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images.reshape(-1, 3, 64, 64)).reshape(*images.shape[:-3], 1024)


class Decoder(nn.Module):
    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feature_size, 1024)
        # [1024,1,1] -> [128,5,5] -> [64,13,13] -> [32,30,30] -> [3,64,64]
        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 不加 sigmoid：观测分布为单位方差高斯，输出是其均值。
        x = self.fc(features).reshape(-1, 1024, 1, 1)
        return self.net(x).reshape(*features.shape[:-1], 3, 64, 64)


def mlp(input_size: int, hidden: int, output_size: int) -> nn.Sequential:
    """两个 ReLU 隐藏层，末层线性；奖励头输出 [...,1]。"""
    return nn.Sequential(
        nn.Linear(input_size, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output_size),
    )
