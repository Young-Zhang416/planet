"""公共数据协议：所有序列采用 batch-first，B 批大小，T 转移数。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Independent, Normal


@dataclass
class State:
    """RSSM 状态。单步 [..., D/S]，序列 [B, T, D/S]。"""

    deter: Tensor
    stoch: Tensor
    mean: Tensor
    std: Tensor

    @property
    def features(self) -> Tensor:
        # 确定性记忆和随机状态共同用于图像重建、奖励预测。
        return torch.cat((self.deter, self.stoch), -1)  # [..., D+S]

    @property
    def distribution(self) -> Independent:
        return Independent(Normal(self.mean, self.std), 1)  # batch [...], event [S]

    def detach(self) -> State:
        return State(*(x.detach() for x in self.tensors()))

    def tensors(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.deter, self.stoch, self.mean, self.std

    def at(self, t: int) -> State:
        return State(*(x[:, t] for x in self.tensors()))

    @staticmethod
    def stack(states: Sequence[State]) -> State:
        return State(
            *(torch.stack(xs, 1) for xs in zip(*(s.tensors() for s in states)))
        )


@dataclass
class SequenceBatch:
    """完整片段，不跨 episode。

    images: uint8 [B,T+1,3,64,64]，含初始观测 o_0。
    actions: float [B,T,A]，a_t 将 o_t 推进到 o_{t+1}。
    rewards: float [B,T]，执行 a_t 得到的奖励，由 s_{t+1} 预测。
    终止帧可以是最后一帧；禁止把 reset 后的图像当成终止帧。
    """

    images: Tensor
    actions: Tensor
    rewards: Tensor

    def to(self, device: torch.device | str) -> SequenceBatch:
        return SequenceBatch(
            self.images.to(device), self.actions.to(device), self.rewards.to(device)
        )

    def validate(self, action_size: int) -> None:
        b, t, a = self.actions.shape
        if (
            a != action_size
            or self.images.shape != (b, t + 1, 3, 64, 64)
            or self.rewards.shape != (b, t)
        ):
            raise ValueError("片段形状不满足 [B,T+1,3,64,64]/[B,T,A]/[B,T]")
        if t < 1 or self.images.dtype != torch.uint8:
            raise ValueError("至少需要一条转移，images 必须为 uint8")
