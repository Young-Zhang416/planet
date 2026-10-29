"""以完整 episode 存储，采样不会越过终止/reset 边界。"""

from __future__ import annotations

import random
from collections import deque

import torch

from .types import SequenceBatch


class ReplayBuffer:
    def __init__(self, capacity_steps: int = 1_000_000, seed: int = 0) -> None:
        if capacity_steps < 1:
            raise ValueError("capacity_steps 必须为正")
        self.capacity = capacity_steps
        self.episodes: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque()
        self.steps = 0
        self.rng = random.Random(seed)

    def add(
        self, images: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ) -> None:
        """单 episode: [T+1,C,H,W], [T,A], [T]；内部 CPU 副本隔离调用者修改。"""
        batch = SequenceBatch(images[None], actions[None], rewards[None])
        batch.validate(actions.shape[-1])
        if len(actions) > self.capacity:
            raise ValueError("单 episode 超过缓冲区容量")
        self.episodes.append(
            tuple(x.detach().cpu().clone() for x in (images, actions, rewards))
        )
        self.steps += len(actions)
        while self.steps > self.capacity:
            self.steps -= len(self.episodes.popleft()[1])

    def sample(self, batch_size: int, length: int) -> SequenceBatch:
        if min(batch_size, length) < 1:
            raise ValueError("batch_size 和 length 必须为正")
        # 按有效窗口数加权，确保所有可用子序列等概率。
        episodes = [e for e in self.episodes if len(e[1]) >= length]
        if not episodes:
            raise ValueError("回放中没有足够长的 episode")
        selected = self.rng.choices(
            episodes, weights=[len(e[1]) - length + 1 for e in episodes], k=batch_size
        )
        chunks = []
        for images, actions, rewards in selected:
            s = self.rng.randrange(len(actions) - length + 1)
            chunks.append(
                (
                    images[s : s + length + 1],
                    actions[s : s + length],
                    rewards[s : s + length],
                )
            )
        return SequenceBatch(*(torch.stack(xs) for xs in zip(*chunks)))
