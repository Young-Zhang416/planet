"""在线滤波/决策与优化封装：二者共享模型，但不互相持有状态。"""

from __future__ import annotations

from os import PathLike

import torch

from .model import WorldModel
from .networks import preprocess
from .planner import CEMPlanner
from .types import SequenceBatch, State


class Agent:
    def __init__(self, model: WorldModel, planner: CEMPlanner) -> None:
        self.model, self.planner = model, planner
        self.state: State | None = None
        self.previous_action: torch.Tensor | None = None

    def reset(self) -> None:
        self.state = self.previous_action = None

    @torch.no_grad()
    def act(self, image: torch.Tensor, exploration_std: float = 0.0) -> torch.Tensor:
        """单环境 uint8 [3,64,64] -> 归一化动作 [A]。

        保存的是实际加噪并裁剪后的动作；下一次滤波必须使用实际执行动作。
        """
        device = next(self.model.parameters()).device
        embedding = self.model.encoder(preprocess(image.to(device)[None], False))
        if self.state is None:
            prior = self.model.rssm.initial(1)
        else:
            assert self.previous_action is not None
            prior = self.model.rssm.imagine_step(self.state, self.previous_action)
        self.state = self.model.rssm.correct(prior, embedding)
        action = self.planner(self.state)
        action = (action + exploration_std * torch.randn_like(action)).clamp(-1, 1)
        self.previous_action = action
        return action[0].cpu()


class Trainer:
    def __init__(
        self, model: WorldModel, learning_rate: float = 1e-3, grad_clip: float = 1000.0
    ) -> None:
        self.model = model
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, eps=1e-4
        )
        self.updates = 0

    def step(self, batch: SequenceBatch) -> dict[str, float]:
        self.model.train()
        batch = batch.to(next(self.model.parameters()).device)
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.model.loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("世界模型损失非有限")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.updates += 1
        return {**metrics, "grad_norm": float(norm)}

    def save(self, path: str | PathLike[str]) -> None:
        from dataclasses import asdict

        torch.save(
            {
                "config": asdict(self.model.config),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "updates": self.updates,
            },
            path,
        )

    def load(self, path: str | PathLike[str]) -> None:
        from dataclasses import asdict

        checkpoint = torch.load(
            path, map_location=next(self.model.parameters()).device, weights_only=True
        )
        if checkpoint["config"] != asdict(self.model.config):
            raise ValueError("checkpoint 模型配置不一致")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.updates = checkpoint["updates"]
