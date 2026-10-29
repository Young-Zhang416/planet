"""交叉熵法优化动作序列；只使用动力学先验和奖励头。"""

from __future__ import annotations

import torch

from .config import PlannerConfig
from .model import WorldModel
from .types import State


class CEMPlanner:
    def __init__(self, model: WorldModel, config: PlannerConfig | None = None) -> None:
        self.model, self.config = (
            model,
            config if config is not None else PlannerConfig(),
        )

    @torch.no_grad()
    def __call__(self, belief: State) -> torch.Tensor:
        c, m = self.config, self.model
        b, a = belief.deter.shape[0], m.config.action_size
        mean = belief.deter.new_zeros(b, c.horizon, a)
        std = torch.ones_like(mean)
        for _ in range(c.iterations):
            # [B,J,H,A]：候选之间并行，时间轴必须递推。
            actions = (
                mean[:, None]
                + std[:, None]
                * torch.randn(b, c.candidates, c.horizon, a, device=mean.device)
            ).clamp(-1, 1)
            state = State(
                *(
                    x[:, None]
                    .expand(-1, c.candidates, -1)
                    .reshape(b * c.candidates, -1)
                    for x in belief.tensors()
                )
            )
            returns = mean.new_zeros(b, c.candidates)
            for t in range(c.horizon):
                state = m.rssm.imagine_step(state, actions[:, :, t].reshape(-1, a))
                returns += m.reward(state.features).reshape(b, c.candidates)
            indices = returns.topk(c.elites, dim=1).indices  # [B,K]
            elites = actions.gather(
                1, indices[:, :, None, None].expand(-1, -1, c.horizon, a)
            )
            mean = elites.mean(1)
            std = elites.std(1, unbiased=False).clamp_min(c.min_std)
        return mean[:, 0]  # [B,A]，MPC 只执行首步，每次从 N(0,I) 重新规划。
