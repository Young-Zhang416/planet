"""Recurrent State Space Model：环境无关的单步动力学与滤波。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .types import State


class RSSM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = c = config
        self.input = nn.Linear(c.stoch_size + c.action_size, c.hidden_size)
        self.gru = nn.GRUCell(c.hidden_size, c.deter_size)
        self.prior = nn.Sequential(
            nn.Linear(c.deter_size, c.hidden_size),
            nn.ReLU(),
            nn.Linear(c.hidden_size, 2 * c.stoch_size),
        )
        self.posterior = nn.Sequential(
            nn.Linear(c.deter_size + 1024, c.hidden_size),
            nn.ReLU(),
            nn.Linear(c.hidden_size, 2 * c.stoch_size),
        )

    def initial(self, batch_size: int) -> State:
        p = next(self.parameters())
        h = p.new_zeros(batch_size, self.config.deter_size)
        z = p.new_zeros(batch_size, self.config.stoch_size)
        return State(h, z, z.clone(), torch.ones_like(z))

    def _state(self, h: torch.Tensor, stats: torch.Tensor, sample: bool) -> State:
        mean, raw_std = stats.chunk(2, -1)  # 各 [B,S]
        std = F.softplus(raw_std) + self.config.min_std
        z = mean + std * torch.randn_like(mean) if sample else mean
        return State(h, z, mean, std)

    def imagine_step(
        self, previous: State, action: torch.Tensor, sample: bool = True
    ) -> State:
        """h_t=GRU(h_{t-1}, [z_{t-1},a_{t-1}]); p(z_t|h_t)。"""
        x = F.relu(self.input(torch.cat((previous.stoch, action), -1)))  # [B,H]
        h = self.gru(x, previous.deter)  # [B,D]
        return self._state(h, self.prior(h), sample)

    def correct(
        self, prior: State, embedding: torch.Tensor, sample: bool = True
    ) -> State:
        """q(z_t|h_t,e_t)，后验和先验共享 h_t；embedding [B,1024]。"""
        return self._state(
            prior.deter, self.posterior(torch.cat((prior.deter, embedding), -1)), sample
        )

    def observe(
        self, embeddings: torch.Tensor, actions: torch.Tensor, sample: bool = True
    ) -> tuple[State, State]:
        """[B,T+1,1024] 与 [B,T,A] -> 后验 [B,T+1,*]、先验 [B,T,*]。

        o_0 直接校正零初态，不执行虚构的环境动作；后续逐步先预测再校正。
        """
        state = self.correct(self.initial(actions.shape[0]), embeddings[:, 0], sample)
        posts, priors = [state], []
        for t in range(actions.shape[1]):
            prior = self.imagine_step(state, actions[:, t], sample)
            state = self.correct(prior, embeddings[:, t + 1], sample)
            priors.append(prior)
            posts.append(state)
        return State.stack(posts), State.stack(priors)
