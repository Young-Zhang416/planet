"""世界模型与目标函数；不依赖环境、回放或规划器。"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import kl_divergence

from .config import ModelConfig
from .networks import Decoder, Encoder, mlp, preprocess
from .rssm import RSSM
from .types import SequenceBatch, State


class WorldModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder()
        self.rssm = RSSM(config)
        f = config.deter_size + config.stoch_size
        self.decoder = Decoder(f)
        self.reward = mlp(f, config.hidden_size, 1)

    def loss(self, batch: SequenceBatch) -> tuple[torch.Tensor, dict[str, float]]:
        """返回可微标量与日志。高斯 NLL 省略与参数无关的常数。

        图像误差先对 C,H,W 求和，再对 B,T 平均，不能直接对所有像素平均，
        否则会改变重建项与 KL 的相对权重。free nats 对每个状态的总 KL 裁剪。
        """
        batch.validate(self.config.action_size)
        images = preprocess(batch.images)  # [B,T+1,3,64,64]
        posts, priors = self.rssm.observe(self.encoder(images), batch.actions)
        target = State(*(x[:, 1:] for x in posts.tensors()))  # [B,T,*]
        predicted = self.decoder(target.features)  # [B,T,3,64,64]
        image_loss = 0.5 * (predicted - images[:, 1:]).square().sum((-3, -2, -1)).mean()
        rewards = self.reward(target.features).squeeze(-1)  # [B,T]
        reward_loss = 0.5 * (rewards - batch.rewards).square().mean()
        raw_kl = kl_divergence(target.distribution, priors.distribution)  # [B,T]
        kl = raw_kl.clamp_min(self.config.free_nats).mean()
        overshooting = images.new_zeros(())
        c = self.config
        if c.overshooting_scale and c.overshooting_distance >= 2:
            # 每个有效起点独立向前想象；绝不使用中间观测校正。
            # q 的目标参数 stop-gradient，起始状态也 detach；梯度仍通过整段 p 传播。
            terms = []
            tmax = batch.actions.shape[1]
            for start in range(tmax - 1):
                state = posts.at(start).detach()
                for distance in range(
                    1, min(c.overshooting_distance, tmax - start) + 1
                ):
                    state = self.rssm.imagine_step(
                        state, batch.actions[:, start + distance - 1]
                    )
                    if distance >= 2:
                        q = posts.at(start + distance).detach()
                        terms.append(
                            kl_divergence(q.distribution, state.distribution).clamp_min(
                                c.free_nats
                            )
                        )
            if terms:
                overshooting = torch.stack(terms).mean()
        total = (
            image_loss
            + c.reward_scale * reward_loss
            + c.kl_scale * kl
            + c.overshooting_scale * overshooting
        )
        metrics = {
            "loss": total,
            "image": image_loss,
            "reward": reward_loss,
            "kl": raw_kl.mean(),
            "overshooting": overshooting,
        }
        return total, {k: float(v.detach()) for k, v in metrics.items()}
