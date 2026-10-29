"""python -m planet --help；smoke 不依赖物理仿真器。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .agent import Agent, Trainer
from .config import ModelConfig, PlannerConfig
from .model import WorldModel
from .planner import CEMPlanner
from .replay import ReplayBuffer
from .types import SequenceBatch

if TYPE_CHECKING:
    from .envs import ControlEnv


def collect(
    env: ControlEnv, agent: Agent | None = None, noise: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """一次完整采集；仅将真实 terminal 帧放进当前 episode。"""
    image = env.reset()
    images, actions, rewards = [image], [], []
    if agent:
        agent.reset()
    done = False
    while not done:
        action = (
            agent.act(image, noise) if agent else torch.rand(env.action_size) * 2 - 1
        )
        image, reward, done = env.step(action)
        images.append(image)
        actions.append(action)
        rewards.append(reward)
    return (
        torch.stack(images),
        torch.stack(actions),
        torch.tensor(rewards, dtype=torch.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PlaNet PyTorch：像素世界模型与在线 CEM 规划"
    )
    parser.add_argument("--mode", choices=["smoke", "train", "eval"], default="smoke")
    parser.add_argument("--domain", default="cartpole")
    parser.add_argument("--task", default="swingup")
    parser.add_argument("--action-repeat", type=int, default=8)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="train: 新采集 episode 数；eval: 评估数",
    )
    parser.add_argument("--seed-episodes", type=int, default=5)
    parser.add_argument("--updates-per-episode", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/planet.pt")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overshooting-distance", type=int, default=0)
    parser.add_argument("--overshooting-scale", type=float, default=0.0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.mode == "smoke":
        torch.set_num_threads(2)
        model = WorldModel(
            ModelConfig(
                2,
                deter_size=16,
                stoch_size=4,
                hidden_size=16,
                overshooting_distance=3,
                overshooting_scale=0.1,
            )
        ).to(args.device)
        trainer = Trainer(model)
        batch = SequenceBatch(
            torch.randint(256, (2, 4, 3, 64, 64), dtype=torch.uint8),
            torch.rand(2, 3, 2) * 2 - 1,
            torch.randn(2, 3),
        )
        print(json.dumps(trainer.step(batch)))
        planner = CEMPlanner(
            model, PlannerConfig(horizon=3, iterations=2, candidates=16, elites=4)
        )
        agent = Agent(model, planner)
        print("action:", agent.act(batch.images[0, 0]).tolist())
        return
    if (
        min(
            args.episodes,
            args.batch_size,
            args.sequence_length,
            args.seed_episodes,
            args.updates_per_episode,
        )
        < 1
    ):
        parser.error("采集和训练数量必须为正")
    from .envs import ControlEnv

    env = ControlEnv(
        args.domain, args.task, args.action_repeat, args.seed, args.camera_id
    )
    try:
        model = WorldModel(
            ModelConfig(
                env.action_size,
                overshooting_distance=args.overshooting_distance,
                overshooting_scale=args.overshooting_scale,
            )
        ).to(args.device)
        trainer = Trainer(model)
        if args.resume or args.mode == "eval":
            trainer.load(args.checkpoint)
        agent = Agent(model, CEMPlanner(model))
        if args.mode == "eval":
            model.eval()
            for episode in range(args.episodes):
                data = collect(env, agent, noise=0)
                print(
                    json.dumps(
                        {"episode": episode, "episode_return": float(data[2].sum())}
                    )
                )
            return
        replay = ReplayBuffer(seed=args.seed)
        for _ in range(args.seed_episodes):
            replay.add(*collect(env))
        for episode in range(args.episodes):
            for _ in range(args.updates_per_episode):
                metrics = trainer.step(
                    replay.sample(args.batch_size, args.sequence_length)
                )
            data = collect(env, agent)
            replay.add(*data)
            print(
                json.dumps(
                    dict(
                        episode=episode, episode_return=float(data[2].sum()), **metrics
                    )
                ),
                flush=True,
            )
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            trainer.save(args.checkpoint)
    finally:
        env.close()
