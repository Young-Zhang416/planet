"""小尺寸 CPU 回归测试，验证算法契约而非任务学习曲线。"""

import tempfile
import unittest
from pathlib import Path

import torch

from planet import (
    Agent,
    CEMPlanner,
    ModelConfig,
    PlannerConfig,
    ReplayBuffer,
    SequenceBatch,
    Trainer,
    WorldModel,
)
from planet.networks import preprocess


class PlanetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.model = WorldModel(
            ModelConfig(
                2,
                deter_size=12,
                stoch_size=3,
                hidden_size=16,
                free_nats=0,
                overshooting_distance=3,
                overshooting_scale=0.2,
            )
        )
        self.batch = SequenceBatch(
            torch.randint(256, (2, 4, 3, 64, 64), dtype=torch.uint8),
            torch.rand(2, 3, 2) * 2 - 1,
            torch.randn(2, 3),
        )

    def test_shapes_and_causality(self):
        emb = self.model.encoder(preprocess(self.batch.images, False))
        post, prior = self.model.rssm.observe(emb, self.batch.actions, sample=False)
        self.assertEqual(post.deter.shape, (2, 4, 12))
        self.assertEqual(prior.mean.shape, (2, 3, 3))
        self.assertEqual(
            self.model.decoder(post.features).shape, self.batch.images.shape
        )
        altered = emb.clone()
        altered[:, 2:] += 10
        post2, _ = self.model.rssm.observe(altered, self.batch.actions, sample=False)
        torch.testing.assert_close(post.mean[:, :2], post2.mean[:, :2])
        self.assertGreater(float(prior.std.min().detach()), 0.1)

    def test_gradients_and_checkpoint(self):
        trainer = Trainer(self.model)
        before = self.model.rssm.input.weight.detach().clone()
        metrics = trainer.step(self.batch)
        self.assertGreater(metrics["overshooting"], 0)
        self.assertFalse(torch.equal(before, self.model.rssm.input.weight))
        for name, module in [
            ("encoder", self.model.encoder),
            ("decoder", self.model.decoder),
            ("rssm", self.model.rssm),
            ("reward", self.model.reward),
        ]:
            self.assertTrue(
                any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in module.parameters()
                ),
                name,
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            trainer.save(path)
            restored = Trainer(WorldModel(self.model.config))
            restored.load(path)
            self.assertEqual(restored.updates, 1)
            for x, y in zip(self.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(x, y)

    def test_episode_boundaries(self):
        replay = ReplayBuffer(capacity_steps=6)
        for value in (10, 100, 200):
            replay.add(
                torch.full((4, 3, 64, 64), value, dtype=torch.uint8),
                torch.zeros(3, 2),
                torch.full((3,), float(value)),
            )
        self.assertEqual(replay.steps, 6)
        batch = replay.sample(20, 2)
        self.assertTrue(torch.all(batch.images[:, :, 0, 0, 0] == batch.rewards[:, :1]))
        self.assertFalse(torch.any(batch.images == 10))
        with self.assertRaises(ValueError):
            replay.sample(1, 4)

    def test_cem_optimizes_known_reward(self):
        # 人工动力学把动作写入状态，最优动作已知为 0.7。
        # 独立验证 CEM 的候选/精英维度及排序方向，不依赖未训练的神经网络。
        from types import SimpleNamespace

        from planet import State

        class Dynamics:
            def imagine_step(self, previous, action):
                return State(action, previous.stoch, previous.mean, previous.std)

        model = SimpleNamespace(
            config=SimpleNamespace(action_size=1),
            rssm=Dynamics(),
            reward=lambda f: -(f[..., :1] - 0.7).square(),
        )
        planner = CEMPlanner(
            model, PlannerConfig(horizon=1, iterations=5, candidates=200, elites=20)
        )
        zero = torch.zeros(2, 1)
        action = planner(State(zero, zero, zero, torch.ones_like(zero)))
        torch.testing.assert_close(
            action, torch.full_like(action, 0.7), atol=0.03, rtol=0
        )

    def test_agent_and_planner(self):
        planner = CEMPlanner(
            self.model, PlannerConfig(horizon=2, iterations=2, candidates=10, elites=1)
        )
        agent = Agent(self.model, planner)
        for _ in range(2):
            action = agent.act(self.batch.images[0, 0], exploration_std=0.3)
            self.assertEqual(action.shape, (2,))
            self.assertTrue(torch.isfinite(action).all())
            self.assertTrue((action.abs() <= 1).all())
            torch.testing.assert_close(action, agent.previous_action[0])
        agent.reset()
        self.assertIsNone(agent.state)


if __name__ == "__main__":
    unittest.main()
