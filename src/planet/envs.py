"""可选 DeepMind Control 像素适配器；核心库不需要 dm_control。"""

from __future__ import annotations

import numpy as np
import torch


class ControlEnv:
    def __init__(
        self,
        domain: str,
        task: str,
        action_repeat: int = 4,
        seed: int = 0,
        camera_id: int = 0,
    ) -> None:
        from dm_control import suite

        if action_repeat < 1:
            raise ValueError("action_repeat 必须为正")
        self.env = suite.load(domain, task, task_kwargs={"random": seed})
        self.repeat, self.camera = action_repeat, camera_id
        spec = self.env.action_spec()
        self.low, self.high = spec.minimum, spec.maximum
        self.action_size = spec.shape[0]

    def _image(self) -> torch.Tensor:
        # render HWC uint8 -> CHW uint8，保持与回放的数据协议一致。
        pixels = self.env.physics.render(64, 64, camera_id=self.camera)
        return torch.from_numpy(pixels.copy()).permute(2, 0, 1).contiguous()

    def reset(self) -> torch.Tensor:
        self.env.reset()
        return self._image()

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool]:
        # 世界模型统一学习 [-1,1] 动作，映射仅在环境边界完成。
        normalized = np.clip(action.numpy(), -1, 1)
        native = self.low + (normalized + 1) * 0.5 * (self.high - self.low)
        reward = 0.0
        for _ in range(self.repeat):
            timestep = self.env.step(native)
            reward += float(timestep.reward or 0)
            if timestep.last():
                break
        return self._image(), reward, timestep.last()

    def close(self) -> None:
        self.env.close()
