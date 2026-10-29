from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """论文附录 A 的默认尺寸；输入固定为 RGB 64×64。"""

    action_size: int
    deter_size: int = 200
    stoch_size: int = 30
    hidden_size: int = 200
    min_std: float = 0.1
    free_nats: float = 3.0
    kl_scale: float = 1.0
    reward_scale: float = 1.0
    overshooting_distance: int = 0
    overshooting_scale: float = 0.0

    def __post_init__(self) -> None:
        if (
            min(self.action_size, self.deter_size, self.stoch_size, self.hidden_size)
            < 1
        ):
            raise ValueError("模型尺寸必须为正整数")
        if self.min_std <= 0 or self.free_nats < 0 or self.overshooting_distance < 0:
            raise ValueError("无效的标准差、free nats 或 overshooting 距离")


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 12
    iterations: int = 10
    candidates: int = 1000
    elites: int = 100
    min_std: float = 1e-3

    def __post_init__(self) -> None:
        if (
            min(self.horizon, self.iterations, self.elites) < 1
            or self.candidates < self.elites
        ):
            raise ValueError(
                "CEM 要求 horizon/iterations > 0，1 <= elites <= candidates"
            )
        if self.min_std <= 0:
            raise ValueError("min_std 必须为正")
