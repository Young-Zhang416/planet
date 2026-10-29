# PlaNet：PyTorch 复现

依据 [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)（2019 年 v5）独立实现。包含像素编码/重建、RSSM、奖励预测、变分训练、可选 latent overshooting、CEM/MPC、episode 回放、在线交互、DeepMind Control 训练/评估与 checkpoint。

本项目提供算法实现与 CPU 回归验证，尚未进行论文六个任务的完整训练，也不宣称已复现论文分数。实际训练需要安装仿真依赖和充足计算资源。

## 安装与运行

Python >= 3.12。项目已有 uv 配置；在项目根目录执行：

```powershell
uv sync
uv run planet --mode smoke
uv run python -m unittest discover -s tests -v
```

`smoke` 用随机像素完成一次含 overshooting 的梯度更新和 CEM 决策，不需要 MuJoCo。直接运行源码也可以设置 `$env:PYTHONPATH='src'`，然后使用 `python -m planet`。

训练依赖为可选项：

```powershell
uv sync --extra control
uv run planet --mode train --domain cartpole --task swingup --action-repeat 8 --device cuda
uv run planet --mode eval --domain cartpole --task swingup --action-repeat 8 --device cuda --episodes 10 --checkpoint checkpoints/planet.pt
```

渲染需要可工作的 MuJoCo/OpenGL 环境。服务器可按 dm_control 文档配置 EGL；本地渲染配置依操作系统而异。核心模型不导入 dm_control。其余论文任务为 `reacher/easy`（repeat 4）、`cheetah/run`（4）、`finger/spin`（2）、`ball_in_cup/catch`（4）、`walker/walk`（2）。摄像机可通过 `--camera-id` 指定。

`--resume` 恢复模型、Adam 状态、更新计数；回放与随机数状态不保存在 checkpoint，恢复后重新采集 seed episodes，因此不是逐位一致的训练续跑。请为不同实验指定不同 checkpoint 路径。

## 模块职责

| 文件 | 职责 |
| --- | --- |
| `config.py` | 不可变模型和规划配置 |
| `types.py` | 状态与序列批次协议 |
| `networks.py` | 图像预处理、卷积编码器/解码器、奖励 MLP |
| `rssm.py` | 单步先验、后验校正、序列滤波 |
| `model.py` | 世界模型组装与训练目标 |
| `planner.py` | 批量 CEM 潜空间规划 |
| `replay.py` | 完整 episode 存储与等概率窗口采样 |
| `agent.py` | 在线滤波决策、梯度更新、checkpoint |
| `envs.py` | 可选 DeepMind Control 像素接口与动作重复 |
| `cli.py` | 采集—训练循环、评估与 smoke |

核心模型不依赖任何环境或缓冲区实现；规划器只依赖动力学与奖励预测。可直接导入 `WorldModel`、`Trainer`，用自己的 `SequenceBatch` 训练。

## 时间索引与张量流

使用 batch-first；B 为批大小，T 为转移数，A 为动作维度，D=200，S=30。

```text
images  [B,T+1,3,64,64] uint8：o_0,...,o_T
 actions [B,T,A] float：a_0,...,a_{T-1}，归一化 [-1,1]
 rewards [B,T] float：执行 a_t 后的奖励 r_{t+1}

images -> 5-bit preprocess -> Encoder -> embeddings [B,T+1,1024]
o_0 -> correct(zero_state,e_0) -> posterior_0
(posterior_t, a_t) -> imagine_step -> prior_{t+1}
(prior_{t+1}, e_{t+1}) -> correct -> posterior_{t+1}
[h,z] [B,T,230] -> Decoder -> image mean [B,T,3,64,64]
                -> reward MLP -> reward mean [B,T,1]
```

`State` 包含确定性记忆 `deter`、重参数化随机样本 `stoch`、对角高斯 `mean/std`。`features` 为 `[h,z]`。先验与后验共享同一确定性记忆；GRU 输入为上一随机状态与动作。初始图像用来建立信念，从第一个真实转移后的图像开始计算重建与奖励目标。片段从 episode 内随机截取，片段起始隐状态置零，不使用额外 burn-in。

编码器空间尺寸：64→31→14→6→2，通道数 3→32→64→128→256，展平为 1024。解码器空间尺寸：1→5→13→30→64。关键运算的形状和语义均在源码中注释。

## 目标与默认值

单位方差图像/奖励高斯的负对数似然（省略常数）：

```text
L_image = mean_{B,T}(0.5 * sum_{C,H,W} (image_mean - image_target)^2)
L_reward = mean_{B,T}(0.5 * (reward_mean - reward_target)^2)
L_KL = mean_{B,T}(max(3, KL(q(z_t|h_t,o_t) || p(z_t|h_t))))
L = L_image + reward_scale * L_reward + kl_scale * L_KL
    + overshooting_scale * L_overshooting
```

随机状态标准差为 `softplus(raw_std)+0.1`。训练时像素降低至 5-bit，并加入均匀去量化噪声，交互时关闭该噪声。Adam lr=1e-3、eps=1e-4、全局梯度范数裁剪 1000；默认 batch=50，T=50；随机采集 5 个 episode，每 100 次更新采集一个带标准差 0.3 动作噪声的 episode。

可选 overshooting：`--overshooting-distance 5 --overshooting-scale 1`。从每个后验起点展开多步先验，与未来后验计算 KL；目标后验和起点 detach，保留整个先验展开的梯度，对所有有效的距离>=2的配对平均。片段不跨 episode，无需跨终止的 mask。距离 1 的项由常规 KL 提供；默认禁用，与论文最新版附录 A 一致。

CEM 默认 horizon=12，iterations=10，candidates=1000，elites=100。每次从标准高斯初始化动作序列，裁剪到 [-1,1]，采样随机潜状态，以预测奖励总和选精英，重估均值和标准差，最终只执行第一步均值。无折扣、无 actor/critic、无像素空间 rollout；精英标准差使用总体估计，并以 1e-3 为下界以防退化。在线滤波保存实际加噪裁剪后的动作，保证与真实环境转移一致。

实现选择：遵循论文附录的 200 单元、两层奖励 MLP 和 reward_scale=1；官方公开配置采用三层 300 单元的通用 head，且 reward_scale=10，可在 `ModelConfig` 和 `networks.mlp` 中调整。使用 PyTorch 原生 GRUCell，不保证与 TensorFlow 内核数值一致。没有加入 Dreamer 的 KL balancing、离散 latent、actor/critic 等后续算法设计。

## 验证与参考

测试覆盖形状、时间因果性、标准差下界、各子模块梯度、参数更新、checkpoint 参数恢复、回放终止边界、CEM 单精英稳定性、动作范围和 agent reset。它们验证软件/算法契约，不替代控制任务学习曲线。

- [论文及附录](https://arxiv.org/html/1811.04551)
- [作者原始 TensorFlow 实现](https://github.com/google-research/planet)
- [官方配置](https://github.com/google-research/planet/blob/master/planet/scripts/configs.py)

## VS Code 断点调试

用 VS Code 打开项目根目录，安装推荐的 **Python** 和 **Python Debugger** 扩展，执行 `uv sync`。项目提供 `.vscode/launch.json`，显式使用项目 `.venv` 解释器并设置 `PYTHONPATH=src`，无需手动修改导入路径。

1. 在源码中点击行号左侧设置断点。
2. 按 `Ctrl+Shift+D` 打开“运行和调试”。
3. 选择一个 `PlaNet:` 配置，按 `F5` 启动。

| 配置 | 用途 | 前置条件 |
| --- | --- | --- |
| Smoke | 小模型梯度更新、overshooting 和 CEM，适合逐步检查张量 | `uv sync` |
| 单元测试 | 调试全部回归测试 | `uv sync` |
| Cartpole 训练调试 | CPU、1 个 seed episode、1 次更新、1 个采集 episode，batch=2、T=10 | `uv sync --extra control`，可用的 MuJoCo 渲染环境 |
| Cartpole 评估调试 | 无动作探索噪声，评估 1 个 episode | 同上，且已有 `checkpoints/debug-cartpole.pt` |

建议从 Smoke 开始：在 `WorldModel.loss`、`RSSM.imagine_step`、`RSSM.correct`、`CEMPlanner.__call__` 设置断点，查看 `batch.images.shape`、`state.features.shape` 和 `actions.shape`。调试器默认只进入本项目代码；需要进入 PyTorch 时，将配置中的 `justMyCode` 改为 `false`。

训练调试保留默认 CEM 预算，因此 CPU 采集可能较慢。调试产生的 checkpoint 仅用于检查流程，不代表训练收敛。正式训练可在 `launch.json` 的 `args` 中修改设备、episode 数和批次等参数。
