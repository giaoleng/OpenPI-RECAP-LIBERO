# 核心算法与训练框架

这里集中放置 VLA 的模型、策略、训练数据管线和通信服务。

| 子目录 | 主要功能 |
| --- | --- |
| `models/` | π₀.₅、双专家网络、LoRA、SigLIP、Gemma、Value Model |
| `training/` | 训练配置、DataLoader、优化器、归一化统计、Checkpoint 与参数加载 |
| `policies/` | LIBERO 观测/动作适配和策略输入输出 |
| `serving/` | WebSocket 策略服务 |
| `shared/` | 图像、JAX/Flax、数据、兼容性工具 |
| `models_pytorch/` | PyTorch 实现与相关 Transformer 模块 |

核心检查顺序：先验证模型初始化和权重加载，再检查 LIBERO 数据转换与动作维度，最后验证端到端推理。

**主数据路径**：LIBERO 观测 → 策略输入变换 → 模型输出动作 → 环境执行；训练端另通过 `value_label` / `reward_label` → Value → `adv_ind` 支持条件训练。

Value Model 当前包含 `from gemma.gm.nn...` 导入；依赖声明为 `gemma==3.3.0`，尚需确认远程环境中 Gemma 安装与模块接口兼容。参见 [代码完整性审计](../../docs/CODE_COMPLETENESS.md)。

