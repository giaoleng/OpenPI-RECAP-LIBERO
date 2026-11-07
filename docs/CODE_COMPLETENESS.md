# 源码完整性审计与补齐计划

本报告根据 Git 文件树对比，目标是保证 **LIBERO 仿真 SFT → Rollout → Value → Advantage → RECAP** 的可运行性，而非合并两份上游仓库的全部历史文件。

## 统计范围

固定上游版本：
- PiStar：`d7b6ebdf41fdfeb368ef157965c1c41b7c056ddd`
- Destiny RECAP：`a44350b4a99f1d7f6137635ab3e894a5f72f5bda`

| Git 文件树范围 | 当前项目（补齐前） | PiStar | Destiny RECAP |
| --- | ---: | ---: | ---: |
| 整体 | 87 | 1273 | 1252 |
| `src/` | 49 | 71 | 73 |
| `scripts/` | 9 | 26 | 25 |
| `examples/` | 8 | 50 | 50 |
| `packages/` | 7 | 14 | 14 |
| `gemma/` | 0 | 155 | 148 |
| `control_your_robot/` | 0 | 934 | 920 |

**上述计数是仓库文件数，并非源码行数**。它不能直接衡量训练完整性。由于需要排除真机部署，`control_your_robot/` 不迁移是有意选择。

## 与当前闭环直接相关

| 项目 | 状态 | 处理办法 |
| --- | --- | --- |
| π₀.₅ 模型、LoRA、Value Model | 已纳入 | 测试初始化、权重和输入输出 |
| 策略训练 / Value 训练 / Advantage 标注 | 已纳入 | 检查配置、日志、关键中间变量 |
| LIBERO 客户端与数据转换 | 已纳入 | 缺少 LIBERO 本体；需锁定版本与验证依赖 |
| `scripts/check_value_data.py` | **本次补齐** | 测试 LeRobot batch 范围与字段 |
| `gemma/` 本地源码 | 未纳入 | 现有 `gemma==3.3.0` 依赖需验证；若无法满足再引入与许可证匹配的特定模块 |
| `uv.lock` | 未纳入 | 在测试通过的 Python / CUDA / JAX 组合上重新生成并提交 |
| `scripts/export_vlm_values.py` | 未纳入 | 建议补充以便画出 Value 随时间变化曲线 |
| `packages/openpi-client/runtime/` | 未纳入 | 按真实 import/客户端联调结果补充 |
| 模型、策略和 Client 的单元测试 | 主要未纳入 | 添加必要测试，避免复制大量不相关测试 |
| `src/openpi/training/config.py` | 已纳入但沿用旧 `repo_id` 等默认值 | 修正为本仓库的数据与 checkpoint，并新增 RECAP 独立配置 |
| `third_party/libero/` | 仓库不包含 | 在远程 GPU 环境拉取已固定的 LIBERO 版本 |

## 不属于当前实验的模块

不默认迁移 `control_your_robot/`、真机遥操作、YAM/Piper/RealMan 相关脚本、World Model 训练、Dynamics/Reward Classifier 独立实验、原上游完整 Notebook 与资源图片。今后若增加新实验，应独立记录迁移理由与依赖。

## 目录定位

本目录保留项目运行状态、代码完整性、依赖与兼容性审计记录，便于定位后续工作，而不是放置模型权重或训练数据。

## 验收优先级

1. **P0**：Gemma、JAX/Flax、LeRobot 版本/导入及 `uv` 依赖安装。
2. **P1**：LIBERO 环境导入、Demo 转换、5-step SFT 与策略服务。
3. **P2**：数据集 Batch 检查、Value 模型前向反向与保存恢复。
4. **P3**：Advantage 标注、RECAP 配置与完整单任务闭环。
5. **P4**：标准化四套件评测、日志和实验记录。

只有这些验证通过后，才可将项目标为**仿真闭环可运行**。

