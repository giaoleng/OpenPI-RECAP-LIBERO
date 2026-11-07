# PiStar：LIBERO 仿真流程中文整理

> 本文基于 [ybpy/pistar](https://github.com/ybpy/pistar) 固定提交 `d7b6ebdf41fdfeb368ef157965c1c41b7c056ddd` 的公开 README 与代码，**仅翻译和保留本项目需要的 LIBERO 内容**。不是上游全部真机功能的逐字翻译，也不是本项目实验日志。

## 项目定位

PiStar 在 OpenPI 基础上增加了基于经验与价值条件的机器人策略训练流程，包含数据转换、自主 Rollout、Value Model、Advantage 标注及继续微调。本项目只使用 LIBERO 仿真。

## LIBERO 数据输入

原始专家示范来自 LIBERO RLDS 数据。可使用 `examples/libero/pistar_rlds_demo_processing.py` 转换为 LeRobot 数据，并保留：

| 字段 | 意义 |
|---|---|
| `image`、`wrist_image` | 环境和腕部相机图像 |
| `state` | 机器人状态 |
| `actions` | 动作序列 |
| `intervention` | 人工示范/纠正标识；纯仿真示范使用 1 |
| `reward` | 稀疏成功奖励 |
| `reward_label` | 供 Value/Advantage 计算的逐帧奖励标签 |
| `value_label` | 价值模型监督目标 |
| `adv_ind` | 策略优势条件：positive / negative / none |

## 策略推理与闭环评测

`examples/libero/main.py` 可选择完整套件或部分 Task ID：

```bash
# 只评测 Spatial 中的任务 0、3、7，每任务 10 次。
python examples/libero/main.py --args.task_suite_name libero_spatial --args.task_ids 0 3 7 --args.num_trials_per_task 10
```

```bash
# 评测 Long 并保存 Rollout。
python examples/libero/main.py --args.task_suite_name libero_10 --args.save_lerobot_rollout true --args.rollout_output_dir outputs/libero_rollouts
```

运行时还需要启动策略服务端，模型训练环境必须已经安装、Checkpoint 可用，且 LIBERO 依赖已正确导入。

## 策略训练与后训练

- `scripts/train.py`：JAX/Flax 参数训练入口，使用已注册的 TrainConfig。
- `scripts/train_value.py`：Value Model 训练。
- `scripts/label_advantage_from_vlm.py`：依据 Value 估计 N-step Advantage，并写入 `adv_ind`。
- `scripts/merge_datasets.py`：合并规范化后的 LeRobot 示范和 Rollout。

**注意**：`adv_ind` 依赖模型和数据预处理配置，不应把正负样本简单当作动作监督标签。真实实验必须先完成 Stage 4/5 数据一致性验证。

## 本项目与上游的区别

- 删除真机训练部署目标，公开范围仅为 LIBERO。
- 按 Destiny 修复价值训练与数据接口。
- 增加中文代码注释及中文实验审计文档。
- 上游 README 的实验设置只作参考，不作为本项目成绩。
