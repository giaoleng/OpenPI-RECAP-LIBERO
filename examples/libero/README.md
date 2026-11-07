# LIBERO 仿真评测与 Rollout

本目录提供 LIBERO 策略评测客户端，支持按任务套件和 Task ID 进行评测，并导出 LeRobot 格式的自主 Rollout 轨迹。

> **状态**：评测脚本已整合；尚未完成本仓库的 GPU 端到端评测。下面均为运行示例，不是已执行的实验记录。

## 任务套件

| 套件 | 主要任务类型 |
| --- | --- |
| `libero_spatial` | 空间关系 |
| `libero_object` | 物体变化 |
| `libero_goal` | 目标条件变化 |
| `libero_10` | 长时序任务 |

每套件包含 10 个任务。正式评测需要分别统计每个任务的成功次数、Episode 总数和成功率。

## 准备仿真环境

LIBERO 本体未包含在当前仓库。需要在项目根目录准备其源码，并按固定版本安装适配依赖：

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
```

建议将训练环境与 MuJoCo/LIBERO 客户端环境隔离，避免依赖版本冲突；具体版本和兼容性仍需远程验证。

## 评测与数据采集

先启动匹配所用 checkpoint 的策略服务，再运行 LIBERO 客户端。示例：

```bash
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python examples/libero/main.py \
  --args.adv_ind_input positive \
  --args.task_suite_name libero_10 \
  --args.num_trials_per_task 5
```

只选择部分任务：

```bash
python examples/libero/main.py \
  --args.task_suite_name libero_spatial \
  --args.task_ids 0 3 7 \
  --args.num_trials_per_task 10
```

保存自主 Rollout 数据：

```bash
python examples/libero/main.py \
  --args.adv_ind_input positive \
  --args.task_suite_name libero_10 \
  --args.num_trials_per_task 5 \
  --args.save_lerobot_rollout true \
  --args.rollout_output_dir outputs/libero_rollouts \
  --args.rollout_repo_id libero_rollout_round1 \
  --args.rollout_overwrite true
```

`adv_ind_input=positive` 是**推理时送给策略的条件**；新导出的自主 Rollout 可先以 `adv_ind=none` 占位，后续再用 Value Model 推断得到 `positive` 或 `negative` 标签。

## 输出检查

每次评测至少检查控制台中的任务成功统计和保存的视频；开启 Rollout 导出时，还应核对 LeRobot 数据集的 `meta/`、`data/` 与 `videos/` 是否齐全。

## 评测口径与记录

统一固定 checkpoint、任务 ID、随机种子、每任务 Episode 数、是否允许重置和成功判定方式。成功率定义为 `成功 Episode 数 / 总 Episode 数`。不得将 README 中的命令示例或模拟预测表写作实测结果。

项目整体流程和预测情景见 [根目录 README](../../README.md)。上游代码来源与修改边界见 [整合说明](../../docs/INTEGRATION.md)，许可证见 [LICENSE](../../LICENSE)。
