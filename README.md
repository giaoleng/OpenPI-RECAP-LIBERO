# OpenPI-RECAP-Sim

基于 **π₀.₅ VLA 与 RECAP 风格优势条件策略训练**的 LIBERO 仿真闭环学习项目，覆盖专家示范微调、策略自主 Rollout、数据融合、VLM Value Model 训练、N-step Advantage 标注和策略再训练。

本 README 同时是**实验操作手册**：按照章节顺序执行，每一步保留输入、命令、预期产物与验收条件。训练、推理、权重加载及 LIBERO Rollout 统一在远程 GPU 环境完成；本地 Ubuntu 用于修改代码、Git 提交与阅读日志。

> **模块导览**：[训练与数据处理脚本](scripts/README.md) · [核心算法和策略框架](src/openpi/README.md) · [远程策略客户端](packages/openpi-client/README.md) · [LIBERO 仿真客户端](examples/libero/README.md) · [源码完整性与缺口审计](docs/CODE_COMPLETENESS.md)

> **状态说明**：当前已整合主要训练与仿真代码，部分依赖和接口仍需端到端验收。下文命令是待执行流程，不代表已经完成全部阶段。实验预测与实际成功率严格分开记录。

## 目录

1. [环境与目录准备](#1-环境与目录准备)
2. [闭环训练流程](#2-闭环训练流程)
3. [数据格式与字段定义](#3-数据格式与字段定义)
4. [准备 LIBERO 专家示范](#4-准备-libero-专家示范)
5. [配置 LIBERO 仿真环境](#5-配置-libero-仿真环境)
6. [初始 SFT 训练](#6-初始-sft-训练)
7. [策略服务与初始评测](#7-策略服务与初始评测)
8. [自主 Rollout 数据采集](#8-自主-rollout-数据采集)
9. [合并专家示范与 Rollout](#9-合并专家示范与-rollout)
10. [训练 VLM Value Model](#10-训练-vlm-value-model)
11. [N-step Advantage 标注](#11-n-step-advantage-标注)
12. [RECAP 风格策略再训练](#12-recap-风格策略再训练)
13. [LIBERO 对照评测与记录](#13-libero-对照评测与记录)
14. [常见问题与验收清单](#14-常见问题与验收清单)

## 1. 环境与目录准备

### 1.1 获取项目

```bash
git clone https://github.com/giaoleng/OpenPI-RECAP-Sim.git
cd OpenPI-RECAP-Sim
git submodule update --init --recursive
```

仓库中的 `src/openpi/` 是核心训练与策略代码，`scripts/` 提供训练和数据处理入口，`examples/libero/` 提供仿真评测客户端。数据、模型权重、Checkpoint 和视频不提交到 GitHub。

建议在**远程 GPU 容器**中使用下列目录，路径可依平台实际挂载位置调整：

```text
/workspace/OpenPI-RECAP-Sim/           # Git 仓库
/workspace/recap_data/
├── raw_libero_rlds/                  # 原始专家示范
├── lerobot_datasets/
│   ├── libero_demo/                  # 转换后的专家示范
│   ├── libero_rollout_round1/        # 第 1 轮自主 Rollout
│   ├── libero_mixed_round1/          # 合并后的数据
│   └── libero_mixed_round1_labeled/  # Advantage 标注专用副本
├── weights/
│   ├── pi05_base/                    # π₀.₅ 基础权重
│   └── vlm_ckpt/                     # Value Model 基础权重与 tokenizer
├── checkpoints/
└── outputs/
    ├── eval/
    └── logs/
```

设置工作路径（**在远程容器中运行**）：

```bash
export PROJECT_ROOT=/workspace/OpenPI-RECAP-Sim
export WORK_ROOT=/workspace/recap_data
mkdir -p "$WORK_ROOT"/{raw_libero_rlds,lerobot_datasets,weights,checkpoints,outputs/eval,outputs/logs}
cd "$PROJECT_ROOT"
```

**验收**：`pwd` 指向项目根目录，且 `ls scripts/train.py examples/libero/main.py pyproject.toml` 能找到三个文件。反馈目录路径和缺失文件信息即可。

### 1.2 Python 训练环境

项目的 `pyproject.toml` 要求 Python ≥3.11，包含 JAX、Flax、PyTorch、LeRobot、Gemma 等依赖。建议使用 `uv`，并让训练环境与仿真客户端环境隔离。

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --active
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
python -c "import jax, flax, torch; print('JAX:',jax.__version__,'Flax:',flax.__version__,'Torch:',torch.__version__); print('JAX devices:',jax.devices())"
```

**验收**：记录 Python、JAX、Torch 版本和 `jax.devices()`，GPU 应被正确识别。部分远程平台网络或 CUDA/JAX wheel 可能不匹配，应先解决依赖问题，再启动训练。仓库**没有** `pistar_requirements.txt`，不要照搬其他仓库的该文件安装命令。

### 1.3 模型与数据下载原则

- π₀.₅ 基础权重：上游公开的 `gs://openpi-assets/checkpoints/pi05_base`；下载至 `$WORK_ROOT/weights/pi05_base`。如平台无法连接 GCS，可使用合规镜像或已有权重副本，核对权重完整性。
- VLM Value Model 权重与 tokenizer：参考 `https://huggingface.co/ybpy/vlm_ckpt`，下载到 `$WORK_ROOT/weights/vlm_ckpt`。
- LIBERO RLDS 示范数据：`https://huggingface.co/datasets/openvla/modified_libero_rlds`。请确认实际目录结构后再填写转换参数。
- **不要提交**数据集、模型参数、训练输出到 Git。

**验收**：提供目录树和 Checkpoint 文件清单；不能只凭“下载命令退出成功”认定模型格式匹配。

## 2. 闭环训练流程

```text
LIBERO RLDS 专家示范
          │  格式转换 + value/reward/adv 标签
          ▼
     LeRobot Demo ──────────────┐
          │                     │
          ▼                     │
     π₀.₅ SFT                   │
          │                     │
          ▼                     │
  Policy Server + LIBERO        │
          │                     │
          ▼                     │
    Autonomous Rollout          │
    (成功 / 失败轨迹)            │
          │                     │
          └──── 数据合并 ◄───────┘
                    │
                    ▼
             Value Model
         学习 V(observation)
                    │
                    ▼
        N-step Advantage 估计
        + adv_ind 重新标注
                    │
                    ▼
     Advantage-conditioned SFT
                    │
                    ▼
         LIBERO 对照评测
       SFT vs. RECAP-style
```

分阶段开展：**Phase 0 环境 → Phase 1 数据 → Phase 2 SFT → Phase 3 Rollout → Phase 4 Value → Phase 5 Advantage → Phase 6 再训练 → Phase 7 评测**。先在 `libero_10` 的单个 Task 上跑通，再扩展为完整任务套件。

## 3. 数据格式与字段定义

使用 LeRobot 数据结构，至少检查以下字段：

| 字段 | 作用 | Demo / Rollout 的区别 |
| --- | --- | --- |
| `image` | 主视角 RGB 图像 | 两者均需要 |
| `wrist_image` | 腕部相机图像 | 两者均需要 |
| `state` | 机器人状态 | 两者均需要 |
| `actions` | 动作监督目标 | 两者均需要 |
| `intervention` | 是否为示范/人工干预帧 | Demo 为 1，自主 Rollout 为 0 |
| `value_label` | Value Model 的监督标签 | 由成功/失败及时间步规则生成 |
| `reward` | 稀疏任务奖励 | 成功轨迹通常末帧为 1 |
| `reward_label` | 用于 N-step Advantage 的奖励信号 | 按轨迹规则构造 |
| `adv_ind` | 优势条件 `positive` / `negative` / `none` | Demo 通常 positive，Rollout 初始为 none |

此外还需要 `timestamp`、`frame_index`、`episode_index`、`index` 和 `task_index` 等索引字段。**合并脚本不会自动补列、补奖励或修改图像布局**，不同来源的数据必须先对齐。

对于一条长度为 `T` 的成功 Demo，当前转换脚本的主要标签规则为：

```text
intervention = 1
adv_ind = positive
value_label(t) = -(T - 1 - t) / T
reward(t) = 1 (仅最后一帧), 否则 0
reward_label(t) = -1/T (非终止帧), 最后一帧为 0
```

**验收**：检查一条 Episode 的首帧、末帧和字段类型，确认 `image` 为 256×256 RGB、`state` 为 8 维、`actions` 为 7 维（此维度对应当前转换脚本的配置，不代表所有机器人通用）。

## 4. 准备 LIBERO 专家示范

### 4.1 转换 RLDS → LeRobot

先将公开的 LIBERO RLDS 数据下载/解压至 `$WORK_ROOT/raw_libero_rlds`，确认其中包含 `libero_10_no_noops`、`libero_goal_no_noops`、`libero_object_no_noops`、`libero_spatial_no_noops` 的相关数据目录，然后在**训练环境**运行：

```bash
python examples/libero/pistar_rlds_demo_processing.py \
  --data_dir "$WORK_ROOT/raw_libero_rlds" \
  --output_dir "$WORK_ROOT/lerobot_datasets" \
  --repo_name libero_demo
```

预期输出：`$WORK_ROOT/lerobot_datasets/libero_demo/`，包含 `meta/`、`data/`、`videos/` 等 LeRobot 数据结构。

> 转换脚本的 `--overwrite` 默认为 true，会删除同名既有输出目录。首次运行前先确认路径，避免覆盖数据。每个任务的真实 Episode 数以转换日志为准。

### 4.2 检查结果

```bash
ls "$WORK_ROOT/lerobot_datasets/libero_demo"
find "$WORK_ROOT/lerobot_datasets/libero_demo/meta" -maxdepth 1 -type f
```

**验收**：输出目录存在、Episode 数非零、视频与 Parquet 数据可读取、每个 Episode 的任务标签可对应。

## 5. 配置 LIBERO 仿真环境

推荐在远程容器中为 MuJoCo/LIBERO 单独准备 Python 3.10 环境，以减少与 JAX 训练栈的依赖冲突。

```bash
cd "$PROJECT_ROOT"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
uv venv --python 3.10 .venv-libero
source .venv-libero/bin/activate
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
uv pip install --no-deps git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
export PYTHONPATH="$PROJECT_ROOT/third_party/libero:$PYTHONPATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

**注意**：还需要根据 `examples/libero/requirements.txt` 与 `third_party/libero/requirements.txt` 安装对应依赖；`third_party/libero` 的版本、安装状态和 EGL 支持必须先检查，这组命令是环境构建起点，并非已验证的锁定安装方案。

先做**不调用策略模型**的仿真观测测试：

```bash
python examples/libero/main.py \
  --args.debug_raw_obs_only true \
  --args.debug_task_id 0 \
  --args.debug_num_steps 5
```

**验收**：环境能够 reset 和 step，原始图像正常导出，无 `libero` 导入错误及 EGL 异常。

若 EGL 失败，可先安装系统图形库；仍不成功时尝试 `MUJOCO_GL=glx` 配合 `xvfb-run`，不要在无图形服务的容器中直接假设 GLX 可用。

## 6. 初始 SFT 训练

**切回训练环境：**

```bash
cd "$PROJECT_ROOT"
source .venv/bin/activate
```

### 6.1 核对训练配置

当前仓库 `src/openpi/training/config.py` 已有 `pi05_star_libero` 和 `pi05_star_libero_infer` 配置，但仍保留上游默认数据集标识 `ybpy/libero_pistar`、基础权重 GCS 路径及较大的默认 batch size（256）。**必须先将对应 `repo_id` 和 `weight_loader` 改成实际数据与远程权重路径**，再检查 GPU 数量、batch size、归一化设置和 checkpoint 输出位置。

核对配置名称：

```bash
grep -n -A 24 'name="pi05_star_libero"' src/openpi/training/config.py
```

**验收**：确认 `repo_id`、权重路径、`batch_size`、`num_train_steps` 都符合实际环境。上述 256 只是当前源码默认值，不推荐不加调整就用于 2×RTX 3090。

### 6.2 归一化统计

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/compute_norm_stats.py --config-name pi05_star_libero
```

**验收**：完成数据统计，输出归一化资产，日志中没有数据字段/维度异常。

### 6.3 SFT 启动

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi05_star_libero --exp-name=demo_init --overwrite
```

**验收**：训练日志出现有效 loss 与 step；Checkpoint 目录有完整权重。若要继续断点训练，使用同一实验名并改为 `--resume`，不要再用 `--overwrite`。

> **资源约束**：初始参数和步骤数只是上游配置样例。实际运行前优先完成 1～5 step 的最小训练测试，确认内存、数据吞吐和保存机制后再决定正式训练预算。

## 7. 策略服务与初始评测

### 7.1 启动策略服务（训练环境）

```bash
source "$PROJECT_ROOT/.venv/bin/activate"
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_star_libero_infer \
  --policy.dir=checkpoints/pi05_star_libero/demo_init/<实际step目录>
```

`<实际step目录>` 为占位符，需替换为真实保存的 Checkpoint step 目录。推理配置必须与训练配置/权重匹配，且 `adv_ind_dropout=False`。

### 7.2 启动客户端评测（另一个终端，LIBERO 环境）

```bash
source "$PROJECT_ROOT/.venv-libero/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/third_party/libero:$PYTHONPATH"
export MUJOCO_GL=egl

python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.adv_ind_input positive \
  --args.task_suite_name libero_10 \
  --args.task_ids 0 \
  --args.num_trials_per_task 5 \
  --args.seed 7
```

**验收**：客户端能够连接服务端、模型返回动作、LIBERO 完成完整 Episode，日志包含成功或失败结果和视频。此处的 5 次测试是**冒烟验证**，不能用作稳定的最终成功率。

## 8. 自主 Rollout 数据采集

确保策略服务仍在运行，使用同一客户端脚本开启 LeRobot 保存功能：

```bash
python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.adv_ind_input positive \
  --args.task_suite_name libero_10 \
  --args.task_ids 0 \
  --args.num_trials_per_task 10 \
  --args.save_lerobot_rollout true \
  --args.rollout_output_dir "$WORK_ROOT/lerobot_datasets" \
  --args.rollout_repo_id libero_rollout_round1
```

预期输出：`$WORK_ROOT/lerobot_datasets/libero_rollout_round1`。注意：`--args.rollout_overwrite true` 会删除同名既有 Rollout 数据集，仅在明确需要重建时使用。

- `adv_ind_input=positive`：传给模型的**推理条件**。
- 数据集中的 `adv_ind=none`：尚未经过 Value Model 估计的**待标注状态**。
- `intervention=0`：仿真环境的自主执行帧；需同时保留成功和失败轨迹。

**验收**：核对 Episode 成功失败分布、数据字段、视频文件与 Parquet 帧数。若全部成功或全部失败，Value/Advantage 阶段的信息量可能不足，需要重新评估采集策略。

## 9. 合并专家示范与 Rollout

```bash
source "$PROJECT_ROOT/.venv/bin/activate"
python scripts/merge_datasets.py \
  --sources \
    "$WORK_ROOT/lerobot_datasets/libero_demo" \
    "$WORK_ROOT/lerobot_datasets/libero_rollout_round1" \
  --output "$WORK_ROOT/lerobot_datasets/libero_mixed_round1"
```

**验收**：合并前后 Episode 数、总帧数和任务 ID 映射一致；`intervention` 能区分 Demo 与 Rollout。若同名数据集已存在，脚本通常拒绝覆盖；仅在确定要删除重建时使用 `--overwrite`。

## 10. 训练 VLM Value Model

Value Model 从图像与任务信息学习 `V(o_t)`，监督来源为 `value_label`。它用于给自主 Rollout 排序/打标签，**不是 LIBERO 环境的任务成功判定器**。

先确认 Gemma tokenizer 文件存在，以及 Value 基础权重与模型配置兼容；优先使用小 batch 的短跑测试：

```bash
python scripts/train_value.py \
  --data_dir "$WORK_ROOT/lerobot_datasets/libero_mixed_round1" \
  --checkpoint_dir "$WORK_ROOT/checkpoints/value_model/round1_smoke" \
  --batch_size 4 \
  --num_train_steps 5 \
  --save_interval 5 \
  --val_interval 0 \
  --wandb_mode disabled \
  --load_pretrained \
  --tokenizer_path "$WORK_ROOT/weights/vlm_ckpt/tokenizer.model"
```

**验收**：模型初始化、前向/反向传播、loss 输出、Checkpoint 保存都成功。这里的 5 step 仅为功能检查，不代表完成有效训练。正式训练时另建 `round1` 目录，调整训练步数和验证策略，并保存曲线日志。

`--load_pretrained` 还依赖 `ValueModelWeightLoader` 能找到对应基础权重；如果使用了非默认路径，需要先配置 `OPENPI_VLM_CKPT_DIR` 或修改对应加载配置。**只指定 tokenizer 路径不等于完成权重路径配置。**

## 11. N-step Advantage 标注

算法使用 Value Model 对 Rollout 估计 `V_t`，结合 `reward_label` 计算 N-step Advantage：

```text
A_t = Σ_{k=0}^{N-1} r_(t+k) + V_(t+N) − V_t
```

其中跨 Episode 或临近结束时的边界处理应以脚本实现为准。再按 Advantage 分位数划分 `positive` / `negative` 标签，示范或干预帧通常保留 `positive`。

**重要：此步骤会修改数据集，先复制一份**：

```bash
cp -a "$WORK_ROOT/lerobot_datasets/libero_mixed_round1" \
      "$WORK_ROOT/lerobot_datasets/libero_mixed_round1_labeled"
```

配置好可用的 Value Checkpoint 后执行：

```bash
python scripts/label_advantage_from_vlm.py \
  --data_dir "$WORK_ROOT/lerobot_datasets/libero_mixed_round1_labeled" \
  --checkpoint_dir "$WORK_ROOT/checkpoints/value_model/round1" \
  --tokenizer_path "$WORK_ROOT/weights/vlm_ckpt/tokenizer.model" \
  --batch_size 8 \
  --lookahead 50 \
  --top_percent 30
```

**验收**：自主帧被标注为 `positive` 或 `negative`，示范帧标签未被误改，`none` 数量和各标签比例符合预期。若 `round1` 尚未形成有效 Checkpoint，就不能执行上述正式标注；先完成第 10 节训练。

## 12. RECAP 风格策略再训练

返回 `src/openpi/training/config.py`，新建或复制一个 LIBERO 训练配置（例如命名为 `pi05_star_libero_recap`）：

- `repo_id` 指向第 11 节**已经标注 Advantage 的数据集**，不能继续读原始 Demo。
- `weight_loader` 指向第 6 节已训练好的 SFT checkpoint（或明确记录重新初始化策略的实验设置）。
- 检查 `adv_ind` 是否进入 tokenizer 与训练输入，训练与推理的 dropout 设置是否区分。
- 训练样本保留专家数据、自主成功轨迹及自主失败轨迹，并按明确策略混合。

> `pi05_star_libero_recap` 是**建议新增的配置名**，当前仓库未确认已经注册，不能在修改 `config.py` 之前直接运行。

完成配置后，正式训练命令形式为：

```bash
python scripts/train.py pi05_star_libero_recap \
  --exp-name=recap_round1 --overwrite
```

**验收**：训练日志显示实际读取的 LeRobot 数据集标识、`adv_ind` 生效、loss 能更新、Checkpoint 可恢复。为该 checkpoint 配置对应的 `_infer` 推理配置并使用 `positive` 条件评测。

## 13. LIBERO 对照评测与记录

使用相同的任务、seed、Episode 数、相机与动作设置对比：

- **Baseline**：第 6 节 SFT Checkpoint。
- **RECAP-style**：第 12 节训练的 Checkpoint。
- **Task Suites**：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`（每套件 10 个任务）。
- **成功率**：`SR = 成功 Episode 数 / 评测 Episode 总数 × 100%`。
- **汇总**：先算各套件平均，再算四套件宏平均；保留每任务统计，不只留一个总数。

### 13.1 实验目标情景（估计值，尚未测量）

下表保留目前规划时使用的数字，仅用于设置目标和演示预期汇总格式。**不是本项目的真实成功率或已经完成的对照实验**。

| 任务套件 | π₀.₅ SFT（目标情景） | SFT + RECAP（目标情景） | 目标差值 |
| --- | ---: | ---: | ---: |
| LIBERO-Spatial | 85.0% | 89.0% | +4.0 pp |
| LIBERO-Object | 88.0% | 91.0% | +3.0 pp |
| LIBERO-Goal | 84.0% | 88.0% | +4.0 pp |
| LIBERO-10 | 68.0% | 77.0% | +9.0 pp |
| **四套件宏平均** | **81.25%** | **86.25%** | **+5.0 pp** |

### 13.2 实测结果填写模板

| 阶段 | Checkpoint | Task Suite | Seed | 每任务 Episodes | 成功次数 | 实测 SR |
| --- | --- | --- | --- | ---: | ---: | ---: |
| SFT | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| RECAP | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

建议至少保存：运行命令、Git commit、模型权重版本、数据集统计、训练日志、评测视频和成功失败明细。仅有高层预测表无法代替可复核的任务结果。

## 14. 常见问题与验收清单

| 问题 | 处理思路 |
| --- | --- |
| 找不到 `pistar_requirements.txt` | 本仓库按 `pyproject.toml` 安装；不要直接引用不存在的依赖文件 |
| 训练配置仍读取旧数据集 | 修改 `src/openpi/training/config.py` 中的 `repo_id`，确认数据集路径及 LeRobot 解析机制 |
| GCS / Hugging Face 下载失败 | 将权重放在远程共享存储，检查 `weight_loader`、tokenizer 路径和文件完整性 |
| `ModuleNotFoundError: libero` | 安装 LIBERO，检查 `third_party/libero` 及 `PYTHONPATH` |
| EGL / MuJoCo 渲染失败 | 检查驱动、EGL 库和容器图形设备；必要时验证 Xvfb + GLX |
| `KeyError: actions` | 核查 LeRobot schema；Value 数据加载不应错误地请求动作序列 |
| Value 模型在 step 0 崩溃 | 先查主机 RAM、JAX 设备、Gemma tokenizer、模型权重和依赖版本 |
| Rollout 仍是 `adv_ind=none` | 正常的初始状态；须执行第 11 节的 Advantage 重标注 |
| RECAP 没有提升 | 查 Value 过拟合、标签分布、数据不平衡、Checkpoint 一致性与评测方差 |

### 按顺序完成的最小里程碑

| 阶段 | 最低验收标准 | 状态 |
| --- | --- | --- |
| P0：环境 | 训练环境可导入 JAX/Flax，LIBERO 原始观测可运行 | 待验证 |
| P1：数据 | Demo 转换成功、字段/视频/标签正确 | 待验证 |
| P2：SFT | 5-step smoke test 与 Checkpoint 可保存 | 待验证 |
| P3：Rollout | 至少一个完整 Episode 与 LeRobot 导出 | 待验证 |
| P4：Value | Value Model 能前向、反向、保存和恢复 | 待验证 |
| P5：Advantage | 自主帧从 `none` 正确变为正/负标签 | 待验证 |
| P6：RECAP | 策略可从标注数据继续训练 | 待验证 |
| P7：评测 | SFT/RECAP 使用同一协议完成任务级对照 | 待验证 |

项目基于现有开源 VLA/RECAP 技术进行工程复现与适配；第三方实现来源、许可证和兼容性修改说明见 [整合文档](docs/INTEGRATION.md)、[修复审计](docs/PATCH_AUDIT.md) 与 [LICENSE](LICENSE)。
