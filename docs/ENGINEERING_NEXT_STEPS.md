# 工程修复与验证操作单（2026-11-08 历史阶段记录）

> **当前状态说明（2026-11-09）：** 此文记录 11 月 8 日那轮源码修复和静态检查。“本轮未安装/未启动”“下一步”等仅限该轮审计，不代表整个项目仍未完成。项目维护者确认后续已完成远程 GPU 权重加载、SFT、LIBERO Episode、Value、Advantage、RECAP 和对照评测。最新项目状态见 [PROJECT_STATE.md](PROJECT_STATE.md)；原始训练日志与模型未公开。以下命令仅作为技术复核参考。

## 已处理

| 项目 | 代码位置 | 验证 |
|---|---|---|
| 配置字段缺失/本地数据root丢失 | `training/config.py`、`data_loader.py`、Value/Adv脚本 | 本地root、逻辑ID、缺元数据提前失败测试 |
| wheel包发现 | `pyproject.toml` | TOML与src/openpi映射检查；实际build待环境 |
| 优势公式重复/边界 | `label_advantage_from_vlm.py` | 生产共用函数，理想Value残差、终止bootstrap测试 |
| 跨Episode误算优势 | `_extract_episode_id` | 多Episode parquet提前拒绝 |
| Value保存恢复 | `train_value.py` | 优化器状态、实际完成step、旧格式与同目录train/val检查 |
| SFT/RECAP路径配置 | 新增 `scripts/libero_pipeline.py` | norm/train/serve共享配置，本地schema、权重、资产ID、dropout测试 |
| 状态输入开关 | 同上 `--discrete-state-input` | 默认保留原配置，可显式切换供后续对照 |

已运行：`python scripts/verify_source.py`，72个Python AST通过；`python -m unittest discover -s tests -v`，21项通过；`git diff --check`通过。新入口train/serve的dry-run通过。测试不加载真实模型；LeRobot/Orbax边界采用隔离测试。

## 1. 先验收环境和数据接口

在目标训练环境记录Python版本与 `jax/flax/torch/lerobot/gemma/orbax-checkpoint` 安装版本，以及Git SHA、GPU可见范围。只使用已经确认的GPU预算，先单卡。LIBERO客户端与训练环境分开；不要同时升级两个环境。

先确定五个真实路径（以下为示例路径）：

- Demo：`/datasets/libero_demo`，包含meta/info.json、meta/tasks.jsonl、逐Episode parquet与图像/视频。
- 混合并已标注数据：`/datasets/libero_mixed_labeled`。
- 基础模型：`/weights/pi05_base/params/`。
- Value基础模型/tokenizer：按当前weight_loader实际路径与OPENPI_VLM_CKPT_DIR检查。
- 输出：项目 `assets/` 和 `checkpoints/`；运行前确认空间、目录归属和实验名。

新入口的 `--init-checkpoint` 传入包含params/的父目录，不传params/本身。`--repo-id`是逻辑标识，和checkpoint中归一化资产ID一致。

先打印计划，不读取上述路径：

```bash
python scripts/libero_pipeline.py train --stage sft \
  --dataset-root /datasets/libero_demo --repo-id local/libero_demo \
  --init-checkpoint /weights/pi05_base --exp-name sft_smoke --dry-run
```

验收：stage、dataset_root、init_params、batch、步数、状态token开关、seed均正确。新入口默认batch4/5step、关闭wandb且不覆盖已有实验；这只是功能验证配置，不能直接作为正式训练预算。

## 2. 归一化与最小SFT

只在依赖/模型资源已就绪且获得计算授权后执行：

```bash
python scripts/libero_pipeline.py norm --stage sft \
  --dataset-root /datasets/libero_demo --repo-id local/libero_demo
```

预期资产：`assets/pi05_star_libero_sft/local/libero_demo/norm_stats.json`。抽查state/action维数、mean/std/quantile是否有限；禁止不同机器人或不同动作单位共用统计量。所有统计覆盖范围应记录；max-frames若用于子采样必须大于等于batch-size。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/libero_pipeline.py train --stage sft \
  --dataset-root /datasets/libero_demo --repo-id local/libero_demo \
  --init-checkpoint /weights/pi05_base --exp-name sft_smoke \
  --batch-size 4 --num-train-steps 5 --save-interval 5
```

验收：loss/grad_norm有限，参数更新，checkpoint含params和assets，重新加载后同输入同噪声的动作一致。显存不够先减batch，不扩GPU数量；若仍不足，记录实际峰值并再决定冻结/LoRA方案，不假定LoRA自动生效。JIT首次编译时间与稳态step时间分开记。

注意现有JAX训练器保存目录可能按循环索引命名；请实际列出产生的step目录，不能推定一定叫5。Value训练器的完成step编号已单独修复。

## 3. 推理与一个完整LIBERO Episode

将下面的step_dir替换为真实目录：

```bash
python scripts/libero_pipeline.py serve --stage sft \
  --repo-id local/libero_demo --checkpoint /actual/sft/step_dir --port 8000
```

服务端会检查checkpoint的params和assets/<repo-id>/norm_stats.json；推理关闭优势条件dropout。若训练用了 `--discrete-state-input`，serve也必须加同一参数。

在已配置的LIBERO客户端环境按既有main.py执行task0/seed7/1trial；服务端与客户端都记录版本。先核对两相机翻转、state8维、输出7维、chunk15与replan5；再检查Episode结束、视频和导出的图像/动作是否同一时刻。单Episode只用于接口验收。

## 4. Value与Advantage

数据先按Episode划分train/val，再生成各自根目录；不要随机按帧切分相邻轨迹。建议先固定20%左右Episode作为验证，按任务/成败分层，保存episode清单；正式比例由实际样本量决定。目录不同仍可能含同一轨迹，需比较episode清单/内容哈希；本轮同目录检查不能替代去重。

首次功能验证：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_value.py \
  --data_dir /datasets/value_train --val_data_dir /datasets/value_val \
  --checkpoint_dir /checkpoints/value_smoke \
  --batch_size 4 --num_train_steps 5 --save_interval 5 \
  --log_interval 1 --val_interval 5 --num_workers 0 \
  --wandb_mode disabled --load_pretrained \
  --tokenizer_path /weights/vlm_ckpt/tokenizer.model
```

验收：预训练参数匹配、train/val CE有限、输出Value位于[-1,0]、step_00000005可恢复。续训到第6步核对optimizer计数与实际step；真实Orbax恢复仍待该环境验证。旧Value checkpoint缺optimizer时不能直接走完整resume。数据迭代器位置尚未保存；不能把当前续训等同于精确重放。

在独立验证Episode导出时间t、value_label、V_t、reward_label、A_t，至少画成功/失败各若干条曲线：

- Value看误差、排序和任务间标度；不能只看train loss。
- Advantage手算首帧、中间帧、最后N帧；终止bootstrap=0。
- 统计positive/negative/none与intervention，并按任务和成败拆开；全局top30%不保证每个任务30%，并列值还会改变比例。
- 当前只接受逐Episode parquet；检测到多Episode shard会提前停止，避免跨轨迹求和。

标注脚本会写回数据，先准备独立副本；不要在唯一原始数据上试跑。

## 5. 独立RECAP后训练

先对标注数据运行 `norm --stage recap`，repo-id使用 `local/libero_mixed_labeled`，然后：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/libero_pipeline.py train --stage recap \
  --dataset-root /datasets/libero_mixed_labeled --repo-id local/libero_mixed_labeled \
  --init-checkpoint /actual/sft/step_dir --exp-name recap_smoke \
  --batch-size 4 --num-train-steps 5 --save-interval 5
```

此时初始化指向SFT参数，不回到基础模型；资产与输出使用 `pi05_star_libero_recap` 独立名称。推理使用同一repo-id、stage recap与对应checkpoint。训练开始前还需实际读回一batch，确认没有none标签被当作正式优势条件，demo/rollout比例符合记录。

## 6. 需要数据/算力后才能解决的实验问题

| 问题 | 具体实施 | 验收与证据 |
|---|---|---|
| 状态条件是否有帮助 | 只改变状态token开关，训练/推理一致；固定其余配置和预算 | 同任务/初态逐Episode结果；至少多seed后才解释效果 |
| 优势条件作用 | 对照无条件、真实优势标签、随机打乱标签；保持数据与优化预算一致 | 固定任务/seed，成功数、区间、失败视频 |
| Value有效性 | 按Episode独立验证，先画标签与预测，再测排序/优势分布 | held-out误差、曲线、任务分组统计 |
| 相机mask是否正确 | 在已安装Flax中核对cross-attention mask参数与shape，构造被mask图像扰动测试 | masked输入变化不应影响输出；需先核对真实API，不盲改权重结构 |
| 全链训练稳定性 | 先5step，再按批准预算扩大；逐步检查loss、grad、显存与读盘 | 一份完整运行配置、checkpoint与可复现命令 |

每次只改变一项。正式对照数量/训练seed/steps另定预算；本轮没有替你启动这些实验。
