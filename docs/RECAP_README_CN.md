# Destiny000621/RECAP：与 LIBERO 相关的兼容性修复中文整理

> 基于 [Destiny000621/RECAP](https://github.com/Destiny000621/RECAP) 的 README，固定提交 `a44350b4a99f1d7f6137635ab3e894a5f72f5bda`。原作者实际训练主线是 YAM 双臂真机；本文只提炼可以迁移至 LIBERO 的机制及程序修复，**不复现其真机运行内容**。

## Value / Advantage / RECAP 训练主线

### Stage 3：策略从 SFT 继续训练

在 SFT 模型基础上增加优势条件输入 `adv_ind`。此时 Value Model 尚未参与，属于训练条件通道的阶段性检查，不应直接称为完整 RECAP。

### Stage 4：训练 VLM Value Model

使用 SigLIP-So400M + Gemma3-270M 和分布式价值预测头（201 个 bins）预测状态价值，监督目标是轨迹中的 `value_label`。

可选的冻结方案：

- `all_backbones`：冻结图像与语言主干，主要训练价值相关模块。
- `siglip_only`：保持视觉编码器冻结，使语言侧可更新。
- `none`：训练全部相关模块。

绝对训练损失并不能直接证明优势排序质量，必须进一步检查预测 Value 的可用性。

### Stage 5：优势条件标注

从 Value 预测值计算 N-step Advantage，仅在同一 Episode 内累计奖励并 Bootstrap。按自主帧上的优势分位数划分 `positive` / `negative`；真实示范帧和干预帧按约定保留。

**这个步骤可能直接修改 LeRobot 数据集，必须在副本上执行。**

### Stage 6：优势条件策略后训练

使用重新标注的 `adv_ind` 继续训练 π₀.₅ / PiStar 策略，推理阶段通常固定正向条件。与 Stage 3 或 SFT 对比时必须使用一致的任务、随机种子、评测 Episode 数与 Checkpoint 来源。

## 15 项修复的核心

作者列出 ValueModelWeightLoader、Gemma、ContextStack、console/progress、DataConfig、ValueDataLoader、TrainState、Tokenizer 等兼容性问题。本项目逐项比对当前 PiStar，而不是用 Destiny 整个仓库覆盖。详细状态见 [PATCH_AUDIT.md](PATCH_AUDIT.md)。

## 注意事项

- 原作者 YAM 的相机、关节状态与 LeRobot 字段映射不能直接搬进 LIBERO。
- `action_sequence_keys=()` 只应用在不需要预测动作序列的 Value/Advantage 入口。
- `repo_id=local/<dataset>` 需要真实的 LeRobot 本地缓存映射。
- 模型权重、Gemma/etils/Kauldron 和 protobuf 仍应在正式环境中核验版本。
- 原作者没有提供可直接照搬的 LIBERO SFT→RECAP 全套成功率数值；本项目不得捏造。
