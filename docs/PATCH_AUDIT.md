# Destiny 的 15 项修复：对 PiStar 的实际审计

> **状态说明（2026-11-09）：** 此表是早期源码兼容性审计。末尾“待核验”描述的是当时的代码审查阶段，不代表整个项目仍未完成；维护者确认后续已完成完整实验。最新进度见 [PROJECT_STATE.md](PROJECT_STATE.md)，本次未改动原补丁细节或执行训练。

锁定版本：PiStar `d7b6ebd` / Destiny `a44350b`。当前表仅为代码对照，不代表 GPU 验收。

| # | 上游报告的问题 | 本项目处理 |
|---|---|---|
| 1 | ValueModelWeightLoader 缺失 | PiStar 已有，保留其实现并处理硬编码权重路径 |
| 2 | gemma.gm.data 缺失 | PiStar 当前树中已有；不重复复制 |
| 3 | kauldron.ktyping 导入问题 | 先检查 Gemma 版本再决定 |
| 4 | etils.edc.ContextStack 顶层导入 | 使用兼容补丁删除不兼容导入 |
| 5 | ContextStack 备用实现 | 保留独立 list 类型 |
| 6 | 缺少 openpi.shared.console | 纳入 Destiny 兼容辅助模块 |
| 7 | 缺少 openpi.shared.progress | 纳入 Destiny 兼容辅助模块 |
| 8 | local_data_dir 不受 DataConfig 支持 | 改用 repo_id |
| 9 | 数据集缺少 actions 字段 | Value 流程禁用动作序列采样 |
| 10 | create_value_data_loader 不存在 | PiStar 已有，不覆盖 |
| 11 | Value 数据加载器缺少长度和 dataset | PiStar 已有独立 ValueDataLoader |
| 12 | TrainState 不是 JAX PyTree | 修改为 flax.struct.PyTreeNode |
| 13 | Value Loader 错取 actions / Tokenizer 不兼容 | 保留专用 ValueDataLoader；修正 Tokenizer 参数 |
| 14 | Stage 5 local_data_dir 异常 | 改用 repo_id 并禁用动作序列采样 |
| 15 | Stage 5 Tokenizer 不接受附加参数 | 兼容并忽略附加参数 |

**待核验**：完整依赖版本、Gemma/etils/Kauldron 的导入、LeRobot 数据目录解析、VLM 参数树、数值训练与评测。逐项检查后才能将“计划修复”改为“验证完成”。

原始修复说明：https://github.com/Destiny000621/RECAP#upstream-patch-reference