# 训练与数据处理脚本

本目录组织从专家示范到 RECAP 风格再训练的可执行入口。

| 脚本 | 独立职责 |
| --- | --- |
| `train.py` | 根据训练配置执行策略 SFT/条件微调 |
| `train_value.py` | 基于 `value_label` 优化 VLM Value Model |
| `label_advantage_from_vlm.py` | Value 推理、N-step Advantage 与 `adv_ind` 重标注 |
| `check_value_data.py` | 训练前检查 Value batch 的区间、NaN/Inf、图像张量 |
| `merge_datasets.py` | 合并已经对齐 schema 的示范和 Rollout |
| `compute_norm_stats.py` | 计算训练所需的数据归一化统计 |
| `serve_policy.py` | 启动策略模型服务端 |
| `filter_success_episodes.py` | 按成功条件筛选 Episode |
| `add_value_labels.py` | 构造 Value 学习所需监督字段 |
| `verify_source.py` | 源码结构/静态检查辅助 |

运行前先核对 `--help` 中的参数、数据目录和对应的训练配置；涉及覆盖参数时先备份现有输出。

推荐排查路径：**先检查数据 batch，再训练 Value，再进行 Advantage 标注**。参数与步骤见 [完整操作手册](../README.md)。脚本文件存在并不等于依赖和端到端训练已经通过。

