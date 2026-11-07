# 项目状态（2026-11-06）

本项目基于 PiStar `d7b6ebdf41fdfeb368ef157965c1c41b7c056ddd`，参考 Destiny000621/RECAP `a44350b4a99f1d7f6137635ab3e894a5f72f5bda` 的兼容性修复，构建**仅 LIBERO 仿真**的 π₀.₅ SFT、Rollout、Value、Advantage 与 RECAP 复现。

**已完成**：创建 GitHub 仓库、中文项目入口，核对上游关键文件，建立修复审计与代码整合分支。

**进行中**：导入并中文化相关源码；修复 Stage 4/5 的接口兼容。

**未完成验收**：依赖安装、GPU 权重加载、完整 LIBERO Episode、SFT、Value Model 与 RECAP 训练，暂无本项目自行测得的成功率。不得将原作者结果记为本仓库结果。

后续按“静态检查 → 远程 GPU 推理 → Value/Advantage 端到端测试 → SFT/RECAP 对照”推进。