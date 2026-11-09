# 项目状态（2026-11-06）

本项目基于 PiStar `d7b6ebdf41fdfeb368ef157965c1c41b7c056ddd`，参考 Destiny000621/RECAP `a44350b4a99f1d7f6137635ab3e894a5f72f5bda` 的兼容性修复，构建**仅 LIBERO 仿真**的 π₀.₅ SFT、Rollout、Value、Advantage 与 RECAP 复现。

**已完成**：创建 GitHub 仓库、中文项目入口，核对上游关键文件，建立修复审计与代码整合分支。

**进行中**：导入并中文化相关源码；修复 Stage 4/5 的接口兼容。

**未完成验收**：依赖安装、GPU 权重加载、完整 LIBERO Episode、SFT、Value Model 与 RECAP 训练，暂无本项目自行测得的成功率。不得将原作者结果记为本仓库结果。

后续按“静态检查 → 远程 GPU 推理 → Value/Advantage 端到端测试 → SFT/RECAP 对照”推进。

## 2026-11-08 中文提交规范整理

已核对当前GitHub main与本地审查基线一致，并记录主要路径最新提交。新增CONTRIBUTING.md、.gitmessage和docs/COMMIT_STYLE_AUDIT.md，采用type前缀与简洁中文技术说明；local commit.template指向.gitmessage。本轮另修正README中旧项目名称及克隆/工作路径，共五处。本轮只创建这些规范与文档修改的本地提交，未提交先前阶段的算法、配置或测试工作，未改旧历史、许可证或版权，未推送。未改动目录继续保留旧标题；若要求现有旧标题全部换名，需要另行审批历史整理。详细路径映射与影响见审查文档。
