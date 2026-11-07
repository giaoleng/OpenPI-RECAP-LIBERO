# OpenPI-RECAP-Sim 项目约定

## 范围

以 PiStar 的 π₀.₅ + LIBERO 仿真为主体，参考 Destiny000621/RECAP 对 Value / Advantage 的兼容性修复。禁止引入 YAM/Piper/RealMan 的真机控制和未经验证的实验成功率。

## 开发与提交

- 保留 Python 函数、类、参数、数据字段英文标识；README、项目文档和核心逻辑注释优先采用中文。
- 上游 Apache-2.0 版权与许可证不得删除；第三方代码保留来源记录。
- 代码修改应有可复查的 Git Diff；每个阶段在 `docs/PROJECT_STATE.md` 中更新已验证和待验证工作。
- 运行 `python3 scripts/verify_source.py` 进行无需依赖的语法检查；这不等于 GPU 推理/训练通过。
- 不上传数据集、Checkpoint、日志、密钥；所有 π₀.₅ 模型和重负载任务都在远程环境运行。
- 真正的成功率必须记录任务 ID、Episode 数、种子、Checkpoint 及原始日志；引用他人的数字必须显式标注原作者。
