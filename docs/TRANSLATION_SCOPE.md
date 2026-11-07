# 中文化覆盖范围与未完成项

## 已中文化

- 仓库入口 README。
- LIBERO 专用评测 README。
- 两个上游项目的 LIBERO / RECAP 相关中文整理。
- 核心模型、训练、数据脚本的中文模块导读及若干关键注释/docstring。
- 上游修复审计、进度和实验结果标注。

## 未完成

- PiStar 底层模型文件仍有部分原始英文注释和 docstring。
- PyTorch/Transformer 的第三方模型实现尚未逐行翻译。
- 运行时 Gemma 包采用其第三方原始许可证，未重新发布或全文翻译。
- 尚未通过端到端环境运行验证。

术语遵循：代码标识、`adv_ind`、`value_label`、`TrainConfig`、`Flow Matching` 等技术字段保持英文，解释文字改用中文。未完成的部分不得被描述为“全部注释中文化完毕”。
