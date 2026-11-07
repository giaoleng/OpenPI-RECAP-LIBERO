# 中文化与源码整合

## 来源

- 主体：[ybpy/pistar](https://github.com/ybpy/pistar)，其源自 Physical-Intelligence/openpi。
- 修复参考：[Destiny000621/RECAP](https://github.com/Destiny000621/RECAP)，重点 Stage 4 / 5。

## 范围

只纳入 LIBERO 仿真、π₀.₅ 模型/训练、Rollout、Value Model、Advantage 标注及 RECAP 后训练。排除 `control_your_robot/`、YAM/Piper/RealMan 真机控制、遥操作、真机部署相关代码和结果。

代码标识符、参数、数据字段名不翻译，避免破坏兼容；项目说明、主要脚本的模块描述与关键注释改写为中文。第三方许可证和版权头原样保留。

源码移植与注释翻译不代表实际运行通过；运行时兼容须另行验收。