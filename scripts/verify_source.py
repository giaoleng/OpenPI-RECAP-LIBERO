#!/usr/bin/env python3
"""无需 GPU 的源码完整性和语法检查，不执行模型加载。"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUST_EXIST = (
    "src/openpi/models/pi0.py",
    "src/openpi/models/value_model.py",
    "src/openpi/training/config.py",
    "src/openpi/training/data_loader.py",
    "src/openpi/shared/console.py",
    "src/openpi/shared/progress.py",
    "src/openpi/shared/hf_datasets_compat.py",
    "scripts/train.py",
    "scripts/train_value.py",
    "scripts/label_advantage_from_vlm.py",
    "examples/libero/main.py",
)

def main():
    for relative in MUST_EXIST:
        assert (ROOT / relative).is_file(), f"缺少模块: {relative}"
    all_sources = [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/**/*.py"),
                   *ROOT.glob("examples/libero/**/*.py"), *ROOT.glob("packages/openpi-client/**/*.py")]
    for path in all_sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not (ROOT / "control_your_robot").exists(), "不应纳入真机代码目录"
    print(f"静态语法检查通过：{len(all_sources)} 个 Python 文件。")
    print("注意：未检查依赖导入、GPU 模型加载或实际 LIBERO 成功率。")

if __name__ == "__main__":
    main()
