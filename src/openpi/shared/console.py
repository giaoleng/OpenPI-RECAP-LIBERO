# 中文导读：训练日志显示兼容接口。
# 上游：Destiny000621/RECAP；保持算法及公共接口，选择性修复兼容问题。
"""为 Value 训练和 Advantage 标注提供彩色日志输出工具。\n来源：Destiny000621/RECAP 的兼容性修复。"""

from __future__ import annotations

# ANSI SGR codes
_RESET = "\033[0m"
_BLUE = "\033[94m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BOLD = "\033[1m"


def info(msg: object) -> str:
    return f"{_BLUE}ℹ {msg}{_RESET}"


def ok(msg: object) -> str:
    return f"{_GREEN}✓ {msg}{_RESET}"


def warn(msg: object) -> str:
    return f"{_YELLOW}⚠ {msg}{_RESET}"


def error(msg: object) -> str:
    return f"{_RED}✗ {msg}{_RESET}"


def bold(msg: object) -> str:
    return f"{_BOLD}{msg}{_RESET}"
