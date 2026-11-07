# 中文导读：进度条显示兼容接口。
# 上游：Destiny000621/RECAP；保持算法及公共接口，选择性修复兼容问题。
"""为训练进度条提供兼容的配色接口，不改变 tqdm 行为。"""

from __future__ import annotations

from typing import Any


def sync_pbar_color(pbar: Any) -> None:  # noqa: ARG001
    """No-op stub. Upstream pistar's intent was probably to recolor a tqdm
    progress bar based on training state, but the implementation isn't
    shipped on main. A no-op is safe — pbars continue to render normally."""
