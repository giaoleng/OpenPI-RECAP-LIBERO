"""在导入 Gemma 前处理旧 API 兼容性。

Gemma 3.3.0 的部分模块依赖 etils.edc.ContextStack。
较新 etils 可能不再提供这一类型；使用最小 list 接口作为回退。
"""

def ensure_context_stack() -> None:
    """为缺失的 etils ContextStack 提供兼容实现。"""
    from etils import edc

    if hasattr(edc, "ContextStack"):
        return

    class ContextStack(list):
        @property
        def stack(self):
            return self

    edc.ContextStack = ContextStack
