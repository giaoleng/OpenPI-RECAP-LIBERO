"""修复 HuggingFace datasets 对旧式 List 字段元数据的兼容性问题。

部分历史 LeRobot Parquet 的 Arrow Schema 将列表标记为 List，
而运行时所装的 datasets 版本可能只识别 Sequence。
仅在尚未注册 List 特征类型时应用兼容处理。
"""

import json

try:
    from datasets.features.features import Features, _FEATURE_TYPES
except ImportError:
    # 没有安装 datasets 时由真正使用数据集的入口报告缺失依赖。
    Features = None

if Features is not None and "List" not in _FEATURE_TYPES:
    _FEATURE_TYPES["List"] = _FEATURE_TYPES["Sequence"]
    _original_from_arrow_schema = Features.from_arrow_schema

    def _from_arrow_schema_compat(schema):
        """在读取旧 Parquet 时，将 List 兼容映射为 Sequence。"""
        metadata = getattr(schema, "metadata", None)
        if metadata and b"info" in metadata:
            try:
                info = json.loads(metadata[b"info"])

                def visit(value):
                    if isinstance(value, dict):
                        if value.get("_type") == "List":
                            value["_type"] = "Sequence"
                        for sub in value.values():
                            visit(sub)
                    elif isinstance(value, list):
                        for sub in value:
                            visit(sub)

                visit(info.get("features", {}))
                patched = dict(metadata)
                patched[b"info"] = json.dumps(info).encode()
                schema = schema.with_metadata(patched)
            except (TypeError, ValueError, AttributeError):
                pass
        return _original_from_arrow_schema(schema)

    Features.from_arrow_schema = _from_arrow_schema_compat
