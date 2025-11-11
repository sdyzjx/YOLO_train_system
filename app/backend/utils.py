from __future__ import annotations

from typing import Dict


def parse_key_value_text(text: str) -> Dict[str, str]:
    """将用户输入的 key=value 文本解析为字典。"""
    result: Dict[str, str] = {}
    if not text:
        return result
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            result[key] = value
    return result
