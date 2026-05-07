from __future__ import annotations

import os
from pathlib import Path


def _env_path(key: str) -> Path | None:
    v = os.getenv(key)
    if not v:
        return None
    return Path(v).expanduser()


# 数据文件路径：优先读环境变量（便于不同机器/不提交数据到仓库）
# - COMMODITY_ROTATION_WORKBOOK: 支持传绝对路径或相对路径
WORKBOOK_PATH: Path = _env_path("COMMODITY_ROTATION_WORKBOOK") or Path("大宗商品轮动_数据2.xlsx")

