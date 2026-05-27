# -*- coding: utf-8 -*-
"""
Regime Tracker — 宏观周期识别与仓位控制模块

每次调用 get_regime_series() 即可获得完整历史时序的：
  - cluster   : 0-5 Cluster编号
  - position  : 总仓位比例
  - mom_weight: 动量权重（对应 app.py 的 mom_w）
  - ic_weight : IC基本面权重（= 1 - mom_weight）
  - name      : Cluster中文名称
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_analysis import build_regime_features, PCA_FEATURES

_CSV_LABELS = Path(__file__).parent / "regime_outputs" / "regime_labels_pca_kmeans.csv"

# ── Cluster 配置 ─────────────────────────────────────────────────────────────
# position  : 该Cluster下的基准总仓位比例
# mom_weight: 动量在综合得分中的权重（IC权重 = 1 - mom_weight）
CLUSTER_CONFIG: dict[int, dict] = {
    0: dict(name="能源/通胀牛市",   position=0.85, mom_weight=0.00),
    1: dict(name="高增长横盘期",    position=0.45, mom_weight=0.30),
    2: dict(name="加息紧缩期",     position=0.00, mom_weight=0.05),
    3: dict(name="大宗全面牛市",   position=1.00, mom_weight=0.10),
    4: dict(name="温和复苏期",     position=0.65, mom_weight=0.10),
    5: dict(name="衰退/通缩/避险", position=0.20, mom_weight=0.10),
}

# C5（衰退/通缩/避险）状态下资产权重的强制约束（黄金保底，工业品封顶）
CLUSTER5_FLOOR = {"黄金": 0.60}
CLUSTER5_CAP   = {"铜": 0.10, "原油": 0.10, "煤炭": 0.10}


def get_regime_series(
    factors: dict,
    prices: pd.DataFrame,
    n_components: int = 3,
    k: int = 6,
) -> pd.DataFrame:
    """
    计算每周的宏观 Cluster 及对应配置。

    Parameters
    ----------
    factors   : load_weekly_factors() 返回的因子字典
    prices    : load_weekly_prices() 返回的价格 DataFrame
    n_components : PCA 主成分数
    k         : KMeans Cluster 数

    Returns
    -------
    DataFrame，index=日期，columns=[cluster, position, mom_weight, ic_weight, name]
    """
    common_idx = prices.index
    feat_df = build_regime_features(factors, common_idx, weekly_prices=prices)

    # 直接读取 regime_analysis 生成的 CSV，避免重跑 KMeans 导致结果不稳定
    csv_labels = pd.read_csv(_CSV_LABELS, index_col=0, parse_dates=True).iloc[:, 0]
    # 对齐到完整时间轴：CSV 覆盖范围内用 CSV 值，更新的日期前向填充最后已知标签
    labels = csv_labels.reindex(common_idx).ffill().bfill()

    # C2（加息紧缩）+ 降息方向 → 自动升级为 C3（大宗全面牛市）
    # us10y_delta < 0 表示12周利率变化方向为负（降息/宽松周期）
    if "us10y_delta" in feat_df.columns:
        rate_dir = feat_df["us10y_delta"].reindex(common_idx).ffill().bfill()
        labels = labels.where(~((labels == 2) & (rate_dir < 0)), other=3)

    result = pd.DataFrame(index=common_idx)
    result["cluster"]    = labels.astype(int)
    result["position"]   = labels.map({c: v["position"]   for c, v in CLUSTER_CONFIG.items()})
    result["mom_weight"] = labels.map({c: v["mom_weight"]  for c, v in CLUSTER_CONFIG.items()})
    result["ic_weight"]  = 1.0 - result["mom_weight"]
    result["name"]       = labels.map({c: v["name"]        for c, v in CLUSTER_CONFIG.items()})

    return result


def apply_asset_constraints(
    target: pd.Series,
    cluster: int,
    assets: list[str],
) -> pd.Series:
    """
    对给定 Cluster 的目标权重应用资产层面约束。
    仅 C5（衰退/通缩/避险）有约束：黄金保底 60%，工业品上限 10%。
    其他 Cluster 直接返回原始 target。
    """
    if cluster != 5:
        return target

    t = target.copy()
    total = t.sum()
    if total <= 0:
        return t

    # 1. 先对工业品执行上限
    for asset, cap in CLUSTER5_CAP.items():
        if asset in t.index:
            t[asset] = min(t[asset], cap * total)

    # 2. 将超出部分加到黄金
    if "黄金" in t.index:
        t["黄金"] = max(t.get("黄金", 0.0), CLUSTER5_FLOOR["黄金"] * total)

    # 3. 重新归一化（保持总权重不变）
    new_total = t.sum()
    if new_total > 0:
        t = t / new_total * total

    return t
