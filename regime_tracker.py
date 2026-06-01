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
# 基于新ETF/LOF价格（512400/石油LOF/煤炭LOF）重新标定，2016-2026样本
# 各Cluster等权年化收益: C3=+37.7% C5=+31.6% C0=+30.3% C2=+10.4% C1=-9.0% C4=-13.2%
# position  : 该Cluster下的基准总仓位比例
# mom_weight: 动量在综合得分中的权重（基本面权重 = 1 - mom_weight）
CLUSTER_CONFIG: dict[int, dict] = {
    0: dict(name="美国过热·通胀扩张",   position=0.85, mom_weight=0.10),
    1: dict(name="经济见顶·高利率承压", position=0.20, mom_weight=0.10),
    2: dict(name="增长预期走弱·利率下行", position=0.50, mom_weight=0.10),
    3: dict(name="全球需求共振·商品牛市", position=1.00, mom_weight=0.10),
    4: dict(name="激进加息·流动性收缩", position=0.00, mom_weight=0.05),
    5: dict(name="衰退预期·货币宽松",   position=0.90, mom_weight=0.10),
}

# C4（全面熊市/衰退）无特殊资产约束，直接空仓
CLUSTER5_FLOOR = {}
CLUSTER5_CAP   = {}


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
    当前各Cluster均无强制约束，直接返回原始 target。
    """
    return target
