
# -*- coding: utf-8 -*-
"""
宏观周期 Regime 分析模块

功能：
1. 构造宏观状态特征（增长/通胀的水平与方向、风险偏好、美元强弱）
2. KMeans 聚类划分 regime（K=2~8 自动选最优，Tracker 固定 K=6）
3. 增长/通胀四象限划分（确定性对照）
4. 输出：regime x 资产表现、regime x 因子IC、regime 切换期分析

运行方式:
    python3 regime_analysis.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from rotation_model import (
    ASSETS,
    load_weekly_factors,
    load_weekly_prices,
)
from config import WORKBOOK_PATH

OUTPUT_DIR = Path("regime_outputs")

# ══════════════════════════════════════════════
# 1. 构造宏观状态特征
# ══════════════════════════════════════════════

def _rolling_z(s: pd.Series, window: int = 52) -> pd.Series:
    """滚动 z-score：当前值相对过去 window 周的位置。"""
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std(ddof=0)
    return ((s - mu) / sd.replace(0, np.nan)).fillna(0.0)


def _delta_z(s: pd.Series, diff_period: int = 12, z_window: int = 52) -> pd.Series:
    """方向 z-score：先取 diff_period 周绝对差分，再做滚动标准化。
    diff_period=12（约3个月）覆盖2~3次月频数据发布，能捕捉趋势而非单次跳变。
    """
    d = s.diff(diff_period)
    return _rolling_z(d, z_window)


def _align_factor(factors: Dict[str, pd.Series], key: str, idx: pd.DatetimeIndex) -> pd.Series:
    """将因子对齐到交易日历并前值填充。"""
    s = factors.get(key, pd.Series(dtype=float))
    if s.empty:
        return pd.Series(0.0, index=idx)
    return s.reindex(idx).ffill().fillna(0.0)


def build_regime_features(
    factors: Dict[str, pd.Series],
    common_idx: pd.DatetimeIndex,
    weekly_prices: pd.DataFrame | None = None,
    z_window: int = 52,
    delta_window: int = 12,
) -> pd.DataFrame:
    """构造宏观状态特征矩阵（每周一行）。

    特征分两层：
    - level（水平）：当前值在历史上处于什么分位（52周滚动z-score）
    - delta（方向）：近12周变化量的z-score，>0 表示改善速度高于历史均值

    新增维度：
    - real_rate_level: 实际利率水平（货币紧缩/宽松）
    - cn_pmi_delta: 中国增长方向（铜/煤炭的核心驱动）
    - commodity_mom: 5品种等权12周动量的z-score（市场自身趋势）
    """
    pmi = _align_factor(factors, "pmi", common_idx)
    ppi = _align_factor(factors, "ppi", common_idx)
    vix = _align_factor(factors, "vix", common_idx)
    dxy = _align_factor(factors, "dxy", common_idx)
    cn_pmi = _align_factor(factors, "cn_pmi", common_idx)
    cn_ppi = _align_factor(factors, "cn_ppi", common_idx)
    real_rate = _align_factor(factors, "real_rate", common_idx)
    us10y = _align_factor(factors, "us10y", common_idx)
    yield_spread = _align_factor(factors, "yield_spread", common_idx)
    credit_impulse = _align_factor(factors, "credit_impulse", common_idx)

    result = {
        "pmi_level": _rolling_z(pmi, z_window),
        "pmi_delta": _delta_z(pmi, delta_window, z_window),
        "ppi_level": _rolling_z(ppi, z_window),
        "ppi_delta": _delta_z(ppi, delta_window, z_window),
        "vix_level": _rolling_z(vix, z_window),
        "dxy_level": _rolling_z(dxy, z_window),
        "real_rate_level": _rolling_z(real_rate, z_window),
        "cn_pmi_level": _rolling_z(cn_pmi, z_window),
        "cn_pmi_delta": _delta_z(cn_pmi, delta_window, z_window),
        "cn_ppi_level": _rolling_z(cn_ppi, z_window),
        "cn_ppi_delta": _delta_z(cn_ppi, delta_window, z_window),
        # 新增因子
        "us10y_level": _rolling_z(us10y, z_window),
        "us10y_delta": _delta_z(us10y, delta_window, z_window),
        "yield_spread_level": _rolling_z(yield_spread, z_window),
        "yield_spread_delta": _delta_z(yield_spread, delta_window, z_window),
        "credit_impulse_level": _rolling_z(credit_impulse, z_window),
        "credit_impulse_delta": _delta_z(credit_impulse, delta_window, z_window),
    }

    if weekly_prices is not None:
        from rotation_model import ASSETS
        px = weekly_prices.reindex(columns=ASSETS)
        eq_mom = px.pct_change(delta_window).mean(axis=1)
        result["commodity_mom"] = _rolling_z(eq_mom, z_window)

    return pd.DataFrame(result, index=common_idx)


# ══════════════════════════════════════════════
# 2. KMeans 聚类
# ══════════════════════════════════════════════

KMEANS_FEATURES = [
    "pmi_level", "pmi_delta",
    "ppi_level", "ppi_delta",
    "vix_level", "dxy_level",
]


def classify_kmeans(
    features_df: pd.DataFrame,
    k_range: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
    random_state: int = 42,
    output_dir: Path | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """KMeans 聚类划分 regime。

    Returns:
        labels: 每周的 regime 标签（按 pmi_level 均值从高到低排序：0=最强，K-1=最弱）
        centers: 聚类中心表（便于解读/命名）
        silhouette_df: 各 K 的轮廓系数 + inertia 对比
    """
    valid = features_df[KMEANS_FEATURES].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(valid)

    results: list[dict] = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labs = km.fit_predict(X)
        sil = silhouette_score(X, labs)
        results.append({
            "k": k,
            "silhouette": sil,
            "inertia": km.inertia_,
            "model": km,
            "labels": labs,
        })

    silhouette_df = pd.DataFrame([
        {"k": r["k"], "silhouette": r["silhouette"], "inertia": r["inertia"]}
        for r in results
    ])

    # 画肘部图 + 轮廓系数图
    if output_dir is not None:
        _plot_elbow_silhouette(silhouette_df, output_dir)

    best = max(results, key=lambda r: r["silhouette"])
    km = best["model"]
    raw_labels = best["labels"]
    best_k = best["k"]

    center_pmi = pd.Series(km.cluster_centers_[:, 0])
    rank_map = center_pmi.rank(ascending=False).astype(int) - 1
    label_remap = {old: int(new) for old, new in rank_map.items()}

    stable_labels = pd.Series(raw_labels, index=valid.index, name="regime_kmeans")
    stable_labels = stable_labels.map(label_remap)

    centers = pd.DataFrame(km.cluster_centers_, columns=KMEANS_FEATURES)
    centers.index = [label_remap[i] for i in range(len(centers))]
    centers = centers.sort_index()
    centers.index.name = "cluster"

    print(f"KMeans 最优 K={best_k}（轮廓系数={best['silhouette']:.3f}）")
    return stable_labels, centers, silhouette_df


def _plot_elbow_silhouette(sil_df: pd.DataFrame, output_dir: Path) -> None:
    """画肘部图（Inertia）和轮廓系数图，保存为 PNG。"""
    plt.rcParams["font.family"] = ["Arial Unicode MS", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ks = sil_df["k"].values

    # 左图：肘部图（Inertia / SSE）
    ax1.plot(ks, sil_df["inertia"].values, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("K (聚类数)", fontsize=12)
    ax1.set_ylabel("Inertia (SSE)", fontsize=12)
    ax1.set_title("肘部图 (Elbow Method)", fontsize=14)
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.3)

    # 右图：轮廓系数
    ax2.plot(ks, sil_df["silhouette"].values, "rs-", linewidth=2, markersize=8)
    best_idx = sil_df["silhouette"].idxmax()
    best_k = int(sil_df.loc[best_idx, "k"])
    best_sil = float(sil_df.loc[best_idx, "silhouette"])
    ax2.axvline(x=best_k, color="gray", linestyle="--", alpha=0.6)
    ax2.annotate(
        f"最优 K={best_k}\n({best_sil:.3f})",
        xy=(best_k, best_sil),
        xytext=(best_k + 0.8, best_sil),
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    ax2.set_xlabel("K (聚类数)", fontsize=12)
    ax2.set_ylabel("轮廓系数 (Silhouette)", fontsize=12)
    ax2.set_title("轮廓系数图", fontsize=14)
    ax2.set_xticks(ks)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "kmeans_elbow_silhouette.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"肘部图/轮廓系数图已保存: {path}")


# PCA + KMeans 用全部特征（包括新增的维度），降维后再聚类
PCA_FEATURES = [
    "pmi_level", "pmi_delta",
    "ppi_level", "ppi_delta",
    "vix_level", "dxy_level",
    "real_rate_level",
    "cn_pmi_level", "cn_pmi_delta",
    "cn_ppi_level", "cn_ppi_delta",
    "commodity_mom",
    # 新增
    "us10y_level", "us10y_delta",
    "yield_spread_level", "yield_spread_delta",
    "credit_impulse_level", "credit_impulse_delta",
]


def classify_pca_kmeans(
    features_df: pd.DataFrame,
    n_components: int = 3,
    k_range: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
    force_k: int | None = None,
    random_state: int = 42,
    output_dir: Path | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """PCA 降维 + KMeans 聚类。

    用全部 9 个特征降到 n_components 个主成分，消除冗余后再做 KMeans。
    force_k: 强制指定 K 值（不按轮廓系数自动选）。

    Returns:
        labels, centers_original_space, silhouette_df, pca_explained
    """
    valid = features_df[PCA_FEATURES].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(valid)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    pca_explained = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(n_components)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_),
    })
    loadings = pd.DataFrame(
        pca.components_.T,
        index=PCA_FEATURES,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    print(f"PCA 前 {n_components} 个主成分累计解释方差: {pca_explained['cumulative'].iloc[-1]:.1%}")
    print("主成分载荷（绝对值最大=主要驱动因子）:")
    print(loadings.round(3).to_string())

    results: list[dict] = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labs = km.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labs)
        results.append({
            "k": k, "silhouette": sil, "inertia": km.inertia_,
            "model": km, "labels": labs,
        })

    sil_df = pd.DataFrame([
        {"k": r["k"], "silhouette": r["silhouette"], "inertia": r["inertia"]}
        for r in results
    ])

    if output_dir is not None:
        _plot_elbow_silhouette_named(sil_df, output_dir, "pca_kmeans_elbow_silhouette.png", "PCA+KMeans")
        loadings.to_csv(output_dir / "pca_loadings.csv", encoding="utf-8-sig")

    if force_k is not None:
        best = next(r for r in results if r["k"] == force_k)
        print(f"PCA+KMeans 强制 K={force_k}（轮廓系数={best['silhouette']:.3f}）")
    else:
        best = max(results, key=lambda r: r["silhouette"])
        print(f"PCA+KMeans 自动选优 K={best['k']}（轮廓系数={best['silhouette']:.3f}）")

    km = best["model"]
    raw_labels = best["labels"]
    best_k = best["k"]

    pmi_col = PCA_FEATURES.index("pmi_level")
    cluster_pmi_mean = pd.Series(
        [X_scaled[raw_labels == c, pmi_col].mean() for c in range(best_k)]
    )
    rank_map = cluster_pmi_mean.rank(ascending=False).astype(int) - 1
    label_remap = {old: int(new) for old, new in rank_map.items()}

    stable_labels = pd.Series(raw_labels, index=valid.index, name="regime_pca_kmeans")
    stable_labels = stable_labels.map(label_remap)

    centers_pca = km.cluster_centers_
    centers_original = scaler.inverse_transform(pca.inverse_transform(centers_pca))
    centers_df = pd.DataFrame(centers_original, columns=PCA_FEATURES)
    centers_df.index = [label_remap[i] for i in range(len(centers_df))]
    centers_df = centers_df.sort_index()
    centers_df.index.name = "cluster"
    return stable_labels, centers_df, sil_df, pca_explained


def classify_gmm(
    features_df: pd.DataFrame,
    k_range: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
    random_state: int = 42,
    output_dir: Path | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """高斯混合模型（GMM）划分 regime。

    与 KMeans 的区别：
    - 允许椭圆形 cluster（不强制球形）
    - 输出概率（每周属于各 cluster 的概率），更适合宏观状态的模糊边界
    - 用 BIC 选最优 K（越低越好）

    Returns:
        labels, centers, metrics_df, probabilities
    """
    valid = features_df[KMEANS_FEATURES].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(valid)

    results: list[dict] = []
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
            n_init=5,
            max_iter=300,
        )
        gmm.fit(X)
        labs = gmm.predict(X)
        sil = silhouette_score(X, labs)
        results.append({
            "k": k,
            "silhouette": sil,
            "bic": gmm.bic(X),
            "aic": gmm.aic(X),
            "model": gmm,
            "labels": labs,
        })

    metrics_df = pd.DataFrame([
        {"k": r["k"], "silhouette": r["silhouette"], "bic": r["bic"], "aic": r["aic"]}
        for r in results
    ])

    if output_dir is not None:
        _plot_gmm_metrics(metrics_df, output_dir)

    # GMM 用 BIC 选最优 K（越小越好）
    best = min(results, key=lambda r: r["bic"])
    gmm = best["model"]
    raw_labels = best["labels"]
    best_k = best["k"]
    best_sil = best["silhouette"]

    # 按 pmi_level 的 cluster 均值排序
    pmi_col = KMEANS_FEATURES.index("pmi_level")
    cluster_pmi_mean = pd.Series(
        [X[raw_labels == c, pmi_col].mean() for c in range(best_k)]
    )
    rank_map = cluster_pmi_mean.rank(ascending=False).astype(int) - 1
    label_remap = {old: int(new) for old, new in rank_map.items()}

    stable_labels = pd.Series(raw_labels, index=valid.index, name="regime_gmm")
    stable_labels = stable_labels.map(label_remap)

    # 概率矩阵
    proba = gmm.predict_proba(X)
    proba_df = pd.DataFrame(proba, index=valid.index)
    proba_df.columns = [f"prob_cluster_{label_remap[i]}" for i in range(proba.shape[1])]
    proba_df = proba_df.reindex(columns=sorted(proba_df.columns))

    # 聚类中心（GMM 的 means_）
    centers = pd.DataFrame(gmm.means_, columns=KMEANS_FEATURES)
    centers.index = [label_remap[i] for i in range(len(centers))]
    centers = centers.sort_index()
    centers.index.name = "cluster"

    print(f"GMM 最优 K={best_k}（BIC 最低, 轮廓系数={best_sil:.3f}）")
    return stable_labels, centers, metrics_df, proba_df


def _plot_elbow_silhouette_named(sil_df: pd.DataFrame, output_dir: Path, filename: str, title_prefix: str) -> None:
    """通用版肘部图/轮廓系数图（带自定义文件名和标题）。"""
    plt.rcParams["font.family"] = ["Arial Unicode MS", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ks = sil_df["k"].values

    ax1.plot(ks, sil_df["inertia"].values, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("K (聚类数)", fontsize=12)
    ax1.set_ylabel("Inertia (SSE)", fontsize=12)
    ax1.set_title(f"{title_prefix} — 肘部图", fontsize=14)
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, sil_df["silhouette"].values, "rs-", linewidth=2, markersize=8)
    best_idx = sil_df["silhouette"].idxmax()
    best_k = int(sil_df.loc[best_idx, "k"])
    best_sil = float(sil_df.loc[best_idx, "silhouette"])
    ax2.axvline(x=best_k, color="gray", linestyle="--", alpha=0.6)
    ax2.annotate(f"最优 K={best_k}\n({best_sil:.3f})", xy=(best_k, best_sil),
                 xytext=(best_k + 0.8, best_sil), fontsize=11,
                 arrowprops=dict(arrowstyle="->", color="gray"))
    ax2.set_xlabel("K (聚类数)", fontsize=12)
    ax2.set_ylabel("轮廓系数 (Silhouette)", fontsize=12)
    ax2.set_title(f"{title_prefix} — 轮廓系数", fontsize=14)
    ax2.set_xticks(ks)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{title_prefix} 图已保存: {path}")


def _plot_gmm_metrics(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    """画 GMM 的 BIC/AIC + 轮廓系数三合一图。"""
    plt.rcParams["font.family"] = ["Arial Unicode MS", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ks = metrics_df["k"].values

    ax1.plot(ks, metrics_df["bic"].values, "go-", linewidth=2, markersize=8, label="BIC")
    ax1.plot(ks, metrics_df["aic"].values, "b^--", linewidth=1.5, markersize=7, label="AIC")
    best_bic_idx = metrics_df["bic"].idxmin()
    best_k_bic = int(metrics_df.loc[best_bic_idx, "k"])
    ax1.axvline(x=best_k_bic, color="gray", linestyle="--", alpha=0.6)
    ax1.set_xlabel("K (聚类数)", fontsize=12)
    ax1.set_ylabel("信息准则", fontsize=12)
    ax1.set_title(f"GMM — BIC/AIC (最优 K={best_k_bic})", fontsize=14)
    ax1.set_xticks(ks)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, metrics_df["silhouette"].values, "rs-", linewidth=2, markersize=8)
    best_sil_idx = metrics_df["silhouette"].idxmax()
    best_k_sil = int(metrics_df.loc[best_sil_idx, "k"])
    best_sil = float(metrics_df.loc[best_sil_idx, "silhouette"])
    ax2.axvline(x=best_k_sil, color="gray", linestyle="--", alpha=0.6)
    ax2.annotate(f"最优 K={best_k_sil}\n({best_sil:.3f})", xy=(best_k_sil, best_sil),
                 xytext=(best_k_sil + 0.8, best_sil), fontsize=11,
                 arrowprops=dict(arrowstyle="->", color="gray"))
    ax2.set_xlabel("K (聚类数)", fontsize=12)
    ax2.set_ylabel("轮廓系数 (Silhouette)", fontsize=12)
    ax2.set_title("GMM — 轮廓系数", fontsize=14)
    ax2.set_xticks(ks)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "gmm_bic_silhouette.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"GMM 图已保存: {path}")


# ══════════════════════════════════════════════
# 3. 增长/通胀四象限（对照）
# ══════════════════════════════════════════════

QUADRANT_NAMES = {
    (True, False): "Recovery",
    (True, True): "Overheat",
    (False, True): "Stagflation",
    (False, False): "Recession",
}
QUADRANT_CN = {
    "Recovery": "复苏",
    "Overheat": "过热",
    "Stagflation": "滞涨",
    "Recession": "衰退",
}


def classify_quadrant(features_df: pd.DataFrame, smooth_weeks: int = 0) -> pd.Series:
    """增长/通胀四象限划分。

    增长维度 = pmi_delta > 0（PMI 在改善）
    通胀维度 = ppi_delta > 0（PPI 在加速）
    """
    growth_up = features_df["pmi_delta"] > 0
    inflation_up = features_df["ppi_delta"] > 0

    regime = pd.Series("Unknown", index=features_df.index, name="regime_quadrant")
    for (g, i), name in QUADRANT_NAMES.items():
        mask = (growth_up == g) & (inflation_up == i)
        regime[mask] = name

    if smooth_weeks > 0:
        regime = _smooth_regime(regime, smooth_weeks)

    return regime


def _smooth_regime(regime: pd.Series, confirm_weeks: int) -> pd.Series:
    """平滑：连续 N 周确认才切换，避免频繁跳转。"""
    smoothed = regime.copy()
    current = regime.iloc[0]
    count = 0
    for i in range(1, len(regime)):
        if regime.iloc[i] != current:
            count += 1
            if count >= confirm_weeks:
                current = regime.iloc[i]
                count = 0
            else:
                smoothed.iloc[i] = current
        else:
            count = 0
    return smoothed


# ══════════════════════════════════════════════
# 4. Regime x 资产表现分析
# ══════════════════════════════════════════════

def analyze_regime_performance(
    regime_labels: pd.Series,
    weekly_ret: pd.DataFrame,
    assets: list[str],
) -> pd.DataFrame:
    """分 regime 统计各资产的年化收益/波动/胜率/回撤/Sharpe。"""
    total_weeks = len(regime_labels)
    rows: list[dict] = []

    for regime_name in sorted(regime_labels.unique()):
        mask = regime_labels == regime_name
        n_weeks = int(mask.sum())

        for asset in assets:
            r = weekly_ret.loc[mask, asset].dropna()
            if len(r) < 4:
                continue

            cum = (1 + r).prod()
            ann_ret = float(cum ** (52 / len(r)) - 1) if cum > 0 else float("nan")
            ann_vol = float(r.std(ddof=0) * np.sqrt(52))
            sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")

            nav = (1 + r).cumprod()
            max_dd = float((nav / nav.cummax() - 1).min())
            win_rate = float((r > 0).mean())

            rows.append({
                "regime": regime_name,
                "asset": asset,
                "weeks": n_weeks,
                "pct_of_total": round(n_weeks / total_weeks, 3),
                "ann_return": round(ann_ret, 4),
                "ann_vol": round(ann_vol, 4),
                "sharpe": round(sharpe, 3),
                "max_drawdown": round(max_dd, 4),
                "win_rate": round(win_rate, 3),
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# 5. Regime x 因子 IC 分析（研究诊断）
# ══════════════════════════════════════════════

def _build_factor_signals(
    factors: Dict[str, pd.Series],
    common_idx: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """把各因子变化率展开为 (date x asset) 的 DataFrame，用于截面 IC 计算。
    对于宏观因子（各资产共享同一值），构造的截面 IC 反映的是
    "该因子变化方向与各资产收益排序的一致性"。
    """

    def _chg(key: str, periods: int | None = None) -> pd.Series:
        monthly_like = {"pmi", "ppi", "cn_pmi", "cn_ppi", "real_rate"}
        weekly_like = {"gold_oi", "silver_oi", "copper_inventory", "oil_inventory", "coal_inventory"}
        if periods is None:
            periods = 1 if key in monthly_like or key in weekly_like else 4
        s = factors.get(key, pd.Series(dtype=float))
        if s.empty:
            return pd.Series(0.0, index=common_idx)
        s2 = s.reindex(common_idx).ffill()
        return s2.pct_change(periods).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _zscore(key: str, window: int = 52) -> pd.Series:
        s = factors.get(key, pd.Series(dtype=float))
        if s.empty:
            return pd.Series(0.0, index=common_idx)
        s2 = s.reindex(common_idx).ffill()
        mu = s2.rolling(window, min_periods=window // 2).mean()
        sd = s2.rolling(window, min_periods=window // 2).std(ddof=0)
        return ((s2 - mu) / sd.replace(0, np.nan)).fillna(0.0)

    # 与 rotation_model.build_signal_panel 中的因子口径一致
    signal_map: Dict[str, pd.Series] = {
        "real_rate_chg": -_chg("real_rate"),
        "dxy_chg": _chg("dxy"),
        "pmi_chg": _chg("pmi"),
        "ppi_chg": _chg("ppi"),
        "vix_chg": _chg("vix"),
        "fxi_chg": _chg("fxi"),
        "gold_oi_chg": _chg("gold_oi"),
        "silver_oi_chg": _chg("silver_oi"),
        "gold_cot_z": _zscore("gold_cot"),
        "copper_cot_z": _zscore("copper_cot"),
        "oil_cot_z": _zscore("oil_cot"),
    }

    # 宏观因子对所有资产是同一个值，展开为 (date x asset)
    out: Dict[str, pd.DataFrame] = {}
    for name, s in signal_map.items():
        df = pd.DataFrame({a: s for a in ASSETS}, index=common_idx)
        out[name] = df

    return out


def analyze_regime_factor_ic(
    regime_labels: pd.Series,
    factor_signals: Dict[str, pd.DataFrame],
    weekly_ret: pd.DataFrame,
    assets: list[str],
) -> pd.DataFrame:
    """分 regime 计算各因子对下一周收益的截面 IC。

    注意：5 个品种截面很小，单周 IC 噪声大，看的是 IC 均值和 IC 胜率。
    """
    fwd_ret = weekly_ret[assets].shift(-1)
    rows: list[dict] = []

    for regime_name in sorted(regime_labels.unique()):
        mask = regime_labels == regime_name
        dates_in = regime_labels.index[mask]

        for factor_name, fdf in factor_signals.items():
            ics: list[float] = []
            for dt in dates_in:
                if dt not in fdf.index or dt not in fwd_ret.index:
                    continue
                f_row = fdf.loc[dt, assets].dropna()
                r_row = fwd_ret.loc[dt, assets].dropna()
                common = f_row.index.intersection(r_row.index)
                if len(common) < 3:
                    continue
                ic = float(f_row[common].corr(r_row[common]))
                if not np.isnan(ic):
                    ics.append(ic)

            if len(ics) < 10:
                continue

            rows.append({
                "regime": regime_name,
                "factor": factor_name,
                "ic_mean": round(np.mean(ics), 4),
                "ic_std": round(np.std(ics), 4),
                "ic_ir": round(np.mean(ics) / np.std(ics), 3) if np.std(ics) > 0 else 0.0,
                "ic_win_rate": round(np.mean([1.0 if x > 0 else 0.0 for x in ics]), 3),
                "n_weeks": len(ics),
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# 6. Regime 切换期分析
# ══════════════════════════════════════════════

def analyze_regime_transitions(
    regime_labels: pd.Series,
    weekly_ret: pd.DataFrame,
    assets: list[str],
    window: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """识别 regime 切换点，统计切换前后各 window 周的资产累计收益。

    Returns:
        detail: 每次切换的明细
        summary: 按切换类型汇总的平均表现
    """
    transitions: list[dict] = []
    prev = regime_labels.iloc[0]
    for i in range(1, len(regime_labels)):
        curr = regime_labels.iloc[i]
        if curr != prev:
            transitions.append({
                "date": regime_labels.index[i],
                "from": prev,
                "to": curr,
                "type": f"{prev} -> {curr}",
            })
        prev = curr

    if not transitions:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict] = []
    for t in transitions:
        dt = t["date"]
        idx = regime_labels.index.get_loc(dt)

        pre_start = max(0, idx - window)
        pre_ret = weekly_ret[assets].iloc[pre_start:idx]
        pre_cum = (1 + pre_ret).prod() - 1

        post_end = min(len(weekly_ret), idx + window)
        post_ret = weekly_ret[assets].iloc[idx:post_end]
        post_cum = (1 + post_ret).prod() - 1

        for asset in assets:
            rows.append({
                "date": dt,
                "transition": t["type"],
                "asset": asset,
                "pre_cum_ret": round(float(pre_cum.get(asset, np.nan)), 4),
                "post_cum_ret": round(float(post_cum.get(asset, np.nan)), 4),
            })

    detail = pd.DataFrame(rows)

    summary = detail.groupby(["transition", "asset"]).agg(
        count=("date", "count"),
        avg_pre_ret=("pre_cum_ret", "mean"),
        avg_post_ret=("post_cum_ret", "mean"),
    ).reset_index()
    summary["avg_pre_ret"] = summary["avg_pre_ret"].round(4)
    summary["avg_post_ret"] = summary["avg_post_ret"].round(4)

    return detail, summary


# ══════════════════════════════════════════════
# 7. 文字摘要生成
# ══════════════════════════════════════════════

def _build_summary_text(
    regime_labels: pd.Series,
    perf_df: pd.DataFrame,
    centers: pd.DataFrame,
    silhouette_df: pd.DataFrame,
    quadrant_labels: pd.Series,
) -> str:
    """生成可读的文字摘要。"""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("大宗商品轮动模型 — 宏观周期 Regime 分析报告")
    lines.append("=" * 60)

    # KMeans 概览
    best_k = int(silhouette_df.loc[silhouette_df["silhouette"].idxmax(), "k"])
    best_sil = float(silhouette_df["silhouette"].max())
    lines.append(f"\n【KMeans 聚类】最优 K={best_k}，轮廓系数={best_sil:.3f}")
    lines.append("各 K 轮廓系数：")
    for _, row in silhouette_df.iterrows():
        lines.append(f"  K={int(row['k'])}: {row['silhouette']:.3f}")

    lines.append("\n聚类中心（标准化后的值，0=历史均值）：")
    lines.append(centers.to_string())

    # 各 regime 持续时间
    lines.append("\n【Regime 分布】")
    vc = regime_labels.value_counts().sort_index()
    total = len(regime_labels)
    for r, cnt in vc.items():
        lines.append(f"  Cluster {r}: {cnt} 周 ({cnt/total:.1%})")

    # 各 regime 下表现最好/最差的资产
    lines.append("\n【各 Regime 下资产表现排名（按年化收益）】")
    for regime_name in sorted(perf_df["regime"].unique()):
        sub = perf_df[perf_df["regime"] == regime_name].sort_values("ann_return", ascending=False)
        lines.append(f"\n  --- Regime {regime_name} ({sub.iloc[0]['weeks']} 周, 占 {sub.iloc[0]['pct_of_total']:.1%}) ---")
        for _, row in sub.iterrows():
            lines.append(
                f"    {row['asset']:4s}  年化={row['ann_return']:+.1%}  "
                f"波动={row['ann_vol']:.1%}  Sharpe={row['sharpe']:.2f}  "
                f"回撤={row['max_drawdown']:.1%}  胜率={row['win_rate']:.1%}"
            )

    # 四象限对照
    lines.append("\n\n【增长/通胀四象限分布（对照）】")
    qvc = quadrant_labels.value_counts()
    for r, cnt in qvc.items():
        cn = QUADRANT_CN.get(r, r)
        lines.append(f"  {r}({cn}): {cnt} 周 ({cnt/total:.1%})")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ══════════════════════════════════════════════
# 8. 主流程
# ══════════════════════════════════════════════

def main() -> None:
    if not WORKBOOK_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件: {WORKBOOK_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("读取数据...")
    weekly_prices = load_weekly_prices(WORKBOOK_PATH)
    factors = load_weekly_factors(WORKBOOK_PATH)
    weekly_ret = weekly_prices[ASSETS].pct_change()
    common_idx = weekly_prices.index

    # Step 1: 构造宏观特征
    print("构造宏观状态特征...")
    features = build_regime_features(factors, common_idx, weekly_prices=weekly_prices)
    features.to_csv(OUTPUT_DIR / "regime_features.csv", encoding="utf-8-sig")

    # Step 2: KMeans 聚类
    print("KMeans 聚类...")
    km_labels, centers, sil_df = classify_kmeans(features, output_dir=OUTPUT_DIR)
    km_labels.to_csv(OUTPUT_DIR / "regime_labels_kmeans.csv", encoding="utf-8-sig", header=True)
    centers.to_csv(OUTPUT_DIR / "kmeans_centers.csv", encoding="utf-8-sig")
    sil_df.to_csv(OUTPUT_DIR / "kmeans_silhouette.csv", index=False, encoding="utf-8-sig")

    # Step 3: PCA + KMeans（强制 K=6）
    print("\nPCA + KMeans 聚类...")
    pca_labels, pca_centers, pca_sil_df, pca_explained = classify_pca_kmeans(
        features, force_k=6, output_dir=OUTPUT_DIR
    )
    pca_labels.to_csv(OUTPUT_DIR / "regime_labels_pca_kmeans.csv", encoding="utf-8-sig", header=True)
    pca_centers.to_csv(OUTPUT_DIR / "pca_kmeans_centers.csv", encoding="utf-8-sig")
    pca_sil_df.to_csv(OUTPUT_DIR / "pca_kmeans_silhouette.csv", index=False, encoding="utf-8-sig")
    pca_explained.to_csv(OUTPUT_DIR / "pca_explained_variance.csv", index=False, encoding="utf-8-sig")

    # Step 4: GMM
    print("\nGMM 聚类...")
    gmm_labels, gmm_centers, gmm_metrics, gmm_proba = classify_gmm(
        features, output_dir=OUTPUT_DIR
    )
    gmm_labels.to_csv(OUTPUT_DIR / "regime_labels_gmm.csv", encoding="utf-8-sig", header=True)
    gmm_centers.to_csv(OUTPUT_DIR / "gmm_centers.csv", encoding="utf-8-sig")
    gmm_metrics.to_csv(OUTPUT_DIR / "gmm_metrics.csv", index=False, encoding="utf-8-sig")
    gmm_proba.to_csv(OUTPUT_DIR / "gmm_probabilities.csv", encoding="utf-8-sig")

    # Step 5: 四象限（对照）
    print("\n增长/通胀四象限划分...")
    quad_labels = classify_quadrant(features, smooth_weeks=2)
    quad_labels.to_csv(OUTPUT_DIR / "regime_labels_quadrant.csv", encoding="utf-8-sig", header=True)

    # Step 6: 所有方法的 regime x 资产表现
    print("\n分析 regime x 资产表现...")
    all_methods = {
        "kmeans": km_labels,
        "pca_kmeans": pca_labels,
        "gmm": gmm_labels,
        "quadrant": quad_labels,
    }
    perf_all: dict[str, pd.DataFrame] = {}
    for method_name, labels in all_methods.items():
        common = labels.index.intersection(weekly_ret.index)
        perf = analyze_regime_performance(labels.reindex(common), weekly_ret.reindex(common), ASSETS)
        perf["method"] = method_name
        perf.to_csv(OUTPUT_DIR / f"performance_by_regime_{method_name}.csv", index=False, encoding="utf-8-sig")
        perf_all[method_name] = perf

    # Step 7: 因子 IC（KMeans 版，诊断用）
    print("分析 regime x 因子 IC...")
    factor_signals = _build_factor_signals(factors, common_idx)
    km_common = km_labels.index.intersection(weekly_ret.index)
    ic_km = analyze_regime_factor_ic(km_labels.reindex(km_common), factor_signals, weekly_ret.reindex(km_common), ASSETS)
    ic_km.to_csv(OUTPUT_DIR / "factor_ic_by_regime_kmeans.csv", index=False, encoding="utf-8-sig")

    # Step 8: 切换期分析（KMeans）
    print("分析 regime 切换期...")
    trans_detail, trans_summary = analyze_regime_transitions(km_labels.reindex(km_common), weekly_ret.reindex(km_common), ASSETS)
    if not trans_detail.empty:
        trans_detail.to_csv(OUTPUT_DIR / "regime_transitions_detail.csv", index=False, encoding="utf-8-sig")
        trans_summary.to_csv(OUTPUT_DIR / "regime_transitions_summary.csv", index=False, encoding="utf-8-sig")

    # Step 9: 三种方法对比汇总
    print("\n" + "=" * 60)
    _print_method_comparison(sil_df, pca_sil_df, gmm_metrics, perf_all, OUTPUT_DIR)

    # Step 10: 文字摘要
    summary_text = _build_summary_text(km_labels, perf_all["kmeans"], centers, sil_df, quad_labels)
    (OUTPUT_DIR / "regime_summary.txt").write_text(summary_text, encoding="utf-8")

    print(f"\n所有结果已输出到 {OUTPUT_DIR.resolve()}/")


def _print_method_comparison(
    km_sil: pd.DataFrame,
    pca_sil: pd.DataFrame,
    gmm_metrics: pd.DataFrame,
    perf_all: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """输出三种聚类方法的对比表。"""
    lines: list[str] = []
    lines.append("三种聚类方法对比")
    lines.append("=" * 60)

    # 最优 K 和轮廓系数
    km_best = km_sil.loc[km_sil["silhouette"].idxmax()]
    pca_best = pca_sil.loc[pca_sil["silhouette"].idxmax()]
    gmm_bic_best = gmm_metrics.loc[gmm_metrics["bic"].idxmin()]
    gmm_sil_best = gmm_metrics.loc[gmm_metrics["silhouette"].idxmax()]

    lines.append(f"\n{'方法':<16s} {'最优K':>5s} {'轮廓系数':>10s} {'选K依据':>10s}")
    lines.append("-" * 45)
    lines.append(f"{'KMeans':<16s} {int(km_best['k']):>5d} {km_best['silhouette']:>10.3f} {'Silhouette':>10s}")
    lines.append(f"{'PCA+KMeans':<16s} {int(pca_best['k']):>5d} {pca_best['silhouette']:>10.3f} {'Silhouette':>10s}")
    lines.append(f"{'GMM(BIC)':<16s} {int(gmm_bic_best['k']):>5d} {gmm_bic_best['silhouette']:>10.3f} {'BIC':>10s}")
    lines.append(f"{'GMM(Sil)':<16s} {int(gmm_sil_best['k']):>5d} {gmm_sil_best['silhouette']:>10.3f} {'Silhouette':>10s}")

    # 各方法下 regime 间 Sharpe 差距（越大=区分度越好）
    lines.append(f"\n{'方法':<16s} {'K':>3s} {'最高Sharpe':>10s} {'最低Sharpe':>10s} {'差距':>8s}")
    lines.append("-" * 50)
    for method_name, perf in perf_all.items():
        if perf.empty:
            continue
        regime_sharpe = perf.groupby("regime")["sharpe"].mean()
        best_s = regime_sharpe.max()
        worst_s = regime_sharpe.min()
        n_regimes = regime_sharpe.nunique()
        lines.append(f"{method_name:<16s} {n_regimes:>3d} {best_s:>+10.2f} {worst_s:>+10.2f} {best_s - worst_s:>8.2f}")

    text = "\n".join(lines)
    print(text)
    (output_dir / "method_comparison.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
