# -*- coding: utf-8 -*-
"""
对比实验：剔除 commodity_mom 后用 K=6 重新聚类，生成独立 HTML 报告。
输出文件：regime_report_no_mom.html
"""
from __future__ import annotations

import base64
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import WORKBOOK_PATH
from regime_analysis import (
    build_regime_features,
    analyze_regime_performance,
    PCA_FEATURES,
)
from rotation_model import load_weekly_factors, load_weekly_prices, ASSETS

_BASE    = Path(__file__).parent
HTML_OUT = _BASE / "regime_report_no_mom.html"

# ── 去掉 commodity_mom 的特征列表 ────────────────────────────────────────────
FEATURES_NO_MOM = [f for f in PCA_FEATURES if f != "commodity_mom"]

K = 6
N_COMPONENTS = 3

# ── Cluster 颜色（同主报告）──────────────────────────────────────────────────
COLORS = ["#1e8449", "#27ae60", "#d68910", "#2980b9", "#c0392b", "#7d3c98"]

FACTOR_CN = {
    "pmi_level": "美国PMI·水平", "pmi_delta": "美国PMI·方向",
    "ppi_level": "美国PPI·水平", "ppi_delta": "美国PPI·方向",
    "vix_level": "VIX恐慌·水平", "dxy_level": "美元指数·水平",
    "real_rate_level": "实际利率·水平",
    "cn_pmi_level": "中国PMI·水平", "cn_pmi_delta": "中国PMI·方向",
    "cn_ppi_level": "中国PPI·水平", "cn_ppi_delta": "中国PPI·方向",
    "us10y_level": "美债10Y·水平", "us10y_delta": "美债10Y·方向",
    "yield_spread_level": "利差·水平", "yield_spread_delta": "利差·方向",
    "credit_impulse_level": "信贷脉冲·水平", "credit_impulse_delta": "信贷脉冲·方向",
}

KEY_FEATURES = [
    "pmi_level", "pmi_delta", "ppi_level", "real_rate_level",
    "dxy_level", "vix_level", "us10y_level",
    "cn_pmi_level", "cn_pmi_delta",
]


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _bar(val: float, lo: float = -2.0, hi: float = 2.0) -> str:
    pct = max(0.0, min(100.0, (val - lo) / (hi - lo) * 100))
    mid = (0 - lo) / (hi - lo) * 100
    if val >= 0:
        left, width, color = mid, pct - mid, "#2980b9"
    else:
        left, width, color = pct, mid - pct, "#c0392b"
    return (
        f'<div style="position:relative;height:10px;background:#eee;border-radius:3px">'
        f'<div style="position:absolute;left:{left:.1f}%;width:{max(width,0):.1f}%;'
        f'height:100%;background:{color};border-radius:3px"></div>'
        f'<div style="position:absolute;left:{mid:.1f}%;width:1px;height:100%;background:#aaa"></div>'
        f'</div><span style="font-size:.75em;color:#555">{val:+.2f}</span>'
    )


def _sharpe_td(v: float) -> str:
    if v >= 0.8:   c = "#c0392b"
    elif v >= 0.3: c = "#d68910"
    elif v >= 0:   c = "#888"
    else:          c = "#1e8449"
    return f'<td style="color:{c};font-weight:bold">{v:+.2f}</td>'


def _ret_td(v: float) -> str:
    c = "#c0392b" if v > 0 else "#1e8449"
    return f'<td style="color:{c};font-weight:bold">{v:+.1%}</td>'


# ── 核心分析 ─────────────────────────────────────────────────────────────────

def run_analysis():
    prices  = load_weekly_prices(WORKBOOK_PATH)
    factors = load_weekly_factors(WORKBOOK_PATH)

    common_idx = prices.index
    feat_df = build_regime_features(factors, common_idx, weekly_prices=prices)

    valid = feat_df[FEATURES_NO_MOM].dropna()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(valid)
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    # K=6 聚类
    km = KMeans(n_clusters=K, random_state=42, n_init=20)
    raw_labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, raw_labels)
    print(f"轮廓系数 K={K}（无commodity_mom）: {sil:.3f}")

    # 按 PMI 降序排编号
    pmi_col = FEATURES_NO_MOM.index("pmi_level")
    cluster_pmi = pd.Series(
        [X_sc[raw_labels == c, pmi_col].mean() for c in range(K)]
    )
    rank_map = cluster_pmi.rank(ascending=False).astype(int) - 1
    label_remap = {old: int(new) for old, new in rank_map.items()}
    labels = pd.Series(
        [label_remap[lb] for lb in raw_labels],
        index=valid.index, name="cluster"
    ).reindex(common_idx).ffill().bfill()

    # Cluster 中心（原始特征空间）
    centers_pca = km.cluster_centers_
    centers_orig = scaler.inverse_transform(pca.inverse_transform(centers_pca))
    centers_df = pd.DataFrame(centers_orig, columns=FEATURES_NO_MOM)
    centers_df.index = [label_remap[i] for i in range(K)]
    centers_df = centers_df.sort_index()

    # 资产表现
    weekly_ret = prices.pct_change()
    perf = analyze_regime_performance(labels, weekly_ret, list(prices.columns))
    perf.insert(perf.columns.get_loc("ann_return"), "method", "no_mom")

    # PCA 解释方差
    ev = pca.explained_variance_ratio_
    pca_info = {f"PC{i+1}": f"{ev[i]:.1%}" for i in range(N_COMPONENTS)}
    pca_info["cumulative"] = f"{ev.sum():.1%}"

    # PCA 载荷
    loadings = pd.DataFrame(
        pca.components_.T,
        index=FEATURES_NO_MOM,
        columns=[f"PC{i+1}" for i in range(N_COMPONENTS)],
    )

    return labels, centers_df, perf, pca_info, loadings, sil, feat_df


# ── HTML 构建 ────────────────────────────────────────────────────────────────

def build_html(labels, centers_df, perf, pca_info, loadings, sil, feat_df):
    total = len(labels)
    counts = labels.value_counts().sort_index()

    # ── 当前状态 ──
    latest = labels.index[-1].strftime("%Y-%m-%d")
    cur_cid = int(labels.iloc[-1])
    consec = sum(1 for v in reversed(labels.tolist()) if int(v) == cur_cid)
    cur_color = COLORS[cur_cid]
    sec_current = f"""
<div class="section">
  <h2>⚡ 当前状态（截至 {latest}）</h2>
  <div style="font-size:1.25em;font-weight:bold;color:{cur_color}">
    C{cur_cid}
    <span class="current-badge">当前 · 连续{consec}周</span>
  </div>
  <p style="color:#888;font-size:.9em;margin-top:8px">
    本报告去掉了 commodity_mom（12周动量因子），共 {len(FEATURES_NO_MOM)} 个特征，
    K=6 轮廓系数 = {sil:.3f}
  </p>
</div>"""

    # ── PCA 载荷 ──
    pc_headers = "".join(
        f'<th>PC{i+1} {pca_info[f"PC{i+1}"]}</th>'
        for i in range(N_COMPONENTS)
    )
    load_rows = ""
    for factor, row in loadings.iterrows():
        cn = FACTOR_CN.get(factor, factor)
        def _cell(v):
            w = "bold" if abs(v) >= 0.28 else "normal"
            c = "#2980b9" if v > 0.1 else ("#c0392b" if v < -0.1 else "#888")
            return f'<td style="font-weight:{w};color:{c}">{v:+.3f}</td>'
        cells = "".join(_cell(row[f"PC{i+1}"]) for i in range(N_COMPONENTS))
        load_rows += f'<tr><td style="text-align:left">{cn}</td>{cells}</tr>'

    sec_loadings = f"""
<div class="section">
  <h2>一、PCA 因子载荷（{len(FEATURES_NO_MOM)}因子 → {N_COMPONENTS}主成分，累计 {pca_info['cumulative']}）</h2>
  <p style="font-size:.88em;color:#c0392b;margin-bottom:8px">⚠️ 已剔除：commodity_mom（12周动量）</p>
  <table class="data-table">
    <thead><tr><th>因子</th>{pc_headers}</tr></thead>
    <tbody>{load_rows}</tbody>
  </table>
</div>"""

    # ── Cluster 卡片 + 详情 ──
    cards = ""
    details = ""
    for cid in range(K):
        cnt = int(counts.get(cid, 0))
        pct = cnt / total
        c = COLORS[cid]
        sub = perf[perf["regime"] == cid].sort_values("sharpe", ascending=False)
        best_txt = "".join(
            f'<span class="best">{r["asset"]} Sharpe {r["sharpe"]:+.2f}</span> '
            for _, r in sub.head(2).iterrows()
        )
        cards += f"""
    <div class="card" style="background:{c}">
      <h4>C{cid}</h4>
      <div class="meta">{cnt}周（{pct:.1%}）</div>
      <div style="margin-top:8px">{best_txt}</div>
    </div>"""

        # 宏观特征
        feat_rows = ""
        if cid in centers_df.index:
            row = centers_df.loc[cid]
            for col in KEY_FEATURES:
                if col in row.index:
                    feat_rows += (
                        f'<tr><td style="width:140px;font-size:.85em">{FACTOR_CN.get(col, col)}</td>'
                        f'<td style="padding:4px 8px">{_bar(row[col])}</td></tr>'
                    )
        asset_rows = "".join(
            f'<tr><td>{r["asset"]}</td>'
            + _ret_td(r["ann_return"])
            + f'<td>{r["ann_vol"]:.1%}</td>'
            + _sharpe_td(r["sharpe"])
            + f'<td>{r["max_drawdown"]:.1%}</td>'
            + f'<td>{r["win_rate"]:.1%}</td></tr>'
            for _, r in sub.iterrows()
        )
        details += f"""
  <div class="cluster-detail">
    <div class="cluster-header">
      <div class="cluster-dot" style="background:{c}"></div>
      <span style="font-size:1.1em;font-weight:bold;color:{c}">C{cid}</span>
      <span style="color:#888;font-size:.88em">{cnt}周 ({pct:.1%})</span>
    </div>
    <div style="display:flex;gap:28px;flex-wrap:wrap">
      <div style="flex:1;min-width:240px">
        <h3 style="margin-top:0">宏观特征</h3>
        <table style="width:100%">{feat_rows}</table>
      </div>
      <div style="flex:1;min-width:280px">
        <h3 style="margin-top:0">资产表现</h3>
        <table class="data-table">
          <thead><tr><th>资产</th><th>年化收益</th><th>年化波动</th><th>Sharpe</th><th>最大回撤</th><th>胜率</th></tr></thead>
          <tbody>{asset_rows}</tbody>
        </table>
      </div>
    </div>
  </div>"""

    sec_cards   = f'<div class="section"><h2>二、Cluster 配置（K=6，无动量）</h2><div class="cluster-cards">{cards}</div></div>'
    sec_details = f'<div class="section"><h2>三、各 Cluster 详情</h2>{details}</div>'

    # ── 每周记录 ──
    rows_by_year: dict[int, list[str]] = {}
    for dt, val in labels.items():
        cid = int(val)
        c = COLORS[cid]
        year = dt.year
        rows_by_year.setdefault(year, [])
        rows_by_year[year].append(
            f'<tr><td>{dt.strftime("%Y-%m-%d")}</td>'
            f'<td><span style="background:{c};color:white;padding:2px 10px;'
            f'border-radius:12px;font-size:.82em;font-weight:bold">C{cid}</span></td></tr>'
        )
    year_blocks = ""
    for year in sorted(rows_by_year.keys(), reverse=True):
        yr_rows = "".join(rows_by_year[year])
        year_blocks += f"""
  <details {"open" if year >= labels.index[-1].year - 1 else ""}>
    <summary style="cursor:pointer;font-weight:bold;padding:8px 4px;border-bottom:1px solid #eee">{year} 年（{len(rows_by_year[year])} 周）</summary>
    <table class="data-table" style="margin-top:8px">
      <thead><tr><th>日期</th><th>Cluster</th></tr></thead>
      <tbody>{yr_rows}</tbody>
    </table>
  </details>"""
    sec_weekly = f'<div class="section"><h2>四、每周 Cluster 记录</h2>{year_blocks}</div>'

    body = "\n".join([sec_current, sec_loadings, sec_cards, sec_details, sec_weekly])

    CSS = """
body{font-family:"PingFang SC","Microsoft YaHei",Arial,sans-serif;margin:0;background:#f5f6fa;color:#2c3e50}
.cover{background:linear-gradient(135deg,#1a252f,#2c3e50);color:white;padding:60px 80px}
.cover h1{font-size:2.2em;margin:0 0 10px}
.cover p{font-size:1.1em;opacity:.8;margin:5px 0}
.container{max-width:1100px;margin:30px auto;padding:0 30px}
.section{background:white;border-radius:12px;padding:30px;margin-bottom:24px;box-shadow:0 2px 12px rgba(0,0,0,.07)}
h2{color:#1a252f;border-left:4px solid #3498db;padding-left:12px;font-size:1.4em;margin-top:0}
h3{color:#2c3e50;font-size:1.1em;margin-top:20px}
.cluster-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:16px}
.card{border-radius:10px;padding:18px;color:white}
.card h4{margin:0 0 6px;font-size:1em}
.card .meta{font-size:.82em;opacity:.9;margin:4px 0}
.card .best{background:rgba(255,255,255,.2);border-radius:6px;padding:4px 8px;margin:2px;display:inline-block;font-size:.85em}
.cluster-detail{border:1px solid #e0e0e0;border-radius:8px;padding:20px;margin-bottom:20px}
.cluster-header{display:flex;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap}
.cluster-dot{width:16px;height:16px;border-radius:50%;flex-shrink:0}
.data-table{border-collapse:collapse;width:100%;font-size:.92em;margin-bottom:16px}
.data-table th{background:#2c3e50;color:white;padding:10px 14px;text-align:center}
.data-table td{padding:9px 14px;text-align:center;border-bottom:1px solid #eee}
.data-table tr:hover td{background:#f8f9fa}
.current-badge{display:inline-block;background:#1e8449;color:white;font-size:.62em;padding:3px 10px;border-radius:20px;vertical-align:middle;margin-left:10px;font-weight:normal}
"""

    today = date.today().strftime("%Y-%m-%d")
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>大宗商品轮动 — 无动量因子对比报告</title>
<style>{CSS}</style>
</head>
<body>
<div class="cover">
  <h1>大宗商品轮动模型</h1>
  <p>对比实验：剔除 commodity_mom 后 K=6 聚类结果</p>
  <p>生成日期：{today} &nbsp;·&nbsp; 特征数：{len(FEATURES_NO_MOM)}（原18，去掉12周动量）</p>
</div>
<div class="container">{body}</div>
</body>
</html>"""


def main():
    print("运行对比分析（无 commodity_mom，K=6）...")
    labels, centers_df, perf, pca_info, loadings, sil, feat_df = run_analysis()
    html = build_html(labels, centers_df, perf, pca_info, loadings, sil, feat_df)
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"报告已生成：{HTML_OUT.resolve()}")


if __name__ == "__main__":
    main()
