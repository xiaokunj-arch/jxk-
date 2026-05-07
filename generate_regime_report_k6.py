# -*- coding: utf-8 -*-
"""
独立运行：基于 K=6 PCA+KMeans 生成 v4 风格的 Regime 分析报告。
不影响 regime_outputs/ 里的 K=5 数据，输出到 regime_report_k6.html。

用法:
    python3 generate_regime_report_k6.py
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

from rotation_model import load_weekly_prices, load_weekly_factors, ASSETS
from regime_analysis import build_regime_features, PCA_FEATURES
from config import WORKBOOK_PATH

OUT_DIR  = Path("regime_outputs")
HTML_OUT = Path("regime_report_k6.html")
K        = 6

CLUSTER_CONFIG = {
    0: dict(name="工业金属/能源牛市", position=0.85, mom_weight=0.55, color="#1e8449"),
    1: dict(name="大宗全面牛市",     position=1.00, mom_weight=0.65, color="#27ae60"),
    2: dict(name="停滞横盘期",       position=0.55, mom_weight=0.35, color="#d68910"),
    3: dict(name="温和复苏期",       position=0.65, mom_weight=0.30, color="#2980b9"),
    4: dict(name="加息紧缩期",       position=0.00, mom_weight=0.50, color="#c0392b"),
    5: dict(name="衰退/通缩/避险",   position=0.20, mom_weight=0.20, color="#7d3c98"),
}

FACTOR_CN = {
    "pmi_level": "美国PMI·水平", "pmi_delta": "美国PMI·方向",
    "ppi_level": "美国PPI·水平", "ppi_delta": "美国PPI·方向",
    "vix_level": "VIX恐慌·水平", "dxy_level": "美元指数·水平",
    "real_rate_level": "实际利率·水平",
    "cn_pmi_level": "中国PMI·水平", "cn_pmi_delta": "中国PMI·方向",
    "cn_ppi_level": "中国PPI·水平", "cn_ppi_delta": "中国PPI·方向",
    "commodity_mom": "商品12W动量",
    "us10y_level": "美债10Y·水平", "us10y_delta": "美债10Y·方向",
    "yield_spread_level": "利差·水平", "yield_spread_delta": "利差·方向",
    "credit_impulse_level": "信贷脉冲·水平", "credit_impulse_delta": "信贷脉冲·方向",
}

KEY_FEATURES = [
    "pmi_level", "pmi_delta", "ppi_level", "real_rate_level",
    "dxy_level", "vix_level", "us10y_level",
    "cn_pmi_level", "cn_pmi_delta", "commodity_mom",
]


# ── 分析 ─────────────────────────────────────────────────────────────────────

def run_k6_analysis():
    """加载数据，跑 PCA+KMeans K=6，返回 labels / centers / perf / loadings。"""
    prices  = load_weekly_prices(WORKBOOK_PATH)
    factors = load_weekly_factors(WORKBOOK_PATH)
    feat_df = build_regime_features(factors, prices.index, weekly_prices=prices)

    valid   = feat_df[PCA_FEATURES].dropna()
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(valid)
    pca     = PCA(n_components=3, random_state=42)
    X_pca   = pca.fit_transform(X_sc)

    # PCA 载荷
    loadings = pd.DataFrame(
        pca.components_.T, index=PCA_FEATURES,
        columns=["PC1", "PC2", "PC3"]
    )
    ev = pca.explained_variance_ratio_

    # KMeans K=6
    km       = KMeans(n_clusters=K, random_state=42, n_init=20)
    raw_labs = km.fit_predict(X_pca)

    # 按 PMI 水平排序
    pmi_col  = PCA_FEATURES.index("pmi_level")
    pmi_mean = pd.Series([X_sc[raw_labs == c, pmi_col].mean() for c in range(K)])
    rank_map = {old: int(new) for old, new in (pmi_mean.rank(ascending=False).astype(int) - 1).items()}

    labels = pd.Series([rank_map[l] for l in raw_labs], index=valid.index, name="cluster")

    # 聚类中心（原始空间）
    centers_pca = km.cluster_centers_
    centers_orig = scaler.inverse_transform(pca.inverse_transform(centers_pca))
    centers_df   = pd.DataFrame(centers_orig, columns=PCA_FEATURES)
    centers_df.index = [rank_map[i] for i in range(K)]
    centers_df = centers_df.sort_index()

    # 资产收益
    weekly_ret = prices[ASSETS].pct_change(fill_method=None)
    common     = labels.index.intersection(weekly_ret.index)
    labels_c   = labels.reindex(common)
    ret_c      = weekly_ret.reindex(common)

    # 各 Cluster 资产绩效
    perf_rows = []
    total     = len(labels_c)
    for cid in sorted(labels_c.unique()):
        mask = labels_c == cid
        cnt  = int(mask.sum())
        for asset in ASSETS:
            r = ret_c.loc[mask, asset].dropna()
            if len(r) < 4:
                continue
            ann_r  = float((1 + r).prod() ** (52 / len(r)) - 1)
            ann_v  = float(r.std(ddof=0) * np.sqrt(52))
            sharpe = ann_r / ann_v if ann_v > 0 else 0.0
            cum    = (1 + r).cumprod()
            mdd    = float((cum / cum.cummax() - 1).min())
            wr     = float((r > 0).mean())
            perf_rows.append(dict(
                regime=cid, asset=asset, weeks=cnt,
                pct_of_total=cnt / total,
                ann_return=ann_r, ann_vol=ann_v,
                sharpe=sharpe, max_drawdown=mdd, win_rate=wr,
            ))
    perf = pd.DataFrame(perf_rows)

    # 轮廓系数（已有 CSV，直接读）
    pca_sil = pd.read_csv(OUT_DIR / "pca_kmeans_silhouette.csv")
    km_sil  = pd.read_csv(OUT_DIR / "kmeans_silhouette.csv")
    gmm_met = pd.read_csv(OUT_DIR / "gmm_metrics.csv")
    pca_exp = pd.read_csv(OUT_DIR / "pca_explained_variance.csv")

    return labels, centers_df, perf, loadings, ev, pca_sil, km_sil, gmm_met, pca_exp


# ── 工具 ─────────────────────────────────────────────────────────────────────

def _img_b64(path: Path) -> str:
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


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


def _sharpe_td(v):
    c = "#c0392b" if v >= 0.8 else ("#d68910" if v >= 0.3 else ("#888" if v >= 0 else "#1e8449"))
    return f'<td style="color:{c};font-weight:bold">{v:+.2f}</td>'


def _ret_td(v):
    return f'<td style="color:{"#c0392b" if v>0 else "#1e8449"};font-weight:bold">{v:+.1%}</td>'


# ── Sections ─────────────────────────────────────────────────────────────────

def sec_current(labels):
    latest = labels.index[-1].strftime("%Y-%m-%d")
    cid    = int(labels.iloc[-1])
    cfg    = CLUSTER_CONFIG[cid]
    consec = sum(1 for v in reversed(labels.tolist()) if int(v) == cid)
    c      = cfg["color"]
    return f"""
<div class="section">
  <h2>⚡ 当前状态（截至 {latest}）</h2>
  <div style="font-size:1.25em;font-weight:bold;color:{c}">
    C{cid}&nbsp;&nbsp;{cfg['name']}
    <span class="current-badge">当前 · 连续{consec}周</span>
  </div>
  <div style="display:flex;gap:32px;margin-top:16px;flex-wrap:wrap">
    <div class="stat-box"><div class="stat-label">基准仓位</div>
      <div class="stat-val" style="color:{c}">{cfg['position']:.0%}</div></div>
    <div class="stat-box"><div class="stat-label">动量权重</div>
      <div class="stat-val" style="color:#2980b9">{cfg['mom_weight']:.0%}</div></div>
    <div class="stat-box"><div class="stat-label">基本面权重</div>
      <div class="stat-val" style="color:#16a085">{1-cfg['mom_weight']:.0%}</div></div>
  </div>
</div>"""


def sec_method(pca_sil, km_sil, gmm_met, perf, labels):
    pca5  = float(pca_sil[pca_sil["k"] == 5]["silhouette"].values[0])
    pca6  = float(pca_sil[pca_sil["k"] == 6]["silhouette"].values[0])
    pca2  = float(pca_sil[pca_sil["k"] == 2]["silhouette"].values[0])
    km2   = float(km_sil[km_sil["k"] == 2]["silhouette"].values[0])
    gmm_b = gmm_met.loc[gmm_met["bic"].idxmin()]

    def sr(p):
        rs = p.groupby("regime")["sharpe"].mean()
        return rs.max(), rs.min(), rs.nunique()

    ph, pl, pn = sr(perf)
    quad_perf = pd.read_csv(OUT_DIR / "performance_by_regime_quadrant.csv")
    qh, ql, qn = sr(quad_perf)

    rows = (
        f'<tr><td>KMeans（6因子）</td><td>{int(km_sil.loc[km_sil["silhouette"].idxmax(),"k"])}</td>'
        f'<td>{km2:.3f}</td><td>Silhouette</td><td>2</td><td>—</td><td>—</td><td>—</td></tr>'
        f'<tr style="background:#eaf4fb"><td><b>PCA+KMeans ✓</b></td><td><b>6</b></td>'
        f'<td><b>{pca6:.3f}</b></td><td>策略选用</td><td>{pn}</td>'
        f'<td>{ph:+.2f}</td><td>{pl:+.2f}</td><td><b>{ph-pl:.2f}</b></td></tr>'
        f'<tr><td>GMM（BIC）</td><td>{int(gmm_b["k"])}</td>'
        f'<td>{float(gmm_b["silhouette"]):.3f}</td><td>BIC</td>'
        f'<td>{int(gmm_b["k"])}</td><td>—</td><td>—</td><td>—</td></tr>'
        f'<tr><td>增长/通胀四象限</td><td>4</td><td>—</td><td>规则</td>'
        f'<td>{qn}</td><td>{qh:+.2f}</td><td>{ql:+.2f}</td><td>{qh-ql:.2f}</td></tr>'
    )
    return f"""
<div class="section">
  <h2>一、方法对比汇总</h2>
  <table class="data-table">
    <thead><tr><th>方法</th><th>选K</th><th>轮廓系数</th><th>选K依据</th>
      <th>Cluster数</th><th>最高Sharpe</th><th>最低Sharpe</th><th>区分差距</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <div class="insight">
    ✅ <b>PCA+KMeans（K=6）</b>为策略选用方案：统计最优K=2，但K=6能额外识别
    「加息紧缩（C4，仓位=0%）」和「衰退避险（C5，黄金保底60%）」两个关键状态，
    策略价值显著。轮廓系数 K=6（{pca6:.3f}）vs 最优K=2（{pca2:.3f}）差距极小。
  </div>
</div>"""


def sec_pca_loadings(loadings, ev):
    pc_headers = (
        f'<th>PC1 {ev[0]:.1%}<br><span style="font-weight:normal;font-size:.85em">（增长/通胀景气）</span></th>'
        f'<th>PC2 {ev[1]:.1%}<br><span style="font-weight:normal;font-size:.85em">（货币/流动性方向）</span></th>'
        f'<th>PC3 {ev[2]:.1%}<br><span style="font-weight:normal;font-size:.85em">（实际利率水平）</span></th>'
    )
    rows = ""
    for factor, row in loadings.iterrows():
        cn = FACTOR_CN.get(factor, factor)
        def _cell(v):
            w = "bold" if abs(v) >= 0.28 else "normal"
            c = "#2980b9" if v > 0.1 else ("#c0392b" if v < -0.1 else "#888")
            return f'<td style="font-weight:{w};color:{c}">{v:+.3f}</td>'
        rows += f'<tr><td style="text-align:left">{cn}</td>{_cell(row["PC1"])}{_cell(row["PC2"])}{_cell(row["PC3"])}</tr>'
    return f"""
<div class="section">
  <h2>二、PCA 因子载荷（18因子 → 3主成分，累计解释 {sum(ev):.1%}）</h2>
  <table class="data-table">
    <thead><tr><th>因子</th>{pc_headers}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p style="font-size:.85em;color:#888;margin-top:6px">
    绝对值 ≥ 0.28 加粗显示；蓝色正载荷，红色负载荷。
  </p>
</div>"""


def sec_k_selection(pca_sil, km_sil):
    def sil_rows(df, tracker_k):
        best_k = int(df.loc[df["silhouette"].idxmax(), "k"])
        out = ""
        for _, r in df.iterrows():
            k = int(r["k"]); s = r["silhouette"]
            bg  = "background:#eaf4fb;font-weight:bold" if k == tracker_k else ""
            tag = ""
            if k == tracker_k and k == best_k:
                tag = '<span class="current-badge">最优·选用</span>'
            elif k == tracker_k:
                tag = '<span style="color:#2980b9;font-size:.82em">← 策略选用</span>'
            elif k == best_k:
                tag = '<span class="current-badge">统计最优</span>'
            out += f'<tr style="{bg}"><td>{k}</td><td>{s:.4f}</td><td>{tag}</td></tr>'
        return out

    img_pca = _img_b64(OUT_DIR / "pca_kmeans_elbow_silhouette.png")
    img_km  = _img_b64(OUT_DIR / "kmeans_elbow_silhouette.png")
    imgs = ""
    if img_pca: imgs += f'<img src="{img_pca}" style="max-width:49%;margin-right:1%">'
    if img_km:  imgs += f'<img src="{img_km}"  style="max-width:49%">'

    return f"""
<div class="section">
  <h2>三、K 值选择（轮廓系数 / 肘部图）</h2>
  <div style="display:flex;gap:40px;flex-wrap:wrap;margin-bottom:16px">
    <div><h3>PCA+KMeans（18因子）</h3>
      <table class="data-table" style="width:220px">
        <thead><tr><th>K</th><th>轮廓系数</th><th></th></tr></thead>
        <tbody>{sil_rows(pca_sil, 6)}</tbody>
      </table>
    </div>
    <div><h3>KMeans（6核心因子）</h3>
      <table class="data-table" style="width:220px">
        <thead><tr><th>K</th><th>轮廓系数</th><th></th></tr></thead>
        <tbody>{sil_rows(km_sil, 6)}</tbody>
      </table>
    </div>
  </div>
  {imgs}
</div>"""


def sec_cluster_cards(perf, labels):
    total  = len(labels)
    counts = labels.value_counts().sort_index()
    cards  = ""
    for cid, cfg in CLUSTER_CONFIG.items():
        cnt = int(counts.get(cid, 0))
        pct = cnt / total if total else 0
        sub = perf[perf["regime"] == cid].sort_values("sharpe", ascending=False)
        best_txt = "".join(
            f'<span class="best">{r["asset"]} Sharpe {r["sharpe"]:+.2f}</span> '
            for _, r in sub.head(2).iterrows()
        )
        cards += f"""
    <div class="card" style="background:{cfg['color']}">
      <h4>C{cid}</h4>
      <div class="label">{cfg['name']}</div>
      <div class="meta">仓位 {cfg['position']:.0%} · 动量 {cfg['mom_weight']:.0%} / 基本面 {1-cfg['mom_weight']:.0%}</div>
      <div class="meta">{cnt} 周（{pct:.1%}）</div>
      <div style="margin-top:8px">{best_txt}</div>
    </div>"""
    return f"""
<div class="section">
  <h2>四、Cluster 配置（K=6）</h2>
  <div class="cluster-cards">{cards}</div>
</div>"""


def sec_cluster_detail(centers, perf, labels):
    total  = len(labels)
    counts = labels.value_counts().sort_index()
    out    = '<div class="section"><h2>五、各 Cluster 详情</h2>'
    for cid, cfg in CLUSTER_CONFIG.items():
        cnt = int(counts.get(cid, 0))
        pct = cnt / total if total else 0
        c   = cfg["color"]
        sub = perf[perf["regime"] == cid].sort_values("sharpe", ascending=False)

        feat_rows = ""
        if cid in centers.index:
            row = centers.loc[cid]
            for col in KEY_FEATURES:
                if col in row.index:
                    feat_rows += (
                        f'<tr><td style="width:140px;font-size:.85em">{FACTOR_CN.get(col,col)}</td>'
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
        out += f"""
  <div class="cluster-detail">
    <div class="cluster-header">
      <div class="cluster-dot" style="background:{c}"></div>
      <span style="font-size:1.1em;font-weight:bold;color:{c}">C{cid} · {cfg['name']}</span>
      <span style="color:#888;font-size:.88em">{cnt} 周 ({pct:.1%}) &nbsp;|&nbsp;
        仓位 {cfg['position']:.0%} · 动量 {cfg['mom_weight']:.0%} · 基本面 {1-cfg['mom_weight']:.0%}
      </span>
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
    out += "</div>"
    return out


def sec_performance_table(perf):
    rows = ""
    for cid, cfg in CLUSTER_CONFIG.items():
        sub   = perf[perf["regime"] == cid].sort_values("sharpe", ascending=False)
        first = True
        for _, r in sub.iterrows():
            span = (
                f'<td rowspan="{len(sub)}" style="font-weight:bold;color:{cfg["color"]};vertical-align:middle">'
                f'C{cid}<br>{cfg["name"]}</td>' if first else ""
            )
            first = False
            rows += (
                f'<tr>{span}<td>{r["asset"]}</td>'
                + _ret_td(r["ann_return"])
                + f'<td>{r["ann_vol"]:.1%}</td>'
                + _sharpe_td(r["sharpe"])
                + f'<td>{r["max_drawdown"]:.1%}</td>'
                + f'<td>{r["win_rate"]:.1%}</td>'
                + f'<td>{r["weeks"]}({r["pct_of_total"]:.1%})</td></tr>'
            )
    return f"""
<div class="section">
  <h2>六、各 Cluster 资产表现汇总</h2>
  <table class="data-table">
    <thead><tr>
      <th>Cluster</th><th>资产</th><th>年化收益</th><th>年化波动</th>
      <th>Sharpe</th><th>最大回撤</th><th>胜率</th><th>样本周数</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def sec_dist(labels):
    total  = len(labels)
    counts = labels.value_counts().sort_index()
    bars   = ""
    for cid, cfg in CLUSTER_CONFIG.items():
        cnt = int(counts.get(cid, 0))
        pct = cnt / total if total else 0
        bars += f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
      <div style="width:160px;font-size:.88em;font-weight:bold;color:{cfg['color']}">C{cid} {cfg['name']}</div>
      <div style="flex:1;background:#eee;border-radius:4px;height:18px;position:relative">
        <div style="background:{cfg['color']};width:{pct*100:.1f}%;height:100%;border-radius:4px"></div>
      </div>
      <div style="width:90px;font-size:.85em;color:#555">{cnt}周 ({pct:.1%})</div>
    </div>"""
    span = f"{labels.index[0].strftime('%Y-%m')} ~ {labels.index[-1].strftime('%Y-%m')}"
    return f"""
<div class="section">
  <h2>七、历史 Cluster 分布（共 {total} 周，{span}）</h2>
  {bars}
</div>"""


# ── CSS ─────────────────────────────────────────────────────────────────────

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
.card .label{font-size:1.2em;font-weight:bold;margin-bottom:8px}
.card .meta{font-size:.82em;opacity:.9;margin:4px 0}
.card .best{background:rgba(255,255,255,.2);border-radius:6px;padding:4px 8px;margin:2px;display:inline-block;font-size:.85em}
.cluster-detail{border:1px solid #e0e0e0;border-radius:8px;padding:20px;margin-bottom:20px}
.cluster-header{display:flex;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap}
.cluster-dot{width:16px;height:16px;border-radius:50%;flex-shrink:0}
img{max-width:100%;border-radius:8px;margin:10px 0}
.data-table{border-collapse:collapse;width:100%;font-size:.92em;margin-bottom:16px}
.data-table th{background:#2c3e50;color:white;padding:10px 14px;text-align:center}
.data-table td{padding:9px 14px;text-align:center;border-bottom:1px solid #eee}
.data-table tr:hover td{background:#f8f9fa}
.insight{background:#eaf4fb;border-left:4px solid #3498db;padding:14px 18px;border-radius:0 8px 8px 0;margin:12px 0;font-size:.95em}
.current-badge{display:inline-block;background:#1e8449;color:white;font-size:.62em;padding:3px 10px;border-radius:20px;vertical-align:middle;margin-left:10px;font-weight:normal}
.stat-box{text-align:center;min-width:100px}
.stat-label{font-size:.82em;color:#888;margin-bottom:4px}
.stat-val{font-size:2em;font-weight:bold}
"""


# ── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    print("正在运行 K=6 分析（约需 10~20 秒）...")
    (labels, centers, perf, loadings,
     ev, pca_sil, km_sil, gmm_met, pca_exp) = run_k6_analysis()

    today = date.today().strftime("%Y-%m-%d")
    body  = "\n".join([
        sec_current(labels),
        sec_method(pca_sil, km_sil, gmm_met, perf, labels),
        sec_pca_loadings(loadings, ev),
        sec_k_selection(pca_sil, km_sil),
        sec_cluster_cards(perf, labels),
        sec_cluster_detail(centers, perf, labels),
        sec_performance_table(perf),
        sec_dist(labels),
    ])

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>大宗商品轮动模型 — 宏观周期Regime分析报告 v4（K=6）</title>
<style>{CSS}</style>
</head>
<body>
<div class="cover">
  <h1>大宗商品轮动模型</h1>
  <p>宏观周期 Regime 分析报告 v4 &nbsp;·&nbsp; PCA+KMeans K=6</p>
  <p>生成日期：{today}</p>
</div>
<div class="container">{body}</div>
</body>
</html>"""

    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"报告已生成：{HTML_OUT.resolve()}")


if __name__ == "__main__":
    main()
