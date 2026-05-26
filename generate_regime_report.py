# -*- coding: utf-8 -*-
"""从 regime_outputs/ 的 CSV 结果生成宏观周期 Regime 分析 HTML 报告（v5）。"""
from __future__ import annotations

import base64
from datetime import date
from pathlib import Path

import pandas as pd

_BASE    = Path(__file__).parent
OUT_DIR  = _BASE / "regime_outputs"
HTML_OUT = _BASE / "regime_report.html"

CLUSTER_CONFIG = {
    0: dict(name="能源/通胀牛市",   position=0.85, mom_weight=0.00, color="#1e8449"),
    1: dict(name="高增长横盘期",    position=0.45, mom_weight=0.25, color="#d68910"),
    2: dict(name="加息紧缩期",     position=0.00, mom_weight=0.05, color="#c0392b"),
    3: dict(name="大宗全面牛市",   position=1.00, mom_weight=0.05, color="#27ae60"),
    4: dict(name="温和复苏期",     position=0.65, mom_weight=0.35, color="#2980b9"),
    5: dict(name="衰退/通缩/避险", position=0.20, mom_weight=0.30, color="#7d3c98"),
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


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _img_b64(path: Path) -> str:
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _bar(val: float, lo: float = -2.0, hi: float = 2.0) -> str:
    pct = (val - lo) / (hi - lo) * 100
    pct = max(0.0, min(100.0, pct))
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


# ── 数据加载 ─────────────────────────────────────────────────────────────────

def load_data():
    centers  = pd.read_csv(OUT_DIR / "pca_kmeans_centers.csv",  index_col=0)
    perf     = pd.read_csv(OUT_DIR / "performance_by_regime_pca_kmeans.csv")
    labels   = pd.read_csv(OUT_DIR / "regime_labels_pca_kmeans.csv",
                           index_col=0, parse_dates=True)
    loadings = pd.read_csv(OUT_DIR / "pca_loadings.csv", index_col=0)
    pca_exp  = pd.read_csv(OUT_DIR / "pca_explained_variance.csv")
    pca_sil  = pd.read_csv(OUT_DIR / "pca_kmeans_silhouette.csv")
    km_sil   = pd.read_csv(OUT_DIR / "kmeans_silhouette.csv")
    gmm_met  = pd.read_csv(OUT_DIR / "gmm_metrics.csv")
    ic_df    = pd.read_csv(OUT_DIR / "factor_ic_by_regime_kmeans.csv")
    try:
        trans_sum = pd.read_csv(OUT_DIR / "regime_transitions_summary.csv")
    except FileNotFoundError:
        trans_sum = pd.DataFrame()
    return centers, perf, labels, loadings, pca_exp, pca_sil, km_sil, gmm_met, ic_df, trans_sum


# ── 各 Section ───────────────────────────────────────────────────────────────

def sec_current(labels: pd.DataFrame) -> str:
    latest = labels.index[-1].strftime("%Y-%m-%d")
    cid    = int(labels.iloc[-1, 0])
    cfg    = CLUSTER_CONFIG[cid]
    consec = 0
    for v in reversed(labels.iloc[:, 0].tolist()):
        if int(v) == cid:
            consec += 1
        else:
            break
    c      = cfg["color"]
    return f"""
<div class="section">
  <h2>⚡ 当前状态（截至 {latest}）</h2>
  <div style="font-size:1.25em;font-weight:bold;color:{c}">
    C{cid}&nbsp;&nbsp;{cfg['name']}
    <span class="current-badge">当前 · 连续{consec}周</span>
  </div>
  <div style="display:flex;gap:32px;margin-top:16px;flex-wrap:wrap">
    <div class="stat-box">
      <div class="stat-label">基准仓位</div>
      <div class="stat-val" style="color:{c}">{cfg['position']:.0%}</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">动量权重</div>
      <div class="stat-val" style="color:#2980b9">{cfg['mom_weight']:.0%}</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">基本面权重</div>
      <div class="stat-val" style="color:#16a085">{1-cfg['mom_weight']:.0%}</div>
    </div>
  </div>
</div>"""


def sec_method(pca_sil: pd.DataFrame, km_sil: pd.DataFrame,
               gmm_met: pd.DataFrame, perf: pd.DataFrame) -> str:
    km_best  = pca_sil.loc[pca_sil["silhouette"].idxmax()]
    pca_best = pca_sil.loc[pca_sil["silhouette"].idxmax()]
    gmm_bic  = gmm_met.loc[gmm_met["bic"].idxmin()]
    gmm_sil  = gmm_met.loc[gmm_met["silhouette"].idxmax()]

    def sharpe_range(method_perf):
        rs = method_perf.groupby("regime")["sharpe"].mean()
        return rs.max(), rs.min(), int(rs.nunique())

    pca_h, pca_l, pca_n = sharpe_range(perf)
    km_perf = pd.read_csv(OUT_DIR / "performance_by_regime_kmeans.csv")
    km_h, km_l, km_n = sharpe_range(km_perf)
    quad_perf = pd.read_csv(OUT_DIR / "performance_by_regime_quadrant.csv")
    quad_h, quad_l, quad_n = sharpe_range(quad_perf)
    gmm_perf  = pd.read_csv(OUT_DIR / "performance_by_regime_gmm.csv")
    gmm_h, gmm_l, gmm_n = sharpe_range(gmm_perf)

    pca5_sil = float(pca_sil[pca_sil["k"] == 6]["silhouette"].values[0])
    pca2_sil = float(pca_sil[pca_sil["k"] == 2]["silhouette"].values[0])

    rows = (
        f'<tr><td>KMeans（6因子）</td><td>{int(km_best["k"])}</td>'
        f'<td>{float(km_sil.loc[km_sil["silhouette"].idxmax(),"silhouette"]):.3f}</td>'
        f'<td>Silhouette</td><td>{km_n}</td><td>{km_h:+.2f}</td><td>{km_l:+.2f}</td>'
        f'<td>{km_h-km_l:.2f}</td></tr>'
        f'<tr style="background:#eaf4fb"><td><b>PCA+KMeans ✓</b></td><td><b>6</b></td>'
        f'<td><b>{pca5_sil:.3f}</b></td><td>策略选用</td><td>{pca_n}</td>'
        f'<td>{pca_h:+.2f}</td><td>{pca_l:+.2f}</td><td><b>{pca_h-pca_l:.2f}</b></td></tr>'
        f'<tr><td>GMM（BIC）</td><td>{int(gmm_bic["k"])}</td>'
        f'<td>{float(gmm_bic["silhouette"]):.3f}</td><td>BIC</td>'
        f'<td>{gmm_n}</td><td>{gmm_h:+.2f}</td><td>{gmm_l:+.2f}</td>'
        f'<td>{gmm_h-gmm_l:.2f}</td></tr>'
        f'<tr><td>增长/通胀四象限</td><td>4</td><td>—</td><td>规则</td>'
        f'<td>{quad_n}</td><td>{quad_h:+.2f}</td><td>{quad_l:+.2f}</td>'
        f'<td>{quad_h-quad_l:.2f}</td></tr>'
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
    「加息紧缩（C2，仓位=0%）」和「衰退避险（C5，黄金保底60%）」等关键状态，
    轮廓系数（{pca5_sil:.3f}）与最优K=2（{pca2_sil:.3f}）差距仅 {pca2_sil-pca5_sil:.4f}，策略价值更高。
  </div>
</div>"""


def sec_pca_loadings(loadings: pd.DataFrame, pca_exp: pd.DataFrame) -> str:
    ev = pca_exp["explained_variance_ratio"].tolist()
    n_pcs = len(ev)
    pc_labels = [
        "（增长/通胀景气）", "（货币/流动性方向）", "（实际利率水平）",
        "（动量/周期）", "（信贷/利差）",
    ]
    pc_headers = "".join(
        f'<th>PC{i+1} {ev[i]:.1%}<br><span style="font-weight:normal;font-size:.85em">'
        f'{pc_labels[i] if i < len(pc_labels) else ""}</span></th>'
        for i in range(n_pcs)
    )
    rows = ""
    for factor, row in loadings.iterrows():
        cn = FACTOR_CN.get(factor, factor)
        def _cell(v):
            w = "bold" if abs(v) >= 0.28 else "normal"
            c = "#2980b9" if v > 0.1 else ("#c0392b" if v < -0.1 else "#888")
            return f'<td style="font-weight:{w};color:{c}">{v:+.3f}</td>'
        cells = "".join(_cell(row[f"PC{i+1}"]) for i in range(n_pcs))
        rows += f'<tr><td style="text-align:left">{cn}</td>{cells}</tr>'
    cumul = pca_exp["cumulative"].iloc[-1]
    return f"""
<div class="section">
  <h2>二、PCA 因子载荷（18因子 → {n_pcs}主成分，累计解释 {cumul:.1%}）</h2>
  <table class="data-table">
    <thead><tr><th>因子</th>{pc_headers}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p style="font-size:.85em;color:#888;margin-top:6px">
    绝对值 ≥ 0.28 加粗显示（主要驱动因子）；蓝色正载荷，红色负载荷。
  </p>
</div>"""


def sec_k_selection(pca_sil: pd.DataFrame, km_sil: pd.DataFrame) -> str:
    def sil_rows(df, best_k, tracker_k=6):
        out = ""
        for _, r in df.iterrows():
            k  = int(r["k"])
            s  = r["silhouette"]
            bg = "background:#eaf4fb;font-weight:bold" if k == best_k else ""
            tag = ""
            if k == best_k and k == tracker_k:
                tag = '<span class="current-badge">最优 · Tracker</span>'
            elif k == best_k:
                tag = '<span class="current-badge">统计最优</span>'
            elif k == tracker_k:
                tag = '<span style="color:#2980b9;font-size:.82em">← Tracker</span>'
            out += f'<tr style="{bg}"><td>{k}</td><td>{s:.4f}</td><td>{tag}</td></tr>'
        return out

    pca_best = int(pca_sil.loc[pca_sil["silhouette"].idxmax(), "k"])
    km_best  = int(km_sil.loc[km_sil["silhouette"].idxmax(), "k"])
    img_pca  = _img_b64(OUT_DIR / "pca_kmeans_elbow_silhouette.png")
    img_km   = _img_b64(OUT_DIR / "kmeans_elbow_silhouette.png")
    imgs = ""
    if img_pca: imgs += f'<img src="{img_pca}" style="max-width:49%;margin-right:1%">'
    if img_km:  imgs += f'<img src="{img_km}"  style="max-width:49%">'

    return f"""
<div class="section">
  <h2>三、K 值选择（轮廓系数 / 肘部图）</h2>
  <div style="display:flex;gap:40px;flex-wrap:wrap;margin-bottom:16px">
    <div>
      <h3>PCA+KMeans（18因子）</h3>
      <table class="data-table" style="width:220px">
        <thead><tr><th>K</th><th>轮廓系数</th><th></th></tr></thead>
        <tbody>{sil_rows(pca_sil, pca_best)}</tbody>
      </table>
    </div>
    <div>
      <h3>KMeans（6核心因子）</h3>
      <table class="data-table" style="width:220px">
        <thead><tr><th>K</th><th>轮廓系数</th><th></th></tr></thead>
        <tbody>{sil_rows(km_sil, km_best)}</tbody>
      </table>
    </div>
  </div>
  {imgs}
</div>"""


def sec_cluster_cards(perf: pd.DataFrame, labels: pd.DataFrame) -> str:
    total  = len(labels)
    counts = labels.iloc[:, 0].value_counts().sort_index()
    cards  = ""
    for cid, cfg in CLUSTER_CONFIG.items():
        cnt = int(counts.get(cid, 0))
        pct = cnt / total if total else 0
        sub = perf[perf["regime"] == cid].sort_values("sharpe", ascending=False)
        best_txt = ""
        for _, r in sub.head(2).iterrows():
            best_txt += (
                f'<span class="best">{r["asset"]} Sharpe {r["sharpe"]:+.2f}</span> '
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


def sec_cluster_detail(centers: pd.DataFrame, perf: pd.DataFrame,
                       labels: pd.DataFrame) -> str:
    total  = len(labels)
    counts = labels.iloc[:, 0].value_counts().sort_index()
    out    = '<div class="section"><h2>五、各 Cluster 详情</h2>'

    for cid, cfg in CLUSTER_CONFIG.items():
        cnt  = int(counts.get(cid, 0))
        pct  = cnt / total if total else 0
        c    = cfg["color"]
        sub  = perf[perf["regime"] == cid].sort_values("sharpe", ascending=False)

        # 宏观特征横条
        feature_rows = ""
        if cid in centers.index:
            row = centers.loc[cid]
            for col in KEY_FEATURES:
                if col in row.index:
                    cn = FACTOR_CN.get(col, col)
                    feature_rows += (
                        f'<tr><td style="width:140px;font-size:.85em">{cn}</td>'
                        f'<td style="padding:4px 8px">{_bar(row[col])}</td></tr>'
                    )

        # 资产表现
        asset_rows = ""
        for _, r in sub.iterrows():
            asset_rows += (
                f'<tr><td>{r["asset"]}</td>'
                + _ret_td(r["ann_return"])
                + f'<td>{r["ann_vol"]:.1%}</td>'
                + _sharpe_td(r["sharpe"])
                + f'<td>{r["max_drawdown"]:.1%}</td>'
                + f'<td>{r["win_rate"]:.1%}</td></tr>'
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
        <table style="width:100%">{feature_rows}</table>
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


def sec_performance_table(perf: pd.DataFrame) -> str:
    rows = ""
    for cid, cfg in CLUSTER_CONFIG.items():
        sub   = perf[perf["regime"] == cid].sort_values("sharpe", ascending=False)
        first = True
        for _, r in sub.iterrows():
            if first:
                span = (f'<td rowspan="{len(sub)}" style="font-weight:bold;color:{cfg["color"]};'
                        f'vertical-align:middle">C{cid}<br>{cfg["name"]}</td>')
                first = False
            else:
                span = ""
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


def sec_factor_ic(ic_df: pd.DataFrame) -> str:
    all_factors = ic_df["factor"].unique().tolist()
    header = "".join(f'<th>{f}</th>' for f in all_factors)
    rows = ""
    for cid, cfg in CLUSTER_CONFIG.items():
        sub = ic_df[ic_df["regime"] == cid].set_index("factor")
        cells = ""
        for f in all_factors:
            if f in sub.index:
                ir = sub.loc[f, "ic_ir"]
                c = "#2980b9" if ir > 0.3 else ("#c0392b" if ir < -0.3 else "#888")
                cells += f'<td style="color:{c}">{ir:.2f}</td>'
            else:
                cells += "<td>—</td>"
        rows += f'<tr><td style="font-weight:bold;color:{cfg["color"]}">C{cid} {cfg["name"]}</td>{cells}</tr>'
    return f"""
<div class="section">
  <h2>七、因子 IC_IR（各 Cluster 下因子预测力）</h2>
  <div style="overflow-x:auto">
  <table class="data-table" style="font-size:.82em">
    <thead><tr><th>Cluster</th>{header}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
  </div>
  <p style="font-size:.84em;color:#888;margin-top:6px">IC_IR = IC均值/IC标准差；|IR|>0.3 蓝/红标注（蓝正红负）</p>
</div>"""


def sec_weekly_history(labels: pd.DataFrame) -> str:
    rows_by_year: dict[int, list[str]] = {}
    col = labels.columns[0]
    for dt, row in labels.iterrows():
        cid  = int(row[col])
        cfg  = CLUSTER_CONFIG[cid]
        c    = cfg["color"]
        year = dt.year
        rows_by_year.setdefault(year, [])
        rows_by_year[year].append(
            f'<tr>'
            f'<td>{dt.strftime("%Y-%m-%d")}</td>'
            f'<td><span style="background:{c};color:white;padding:2px 10px;'
            f'border-radius:12px;font-size:.82em;font-weight:bold">C{cid}</span></td>'
            f'<td style="color:{c};font-weight:bold">{cfg["name"]}</td>'
            f'<td>{cfg["position"]:.0%}</td>'
            f'</tr>'
        )

    year_blocks = ""
    for year in sorted(rows_by_year.keys(), reverse=True):
        yr_rows = "".join(rows_by_year[year])
        year_blocks += f"""
  <details {"open" if year >= labels.index[-1].year - 1 else ""}>
    <summary style="cursor:pointer;font-weight:bold;font-size:1em;padding:8px 4px;
      border-bottom:1px solid #eee;color:#2c3e50">{year} 年（{len(rows_by_year[year])} 周）</summary>
    <table class="data-table" style="margin-top:8px">
      <thead><tr><th>日期</th><th>Cluster</th><th>名称</th><th>仓位</th></tr></thead>
      <tbody>{yr_rows}</tbody>
    </table>
  </details>"""

    return f"""
<div class="section">
  <h2>九、每周 Cluster 记录（共 {len(labels)} 周）</h2>
  <p style="font-size:.88em;color:#888;margin-bottom:16px">
    近两年默认展开，更早年份点击展开。
  </p>
  {year_blocks}
</div>"""


def sec_dist(labels: pd.DataFrame) -> str:
    total  = len(labels)
    counts = labels.iloc[:, 0].value_counts().sort_index()
    bars   = ""
    for cid, cfg in CLUSTER_CONFIG.items():
        cnt = int(counts.get(cid, 0))
        pct = cnt / total if total else 0
        bars += f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
      <div style="width:148px;font-size:.88em;font-weight:bold;color:{cfg['color']}">C{cid} {cfg['name']}</div>
      <div style="flex:1;background:#eee;border-radius:4px;height:18px;position:relative">
        <div style="background:{cfg['color']};width:{pct*100:.1f}%;height:100%;border-radius:4px"></div>
      </div>
      <div style="width:90px;font-size:.85em;color:#555">{cnt}周 ({pct:.1%})</div>
    </div>"""
    span = f"{labels.index[0].strftime('%Y-%m')} ~ {labels.index[-1].strftime('%Y-%m')}"
    return f"""
<div class="section">
  <h2>八、历史 Cluster 分布（共 {total} 周，{span}）</h2>
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
    (centers, perf, labels, loadings,
     pca_exp, pca_sil, km_sil, gmm_met,
     ic_df, trans_sum) = load_data()

    today = date.today().strftime("%Y-%m-%d")
    body  = "\n".join([
        sec_current(labels),
        sec_method(pca_sil, km_sil, gmm_met, perf),
        sec_pca_loadings(loadings, pca_exp),
        sec_k_selection(pca_sil, km_sil),
        sec_cluster_cards(perf, labels),
        sec_cluster_detail(centers, perf, labels),
        sec_performance_table(perf),
        sec_factor_ic(ic_df),
        sec_dist(labels),
        sec_weekly_history(labels),
    ])

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>大宗商品轮动模型 — 宏观周期Regime分析报告 v5</title>
<style>{CSS}</style>
</head>
<body>
<div class="cover">
  <h1>大宗商品轮动模型</h1>
  <p>宏观周期 Regime 分析报告 v5 &nbsp;·&nbsp; PCA+KMeans K=6</p>
  <p>生成日期：{today}</p>
</div>
<div class="container">{body}</div>
</body>
</html>"""

    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"报告已生成：{HTML_OUT.resolve()}")


if __name__ == "__main__":
    main()
