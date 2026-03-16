# -*- coding: utf-8 -*-
"""
大宗商品轮动模型 — Streamlit 交互前端

运行方式:
    streamlit run app.py
"""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rotation_model import (
    ASSETS,
    WORKBOOK_PATH,
    BacktestConfig,
    build_etf_weights,
    calc_perf,
    load_weekly_factors,
    load_weekly_prices,
    run_backtest,
    zscore_row,
)

# ─────────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="大宗商品轮动模型",
    page_icon="📊",
    layout="wide",
)

st.title("📊 大宗商品轮动模型")


# ─────────────────────────────────────────────
# 数据加载（缓存，只读一次）
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="读取数据...")
def load_data():
    prices = load_weekly_prices(WORKBOOK_PATH)
    factors = load_weekly_factors(WORKBOOK_PATH)
    return prices, factors


if not WORKBOOK_PATH.exists():
    st.error(f"找不到数据文件：{WORKBOOK_PATH}，请确认 Excel 与本脚本在同一目录。")
    st.stop()

weekly_prices, factors = load_data()


# ─────────────────────────────────────────────
# 信号构建（支持自定义基本面权重）
# ─────────────────────────────────────────────
def build_signal_panel_custom(
    weekly_prices: pd.DataFrame,
    factors: dict,
    cfg: BacktestConfig,
    fw: dict,  # fund_weights
) -> pd.DataFrame:
    weekly_ret = weekly_prices.pct_change()
    mom_short  = weekly_prices.pct_change(cfg.mom_short_weeks)
    mom_mid    = weekly_prices.pct_change(cfg.momentum_lookback_weeks)
    mom_long   = weekly_prices.pct_change(cfg.mom_long_weeks)
    mom_raw = (
        cfg.mom_w_short * mom_short
        + cfg.mom_w_mid * mom_mid
        + cfg.mom_w_long * mom_long
    )
    common_idx = weekly_prices.index

    def aligned_change(key: str, periods: int = 4) -> pd.Series:
        s = factors.get(key, pd.Series(dtype=float))
        if s.empty:
            return pd.Series(0.0, index=common_idx)
        s2 = s.reindex(common_idx).ffill()
        return s2.pct_change(periods).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    macro_real = -aligned_change("real_rate", 4)
    macro_dxy = -aligned_change("dxy", 4)
    macro_pmi = aligned_change("pmi", 4)
    gold_oi_chg = aligned_change("gold_oi", 4)
    copper_inv_chg = aligned_change("copper_inventory", 4)
    macro_ppi = aligned_change("ppi", 4)
    macro_ttf = aligned_change("ttf", 4)
    macro_vix = -aligned_change("vix", 4)

    gs_ratio = (weekly_prices["黄金"] / weekly_prices["白银"]).replace([np.inf, -np.inf], np.nan)
    gs_ma52 = gs_ratio.rolling(52, min_periods=26).mean()
    macro_gs = (gs_ratio / gs_ma52 - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fund = pd.DataFrame(index=common_idx, columns=ASSETS, dtype=float)
    fund["黄金"] = fw["gold_real_rate"] * macro_real + fw["gold_dxy"] * macro_dxy + fw["gold_oi"] * gold_oi_chg
    silver_oi_chg = aligned_change("silver_oi", 4)
    fund["白银"] = (
        fw["silver_real_rate"] * macro_real
        + fw["silver_dxy"] * macro_dxy
        + fw["silver_gs"] * macro_gs
        + fw["silver_oi"] * silver_oi_chg
    )
    fund["铜"] = (
        fw["copper_real_rate"] * macro_real
        + fw["copper_dxy"] * macro_dxy
        + fw["copper_pmi"] * macro_pmi
        + fw["copper_inventory"] * copper_inv_chg
    )
    fund["原油"] = (
        fw["oil_dxy"] * macro_dxy
        + fw["oil_pmi"] * macro_pmi
        + fw["oil_vix"] * macro_vix
    )
    fund["煤炭"] = (
        fw["coal_ttf"] * macro_ttf
        + fw["coal_ppi"] * macro_ppi
        + fw["coal_pmi"] * macro_pmi
    )

    mom_z = zscore_row(mom_raw.reindex(columns=ASSETS).fillna(0.0))
    fund_z = zscore_row(fund.reindex(columns=ASSETS).fillna(0.0))
    score = cfg.score_momentum_weight * mom_z + cfg.score_fundamental_weight * fund_z
    score = score.where(weekly_ret.notna(), np.nan)

    vol_12w = weekly_ret.reindex(columns=ASSETS).rolling(cfg.ivw_weeks).std() * np.sqrt(52)

    return pd.concat(
        {
            "price": weekly_prices.reindex(columns=ASSETS),
            "weekly_ret": weekly_ret.reindex(columns=ASSETS),
            "mom_raw": mom_raw.reindex(columns=ASSETS),
            "fund_raw": fund.reindex(columns=ASSETS),
            "score": score.reindex(columns=ASSETS),
            "vol_12w": vol_12w,
        },
        axis=1,
    )


def run_model(prices, facts, fw, cost_bps, mom_w, mom_lb, mom_weights, use_ivw=False, ivw_weeks=12):
    cfg = BacktestConfig(
        cost_bps=cost_bps,
        momentum_lookback_weeks=mom_lb,
        score_momentum_weight=mom_w,
        score_fundamental_weight=round(1.0 - mom_w, 4),
        no_constraints=True,
        mom_w_short=mom_weights["w_short"],
        mom_w_mid=mom_weights["w_mid"],
        mom_w_long=mom_weights["w_long"],
        use_ivw=use_ivw,
        ivw_weeks=ivw_weeks,
    )
    cfg.max_weight_per_asset = 1.0
    cfg.max_weight_per_sector = 1.0
    cfg.max_turnover = 2.0
    panel = build_signal_panel_custom(prices, facts, cfg, fw)
    weights, strategy_ret = run_backtest(panel, cfg)
    nav = (1.0 + strategy_ret.fillna(0.0)).cumprod().rename("nav")
    return weights, strategy_ret, nav


# ─────────────────────────────────────────────
# 导出 Excel
# ─────────────────────────────────────────────
def build_excel(weights: pd.DataFrame, strategy_ret: pd.Series, nav: pd.Series, weeks: int) -> bytes:
    r = strategy_ret.dropna().tail(weeks)
    if r.empty:
        raise ValueError("收益序列为空。")
    start, end = r.index[0], r.index[-1]
    nav_1y = nav.loc[(nav.index >= start) & (nav.index <= end)]
    w_1y = weights.loc[(weights.index >= start) & (weights.index <= end)]

    cum = (1 + r).cumprod()
    total_ret = float(cum.iloc[-1] - 1)
    ann_ret = float((1 + total_ret) ** (52 / len(r)) - 1)
    ann_vol = float(r.std(ddof=0) * (52**0.5))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    max_dd = float((cum / cum.cummax() - 1).min())
    win_rate = float((r > 0).mean())
    latest = w_1y.iloc[-1] if not w_1y.empty else None

    summary = pd.DataFrame(
        [
            ("区间开始", start.date().isoformat()),
            ("区间结束", end.date().isoformat()),
            ("周数", len(r)),
            ("区间收益", f"{total_ret:.2%}"),
            ("年化收益", f"{ann_ret:.2%}"),
            ("年化波动", f"{ann_vol:.2%}"),
            ("Sharpe", f"{sharpe:.2f}"),
            ("最大回撤", f"{max_dd:.2%}"),
            ("胜率", f"{win_rate:.2%}"),
            ("平均周换手", f"{w_1y['turnover'].mean():.2%}" if not w_1y.empty else ""),
            ("累计交易成本", f"{w_1y['cost'].sum():.4%}" if not w_1y.empty else ""),
            ("最新调仓日", w_1y.index[-1].date().isoformat() if not w_1y.empty else ""),
            ("最新入选品种", str(latest["chosen"]) if latest is not None else ""),
        ],
        columns=["指标", "数值"],
    )

    latest_pos = pd.DataFrame(
        [(a, float(latest.get(f"w_{a}", 0.0)) if latest is not None else 0.0) for a in ASSETS],
        columns=["品种", "最新权重"],
    )

    weekly = r.to_frame("strategy_ret")
    weekly["cum_return"] = (1 + weekly["strategy_ret"]).cumprod() - 1
    weekly.index.name = "date"

    etf_df = build_etf_weights(weights)
    etf_1y = etf_df[(etf_df["date"] >= start) & (etf_df["date"] <= end)]

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="近期汇总")
        latest_pos.to_excel(writer, index=False, sheet_name="最新持仓")
        weekly.to_excel(writer, sheet_name="周收益与累计收益")
        nav_1y.to_frame().to_excel(writer, sheet_name="净值曲线")
        w_1y.to_excel(writer, sheet_name="周度权重与换手")
        etf_1y.to_excel(writer, index=False, sheet_name="ETF映射持仓")
    return buf.getvalue()


# ─────────────────────────────────────────────
# 侧边栏：参数设置
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 全局参数")
    cost_bps = st.slider("交易成本 (bps)", 0.0, 50.0, 0.0, 1.0)
    use_ivw = st.checkbox("反波动率加权", value=False, help="勾选后按各资产波动率倒数调整权重，波动小的资产多配，有助于降低回撤")
    if use_ivw:
        ivw_weeks = st.slider("波动率回溯周数", 4, 52, 12, 1, help="计算反波动率加权所用的滚动波动率窗口")
    mom_weight = st.slider("动量权重", 0.0, 1.0, 0.6, 0.05,
                           help="基本面权重 = 1 - 动量权重")
    st.caption(f"→ 动量 {mom_weight:.0%}  /  基本面 {1-mom_weight:.0%}")
    mom_lookback = st.slider("中期动量回溯周数", 4, 26, 12, 1)

    st.subheader("动量分项权重")
    mw_short = st.slider("短期动量（4周）",  0.0, 1.0, 0.25, 0.05, key="mw_short")
    mw_mid   = st.slider("中期动量（12周）", 0.0, 1.0, 0.50, 0.05, key="mw_mid")
    mw_long  = st.slider("长期动量（52周）", 0.0, 1.0, 0.25, 0.05, key="mw_long")
    mw_total = mw_short + mw_mid + mw_long
    if mw_total > 0:
        st.caption(f"合计：{mw_total:.2f}")
    else:
        st.warning("动量分项权重全为 0，将使用等权。")
        mw_short = mw_mid = mw_long = 1.0 / 3

    st.divider()
    st.header("🏅 黄金")
    g_rr  = st.slider("实际利率",    -1.0, 1.0,  0.45, 0.05, key="g_rr")
    g_dxy = st.slider("美元指数",    -1.0, 1.0,  0.35, 0.05, key="g_dxy")
    g_oi  = st.slider("COMEX 持仓量", -1.0, 1.0,  0.20, 0.05, key="g_oi")

    st.header("🥈 白银")
    s_rr  = st.slider("实际利率",      -1.0, 1.0,  0.25, 0.05, key="s_rr")
    s_dxy = st.slider("美元指数",      -1.0, 1.0,  0.25, 0.05, key="s_dxy")
    s_gs  = st.slider("金银比",        -1.0, 1.0,  0.35, 0.05, key="s_gs")
    s_oi  = st.slider("COMEX 持仓量",  -1.0, 1.0,  0.15, 0.05, key="s_oi")

    st.header("🔩 铜")
    c_rr  = st.slider("实际利率",    -1.0, 1.0,  0.25, 0.05, key="c_rr")
    c_dxy = st.slider("美元指数",    -1.0, 1.0,  0.20, 0.05, key="c_dxy")
    c_pmi = st.slider("PMI",         -1.0, 1.0,  0.30, 0.05, key="c_pmi")
    c_inv = st.slider("LME 铜库存",  -1.0, 1.0, -0.25, 0.05, key="c_inv")

    st.header("🛢️ 原油")
    o_dxy = st.slider("美元指数",    -1.0, 1.0,  0.40, 0.05, key="o_dxy")
    o_pmi = st.slider("PMI",         -1.0, 1.0,  0.25, 0.05, key="o_pmi")
    o_vix = st.slider("VIX恐慌指数", -1.0, 1.0,  0.35, 0.05, key="o_vix")

    st.header("🪨 煤炭")
    coal_ttf = st.slider("TTF天然气",  -1.0, 1.0,  0.50, 0.05, key="coal_ttf")
    coal_ppi = st.slider("PPI",       -1.0, 1.0,  0.30, 0.05, key="coal_ppi")
    coal_pmi = st.slider("PMI",       -1.0, 1.0,  0.20, 0.05, key="coal_pmi")

    st.divider()
    # ── 权重约束校验 ──
    _warns = []
    if abs(mom_weight - 0.5) > 0.49:
        _warns.append(f"动量/基本面权重极端（{mom_weight:.0%} / {1-mom_weight:.0%}），信号可能失衡")
    for _name, _vals in [
        ("黄金", [g_rr, g_dxy, g_oi]),
        ("白银", [s_rr, s_dxy, s_gs, s_oi]),
        ("铜",   [c_rr, c_dxy, c_pmi, c_inv]),
        ("原油", [o_dxy, o_pmi, o_vix]),
        ("煤炭", [coal_ttf, coal_ppi, coal_pmi]),
    ]:
        _s = sum(abs(v) for v in _vals)
        if _s > 2.0:
            _warns.append(f"{_name}因子权重绝对值之和 = {_s:.1f}，建议控制在 1~2 以内")
    if _warns:
        for _w in _warns:
            st.warning(_w)

    run_btn = st.button("▶ 运行回测", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# 收集参数 & 运行
# ─────────────────────────────────────────────
fund_weights = {
    "gold_real_rate": g_rr, "gold_dxy": g_dxy, "gold_oi": g_oi,
    "silver_real_rate": s_rr, "silver_dxy": s_dxy, "silver_gs": s_gs, "silver_oi": s_oi,
    "copper_real_rate": c_rr, "copper_dxy": c_dxy, "copper_pmi": c_pmi, "copper_inventory": c_inv,
    "oil_dxy": o_dxy, "oil_pmi": o_pmi, "oil_vix": o_vix,
    "coal_ttf": coal_ttf, "coal_ppi": coal_ppi, "coal_pmi": coal_pmi,
}

# 动量权重归一化，确保三项之和为 1
_mw_total = mw_short + mw_mid + mw_long
if _mw_total > 0:
    mw_short, mw_mid, mw_long = mw_short / _mw_total, mw_mid / _mw_total, mw_long / _mw_total
mom_weights = {"w_short": mw_short, "w_mid": mw_mid, "w_long": mw_long}

# 首次或点击按钮时运行
if "result" not in st.session_state or run_btn:
    with st.spinner("回测计算中..."):
        weights, strategy_ret, nav = run_model(
            weekly_prices, factors, fund_weights, cost_bps, mom_weight, mom_lookback, mom_weights,
            use_ivw, ivw_weeks if use_ivw else 12
        )
    st.session_state.result = (weights, strategy_ret, nav)

weights, strategy_ret, nav = st.session_state.result
perf = calc_perf(strategy_ret)


# ─────────────────────────────────────────────
# 主区域：绩效指标
# ─────────────────────────────────────────────
st.subheader("绩效指标（全样本）")
m1, m2, m3, m4, m5 = st.columns(5)

def fmt(v, pct=False):
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return f"{v:.1%}" if pct else f"{v:.2f}"

m1.metric("年化收益", fmt(perf["ann_return"], pct=True))
m2.metric("年化波动", fmt(perf["ann_vol"], pct=True))
m3.metric("Sharpe",   fmt(perf["sharpe"]))
m4.metric("最大回撤", fmt(perf["max_drawdown"], pct=True))
m5.metric("胜率",     fmt(perf["win_rate"], pct=True))

# ─────────────────────────────────────────────
# 净值曲线
# ─────────────────────────────────────────────
st.subheader("策略净值曲线")
fig_nav = go.Figure()
fig_nav.add_trace(go.Scatter(
    x=nav.index, y=nav.values,
    name="策略净值",
    line=dict(color="#1f77b4", width=2),
    hovertemplate="%{x|%Y-%m-%d}  净值: %{y:.4f}<extra></extra>",
))
fig_nav.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis_title="净值",
    hovermode="x unified",
)
st.plotly_chart(fig_nav, use_container_width=True)

# ─────────────────────────────────────────────
# 年度收益
# ─────────────────────────────────────────────
st.subheader("年度收益")
annual = strategy_ret.groupby(strategy_ret.index.year).apply(lambda x: (1 + x).prod() - 1)
colors = ["#2ca02c" if v >= 0 else "#d62728" for v in annual.values]
fig_ann = go.Figure(go.Bar(
    x=annual.index.astype(str),
    y=annual.values,
    marker_color=colors,
    text=[f"{v:.1%}" for v in annual.values],
    textposition="outside",
    hovertemplate="%{x}年  %{y:.2%}<extra></extra>",
))
fig_ann.update_layout(
    height=320,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis_tickformat=".0%",
    yaxis_title="收益率",
)
st.plotly_chart(fig_ann, use_container_width=True)

# ─────────────────────────────────────────────
# 最新持仓 + 导出
# ─────────────────────────────────────────────
col_pos, col_exp = st.columns([1, 1])

with col_pos:
    st.subheader("最新持仓")
    latest = weights.iloc[-1]
    pos_df = pd.DataFrame({
        "品种": ASSETS,
        "权重": [f"{float(latest.get(f'w_{a}', 0.0)):.1%}" for a in ASSETS],
    })
    st.dataframe(pos_df, hide_index=True, use_container_width=True)
    st.caption(f"最新调仓日：{weights.index[-1].date()}")
    st.caption(f"入选品种：{latest.get('chosen', '')}")

with col_exp:
    st.subheader("导出 Excel 报告")
    weeks_opt = st.selectbox(
        "导出窗口",
        [52, 26, 104],
        format_func=lambda x: f"近 {x} 周（约 {x//52} 年）" if x >= 52 else f"近 {x} 周",
    )
    try:
        excel_bytes = build_excel(weights, strategy_ret, nav, weeks=weeks_opt)
        st.download_button(
            label="⬇ 下载 Excel 报告",
            data=excel_bytes,
            file_name=f"轮动模型_{pd.Timestamp.today().date()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"生成报告失败：{e}")

# ─────────────────────────────────────────────
# 分时段绩效表
# ─────────────────────────────────────────────
with st.expander("分时段绩效"):
    last_dt = strategy_ret.index.max()
    periods = [
        ("近1个月",  last_dt - pd.DateOffset(months=1)),
        ("近3个月",  last_dt - pd.DateOffset(months=3)),
        ("近6个月",  last_dt - pd.DateOffset(months=6)),
        ("近1年",    last_dt - pd.DateOffset(years=1)),
        ("全样本",   strategy_ret.index.min()),
    ]
    # 近1个月、近3个月显示区间累计收益，避免年化放大失真
    SHORT_LABELS = {"近1个月", "近3个月"}
    rows = []
    for label, since in periods:
        subset = strategy_ret[strategy_ret.index >= since]
        p = calc_perf(subset)
        weeks = int(p["weeks"])
        if label in SHORT_LABELS:
            cum_ret = float((1 + subset.fillna(0)).prod() - 1)
            def fmt_f(v): return f"{v:.2f}" if not (isinstance(v, float) and math.isnan(v)) else "N/A"
            row = {
                "时段": label,
                "收益（累计）": f"{cum_ret:.1%}",
                "年化波动": f"{p['ann_vol']:.1%}" if not math.isnan(p['ann_vol']) else "N/A",
                "Sharpe": fmt_f(p["sharpe"]),
                "最大回撤": f"{p['max_drawdown']:.1%}" if not math.isnan(p['max_drawdown']) else "N/A",
                "Calmar": "N/A",
                "胜率": f"{p['win_rate']:.1%}" if not math.isnan(p['win_rate']) else "N/A",
                "周数": str(weeks),
            }
        else:
            def fmt_pct(v): return f"{v:.1%}" if not (isinstance(v, float) and math.isnan(v)) else "N/A"
            def fmt_f(v):   return f"{v:.2f}" if not (isinstance(v, float) and math.isnan(v)) else "N/A"
            row = {
                "时段": label,
                "收益（累计）": f"{p['ann_return']:.1%}（年化）",
                "年化波动": fmt_pct(p["ann_vol"]),
                "Sharpe": fmt_f(p["sharpe"]),
                "最大回撤": fmt_pct(p["max_drawdown"]),
                "Calmar": fmt_f(p["calmar"]),
                "胜率": fmt_pct(p["win_rate"]),
                "周数": str(weeks),
            }
        rows.append(row)
    perf_tbl = pd.DataFrame(rows)
    st.dataframe(perf_tbl, hide_index=True, use_container_width=True)
