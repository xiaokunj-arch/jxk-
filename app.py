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
    build_ml_fund_scores,
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


@st.cache_data(show_spinner="ML学习基本面因子权重（首次运行约1分钟）...")
def compute_ml_fund(_prices, _factors, min_train_weeks=104):
    return build_ml_fund_scores(_prices, _factors, min_train_weeks)


@st.cache_data(show_spinner="计算滚动IC权重（约需20秒）...")
def compute_ic_fund(_factor_matrix: pd.DataFrame, _weekly_ret: pd.DataFrame, ic_window: int = 52) -> pd.DataFrame:
    """
    按滚动Pearson IC动态加权基本面因子，替代固定权重。
    - 每个(因子, 资产)对独立计算滚动IC
    - fund[asset] = sum(IC_i * factor_i) / sum(|IC_i|)
    - IC接近0的因子自动权重趋近0，无需手写规则
    """
    common_idx = _factor_matrix.index
    fund = pd.DataFrame(0.0, index=common_idx, columns=ASSETS)

    for asset in ASSETS:
        if asset not in _weekly_ret.columns:
            continue
        ret_fwd = _weekly_ret[asset].shift(-1)
        weighted = pd.Series(0.0, index=common_idx)
        total_abs_ic = pd.Series(0.0, index=common_idx)

        for col in _factor_matrix.columns:
            fseries = _factor_matrix[col]
            ic = fseries.rolling(ic_window, min_periods=ic_window // 2).corr(ret_fwd).fillna(0.0)
            weighted += ic * fseries
            total_abs_ic += ic.abs()

        fund[asset] = weighted.div(total_abs_ic.replace(0, np.nan)).fillna(0.0)

    return fund


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
    fund_override: pd.DataFrame | None = None,
    use_ic_fund: bool = False,
    ic_window: int = 52,
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

    def aligned_zscore(key: str, window: int = 52) -> pd.Series:
        s = factors.get(key, pd.Series(dtype=float))
        if s.empty:
            return pd.Series(0.0, index=common_idx)
        s2 = s.reindex(common_idx).ffill()
        mu = s2.rolling(window, min_periods=window // 2).mean()
        sd = s2.rolling(window, min_periods=window // 2).std(ddof=0)
        z = (s2 - mu) / sd.replace(0, np.nan)
        return z.fillna(0.0)

    # 正向因子
    macro_pmi = aligned_change("pmi", 4)
    macro_ppi = aligned_change("ppi", 4)
    macro_ttf = aligned_change("ttf", 4)
    macro_fxi = aligned_change("fxi", 4)
    macro_cn_pmi = aligned_change("cn_pmi", 4)
    macro_cn_ppi = aligned_change("cn_ppi", 4)
    # 反向因子：统一取反，正值=看涨信号
    macro_real          = -aligned_change("real_rate", 4)
    macro_dxy           = -aligned_change("dxy",       4)   # 美元↑→大宗承压（铜/原油用）
    macro_dxy_pos       =  aligned_change("dxy",       4)   # 美元↑→正向（ML：黄金/白银正相关）
    macro_vix           = -aligned_change("vix",       4)   # VIX↑→风险偏好↓（原油用）
    macro_vix_pos       =  aligned_change("vix",       4)   # VIX↑→正向（ML：黄金避险正相关）
    macro_cn_pmi_inv    = -aligned_change("cn_pmi",    4)   # 中国PMI↑→承压（ML：煤炭负相关）
    gold_oi_chg         = -aligned_change("gold_oi",   4)  # 黄金持仓↑→空头增加，承压
    silver_oi_chg_inv   = -aligned_change("silver_oi", 4)  # 白银持仓↑→空头增加，承压
    copper_fxi_inv      = -macro_fxi                        # FXI对铜价IC为负（煤炭保留正向）
    # COT管理资金净头寸（滚动z-score水平位置，IC为负的取反：过度拥挤时价格反转）
    macro_gold_cot   =  aligned_zscore("gold_cot",   52)   # 正向
    macro_silver_cot = -aligned_zscore("silver_cot", 52)   # 反向（拥挤做多→反转）
    macro_copper_cot = -aligned_zscore("copper_cot", 52)   # 反向（拥挤做多→反转）
    macro_oil_cot    =  aligned_zscore("oil_cot",    52)   # 正向

    gs_ratio = (weekly_prices["黄金"] / weekly_prices["白银"]).replace([np.inf, -np.inf], np.nan)
    gs_ma52 = gs_ratio.rolling(52, min_periods=26).mean()
    macro_gs = (gs_ratio / gs_ma52 - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if fund_override is not None:
        fund = fund_override.reindex(index=common_idx, columns=ASSETS).fillna(0.0)
    elif use_ic_fund:
        # IC动态加权：把所有因子打包，滚动计算每个(因子,资产)对的IC，IC作为权重
        factor_matrix = pd.DataFrame({
            "real_rate":    macro_real,
            "dxy_pos":      macro_dxy_pos,
            "dxy_neg":      macro_dxy,
            "vix_pos":      macro_vix_pos,
            "vix_neg":      macro_vix,
            "pmi":          macro_pmi,
            "ppi":          macro_ppi,
            "cn_pmi":       macro_cn_pmi,
            "cn_pmi_inv":   macro_cn_pmi_inv,
            "cn_ppi":       macro_cn_ppi,
            "fxi":          macro_fxi,
            "fxi_inv":      copper_fxi_inv,
            "ttf":          macro_ttf,
            "gold_oi":      gold_oi_chg,
            "silver_oi":    silver_oi_chg_inv,
            "gold_cot":     macro_gold_cot,
            "silver_cot":   macro_silver_cot,
            "copper_cot":   macro_copper_cot,
            "oil_cot":      macro_oil_cot,
            "gs_ratio":     macro_gs,
        }, index=common_idx)
        weekly_ret_for_ic = weekly_prices.reindex(columns=ASSETS).pct_change()
        fund = compute_ic_fund(factor_matrix, weekly_ret_for_ic, ic_window)
    else:
        fund = pd.DataFrame(index=common_idx, columns=ASSETS, dtype=float)
        fund["黄金"] = (
            fw["gold_real_rate"] * macro_real
            + fw["gold_dxy"] * macro_dxy_pos
            + fw["gold_vix"] * macro_vix_pos
            + fw["gold_oi"] * gold_oi_chg
            + fw["gold_cot"] * macro_gold_cot
        )
        fund["白银"] = (
            fw["silver_real_rate"] * macro_real
            + fw["silver_dxy"] * macro_dxy_pos
            + fw["silver_gs"] * macro_gs
            + fw["silver_oi"] * silver_oi_chg_inv
            + fw["silver_cot"] * macro_silver_cot
        )
        fund["铜"] = (
            fw["copper_real_rate"] * macro_real
            + fw["copper_dxy"] * macro_dxy
            + fw["copper_pmi"] * macro_pmi
            + fw["copper_fxi"] * copper_fxi_inv
            + fw["copper_cot"] * macro_copper_cot
        )
        fund["原油"] = (
            fw["oil_dxy"] * macro_dxy
            + fw["oil_pmi"] * macro_pmi
            + fw["oil_vix"] * macro_vix
            + fw["oil_cot"] * macro_oil_cot
        )
        fund["煤炭"] = (
            fw["coal_cn_pmi"] * macro_cn_pmi_inv
            + fw["coal_fxi"] * macro_fxi
            + fw["coal_cn_ppi"] * macro_cn_ppi
        )

    mom_z = zscore_row(mom_raw.reindex(columns=ASSETS).fillna(0.0))
    fund_reindexed = fund.reindex(columns=ASSETS)
    fund_ts_std = fund_reindexed.rolling(52, min_periods=26).std()
    fund_normalized = fund_reindexed.div(fund_ts_std).fillna(0.0)
    fund_z = zscore_row(fund_normalized)
    score = cfg.score_momentum_weight * mom_z + cfg.score_fundamental_weight * fund_z
    score = score.where(weekly_ret.notna(), np.nan)

    vol_12w = weekly_ret.reindex(columns=ASSETS).rolling(cfg.ivw_weeks).std() * np.sqrt(52)

    trend_mom = (
        weekly_prices.pct_change(cfg.trend_filter_weeks).reindex(columns=ASSETS)
        if cfg.trend_filter_weeks > 0
        else pd.DataFrame(np.nan, index=weekly_prices.index, columns=ASSETS)
    )

    return pd.concat(
        {
            "price": weekly_prices.reindex(columns=ASSETS),
            "weekly_ret": weekly_ret.reindex(columns=ASSETS),
            "mom_raw": mom_raw.reindex(columns=ASSETS),
            "fund_raw": fund.reindex(columns=ASSETS),
            "score": score.reindex(columns=ASSETS),
            "vol_12w": vol_12w,
            "trend_mom": trend_mom,
        },
        axis=1,
    )


def run_model(prices, facts, fw, cost_bps, mom_w, mom_lb, mom_weights, use_ivw=False, ivw_weeks=12, cash_threshold=-99.0, top_n_free=5, market_cash_threshold=-99.0, market_full_threshold=-99.0, max_position_ratio=1.0, fund_override=None, trend_filter_weeks=0, use_ic_fund=False, ic_window=52):
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
        cash_threshold=cash_threshold,
        top_n_free=top_n_free,
        market_cash_threshold=market_cash_threshold,
        market_full_threshold=market_full_threshold,
        max_position_ratio=max_position_ratio,
        trend_filter_weeks=trend_filter_weeks,
    )
    cfg.max_weight_per_asset = 1.0
    cfg.max_weight_per_sector = 1.0
    cfg.max_turnover = 2.0
    panel = build_signal_panel_custom(prices, facts, cfg, fw, fund_override=fund_override,
                                      use_ic_fund=use_ic_fund, ic_window=ic_window)
    weights, strategy_ret = run_backtest(panel, cfg)
    nav = (1.0 + strategy_ret.fillna(0.0)).cumprod().rename("nav")
    return weights, strategy_ret, nav


# ─────────────────────────────────────────────
# 导出 Excel
# ─────────────────────────────────────────────
def build_excel(weights: pd.DataFrame, strategy_ret: pd.Series, nav: pd.Series, weeks: int | None) -> bytes:
    r = strategy_ret.dropna() if weeks is None else strategy_ret.dropna().tail(weeks)
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
    use_market_cash = st.checkbox("启用空仓", value=False, help="基于5品种等权平均动量控制整体仓位：低于空仓线→0%，高于满仓线→100%，中间线性过渡")
    market_cash_threshold = -99.0
    market_full_threshold = -99.0
    if use_market_cash:
        market_cash_threshold = st.slider("空仓线（等权动量低于此值→0%仓位）", -0.15, 0.05, -0.05, 0.01,
                                          help="5品种等权平均动量低于此值时完全空仓")
        market_full_threshold = st.slider("满仓线（等权动量高于此值→100%仓位）", -0.10, 0.10, 0.0, 0.01,
                                          help="5品种等权平均动量高于此值时按模型满仓；两线之间按比例持仓（半仓过渡）")
    max_position_ratio = st.slider("整体仓位上限", 0.1, 1.0, 1.0, 0.05,
                                   help="模型最多投入的总仓位比例。例如设0.5则模型最多半仓，无论动量如何")
    use_trend_filter = st.checkbox("启用趋势过滤", value=False, help="单品种自身动量为负时不持有，只做上升趋势的资产；可与空仓叠加使用")
    trend_filter_weeks = 0
    if use_trend_filter:
        trend_filter_weeks = st.slider("趋势过滤回溯周数", 4, 52, 12, 1,
                                       help="若资产过去N周收益为负则排除，不参与当周选股")
    use_cash = st.checkbox("启用低分过滤", value=False, help="单品种综合得分低于阈值时不配置；高于阈值的品种全部配置；全部低于阈值时完全空仓")
    cash_threshold = -99.0
    top_n_free = 5
    if use_cash:
        cash_threshold = st.slider("最低持仓分数（z-score）", -2.0, 0.5, -0.5, 0.1,
                                   help="低于此分数的品种直接排除；高于的品种全部配置（softmax加权）")
    if use_ivw:
        ivw_weeks = st.slider("波动率回溯周数", 4, 52, 12, 1, help="计算反波动率加权所用的滚动波动率窗口")
    use_ml_fund = st.checkbox("ML基本面权重（Ridge）", value=False,
                              help="用机器学习（Ridge回归）从数据自动学习各宏观因子的权重，替代手写规则。首次启用需约1分钟计算。")
    use_ic_fund = st.checkbox("IC动态基本面权重", value=False,
                              help="按各因子与下周收益的滚动相关系数（IC）动态加权，IC高的因子自动多用，IC低的自动少用。首次启用需约20秒。")
    ic_window = 52
    if use_ic_fund:
        ic_window = st.slider("IC回溯窗口（周）", 26, 156, 52, 13,
                              help="计算滚动IC所用的历史窗口，越大越稳定但适应性越低")
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
    g_rr  = st.slider("实际利率",    -1.0, 1.0,  0.35, 0.05, key="g_rr")
    g_dxy = st.slider("美元指数", -1.0, 1.0, 0.25, 0.05, key="g_dxy")
    g_vix = st.slider("VIX恐慌指数", -1.0, 1.0, 0.25, 0.05, key="g_vix")
    g_oi  = st.slider("COMEX 持仓量", -1.0, 1.0,  0.15, 0.05, key="g_oi")
    g_cot = st.slider("COT管理资金", -1.0, 1.0,  0.20, 0.05, key="g_cot")

    st.header("🥈 白银")
    s_rr  = st.slider("实际利率",      -1.0, 1.0,  0.25, 0.05, key="s_rr")
    s_dxy = st.slider("美元指数", -1.0, 1.0, 0.25, 0.05, key="s_dxy")
    s_gs  = st.slider("金银比",        -1.0, 1.0,  0.35, 0.05, key="s_gs")
    s_oi  = st.slider("COMEX 持仓量",  -1.0, 1.0,  0.15, 0.05, key="s_oi")
    s_cot = st.slider("COT管理资金",   -1.0, 1.0,  0.20, 0.05, key="s_cot")

    st.header("🔩 铜")
    c_rr  = st.slider("实际利率",    -1.0, 1.0,  0.25, 0.05, key="c_rr")
    c_dxy = st.slider("美元指数",    -1.0, 1.0,  0.20, 0.05, key="c_dxy")
    c_pmi = st.slider("PMI",         -1.0, 1.0,  0.30, 0.05, key="c_pmi")
    c_fxi = st.slider("FXI中国需求", -1.0, 1.0,  0.25, 0.05, key="c_fxi")
    c_cot = st.slider("COT管理资金", -1.0, 1.0,  0.20, 0.05, key="c_cot")

    st.header("🛢️ 原油")
    o_dxy = st.slider("美元指数",    -1.0, 1.0,  0.40, 0.05, key="o_dxy")
    o_pmi = st.slider("PMI",         -1.0, 1.0,  0.25, 0.05, key="o_pmi")
    o_vix = st.slider("VIX恐慌指数", -1.0, 1.0,  0.35, 0.05, key="o_vix")
    o_cot = st.slider("COT管理资金", -1.0, 1.0,  0.25, 0.05, key="o_cot")

    st.header("🪨 煤炭")
    coal_cn_pmi = st.slider("中国制造业PMI", -1.0, 1.0, 0.40, 0.05, key="coal_cn_pmi")
    coal_fxi    = st.slider("FXI中国需求",   -1.0, 1.0,  0.30, 0.05, key="coal_fxi")
    coal_cn_ppi = st.slider("中国PPI",       -1.0, 1.0,  0.30, 0.05, key="coal_cn_ppi")

    st.divider()
    # ── 权重约束校验 ──
    _warns = []
    if abs(mom_weight - 0.5) > 0.49:
        _warns.append(f"动量/基本面权重极端（{mom_weight:.0%} / {1-mom_weight:.0%}），信号可能失衡")
    for _name, _vals in [
        ("黄金", [g_rr, g_dxy, g_vix, g_oi, g_cot]),
        ("白银", [s_rr, s_dxy, s_gs, s_oi, s_cot]),
        ("铜",   [c_rr, c_dxy, c_pmi, c_fxi, c_cot]),
        ("原油", [o_dxy, o_pmi, o_vix, o_cot]),
        ("煤炭", [coal_cn_pmi, coal_fxi, coal_cn_ppi]),
    ]:
        _s = sum(abs(v) for v in _vals)
        if _s > 2.0:
            _warns.append(f"{_name}因子权重绝对值之和 = {_s:.1f}，建议控制在 1~2 以内")
    if _warns:
        for _w in _warns:
            st.warning(_w)

    run_btn = st.button("▶ 运行回测", type="primary", use_container_width=True)

    st.divider()
    with st.expander("🔍 数据诊断", expanded=False):
        import os
        st.write(f"**数据文件:** `{WORKBOOK_PATH}`")
        st.write(f"**文件存在:** {WORKBOOK_PATH.exists()}")
        if WORKBOOK_PATH.exists():
            st.write(f"**文件大小:** {os.path.getsize(WORKBOOK_PATH):,} bytes")
        st.write(f"**价格数据:** {len(weekly_prices)} 周，{weekly_prices.index[0].date()} ~ {weekly_prices.index[-1].date()}")
        st.write(f"**价格列:** {list(weekly_prices.columns)}")
        st.write("**因子数据行数:**")
        for k, v in factors.items():
            if hasattr(v, '__len__') and len(v) > 0:
                st.write(f"  - `{k}`: {len(v)} 行，{v.index[0].date()} ~ {v.index[-1].date()}")
            else:
                st.write(f"  - `{k}`: 空")

# ─────────────────────────────────────────────
# 收集参数 & 运行
# ─────────────────────────────────────────────
fund_weights = {
    "gold_real_rate": g_rr, "gold_dxy": g_dxy, "gold_vix": g_vix, "gold_oi": g_oi, "gold_cot": g_cot,
    "silver_real_rate": s_rr, "silver_dxy": s_dxy, "silver_gs": s_gs, "silver_oi": s_oi, "silver_cot": s_cot,
    "copper_real_rate": c_rr, "copper_dxy": c_dxy, "copper_pmi": c_pmi, "copper_fxi": c_fxi, "copper_cot": c_cot,
    "oil_dxy": o_dxy, "oil_pmi": o_pmi, "oil_vix": o_vix, "oil_cot": o_cot,
    "coal_cn_pmi": coal_cn_pmi, "coal_fxi": coal_fxi, "coal_cn_ppi": coal_cn_ppi,
}

# 动量权重归一化，确保三项之和为 1
_mw_total = mw_short + mw_mid + mw_long
if _mw_total > 0:
    mw_short, mw_mid, mw_long = mw_short / _mw_total, mw_mid / _mw_total, mw_long / _mw_total
mom_weights = {"w_short": mw_short, "w_mid": mw_mid, "w_long": mw_long}

# 首次或点击按钮时运行
if "result" not in st.session_state or run_btn:
    ml_fund = compute_ml_fund(weekly_prices, factors) if use_ml_fund else None
    with st.spinner("回测计算中..."):
        weights, strategy_ret, nav = run_model(
            weekly_prices, factors, fund_weights, cost_bps, mom_weight, mom_lookback, mom_weights,
            use_ivw, ivw_weeks if use_ivw else 12, cash_threshold, top_n_free,
            market_cash_threshold, market_full_threshold, max_position_ratio,
            fund_override=ml_fund,
            trend_filter_weeks=trend_filter_weeks,
            use_ic_fund=use_ic_fund and ml_fund is None,
            ic_window=ic_window,
        )
    st.session_state.result = (weights, strategy_ret, nav)

weights, strategy_ret, nav = st.session_state.result
perf = calc_perf(strategy_ret)

# 基准：等权持有五个品种
bm_ret = weekly_prices[ASSETS].pct_change().reindex(strategy_ret.index).fillna(0.0).mean(axis=1)
bm_nav = (1.0 + bm_ret).cumprod().rename("bm_nav")
perf_bm = calc_perf(bm_ret)


# ─────────────────────────────────────────────
# 主区域：绩效指标
# ─────────────────────────────────────────────
st.subheader("绩效指标（全样本）")
m1, m2, m3, m4, m5 = st.columns(5)

def fmt(v, pct=False):
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return f"{v:.1%}" if pct else f"{v:.2f}"

def delta(strat, bm, pct=False):
    if isinstance(strat, float) and math.isnan(strat): return None
    if isinstance(bm,    float) and math.isnan(bm):    return None
    diff = strat - bm
    return f"{diff:+.1%}" if pct else f"{diff:+.2f}"

m1.metric("年化收益", fmt(perf["ann_return"], pct=True), delta(perf["ann_return"], perf_bm["ann_return"], pct=True), delta_color="inverse")
m2.metric("年化波动", fmt(perf["ann_vol"],    pct=True), delta(perf["ann_vol"],    perf_bm["ann_vol"],    pct=True), delta_color="normal")
m3.metric("Sharpe",   fmt(perf["sharpe"]),               delta(perf["sharpe"],     perf_bm["sharpe"]),               delta_color="inverse")
m4.metric("最大回撤", fmt(perf["max_drawdown"], pct=True), delta(perf["max_drawdown"], perf_bm["max_drawdown"], pct=True), delta_color="inverse")
m5.metric("胜率",     fmt(perf["win_rate"],    pct=True), delta(perf["win_rate"],   perf_bm["win_rate"],   pct=True), delta_color="inverse")
st.caption(f"↑↓ vs 等权基准（年化收益 {fmt(perf_bm['ann_return'], pct=True)}，Sharpe {fmt(perf_bm['sharpe'])}，最大回撤 {fmt(perf_bm['max_drawdown'], pct=True)}）")

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
fig_nav.add_trace(go.Scatter(
    x=bm_nav.index, y=bm_nav.values,
    name="等权基准",
    line=dict(color="#f5a623", width=1.5, dash="dot"),
    hovertemplate="%{x|%Y-%m-%d}  等权基准: %{y:.4f}<extra></extra>",
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
colors = ["#d62728" if v >= 0 else "#2ca02c" for v in annual.values]
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
        [None, 52, 104, 26],
        format_func=lambda x: "全部（从开始到现在）" if x is None else (f"近 {x} 周（约 {x//52} 年）" if x >= 52 else f"近 {x} 周"),
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
