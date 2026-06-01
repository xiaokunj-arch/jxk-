# -*- coding: utf-8 -*-
"""
大宗商品轮动模型 — 完整版一键运行脚本
包含：ML基本面权重 + IC动态权重 + Regime仓位控制

使用方式：
    python3 run_full.py

修改下方"参数配置"区域即可调整策略，无需改动其他代码。
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

os.chdir(Path(__file__).parent)

from config import WORKBOOK_PATH
from rotation_model import (
    load_weekly_prices, load_weekly_factors,
    BacktestConfig, run_backtest, build_etf_weights,
    ASSETS, zscore_row, build_ml_fund_scores, calc_perf,
)
from regime_tracker import get_regime_series

# ══════════════════════════════════════════════════════════
#  参数配置（按需修改）
# ══════════════════════════════════════════════════════════
MOM_W_SHORT   = 0.3    # 短期动量（4周）权重
MOM_W_MID     = 0.4    # 中期动量（12周）权重
MOM_W_LONG    = 0.3    # 长期动量（52周）权重

IC_WINDOW     = 52     # IC滚动回溯周数
IVW_WEEKS     = 10     # 反波动率加权：波动率回溯周数
TREND_WEEKS   = 0      # 趋势过滤回溯周数（0=禁用）
CASH_THR      = 0.0    # 最低持仓分数（z-score，低于此值不配置）
COST_BPS      = 0.0    # 交易成本（bps）

USE_ML        = True   # 是否启用 ML 基本面权重（Ridge）
USE_IC        = True   # 是否启用 IC 动态基本面权重
USE_REGIME    = True   # 是否启用 Regime 仓位控制

USE_VOL_TARGET     = True  # 是否启用波动率目标仓位
VOL_TARGET         = 0.15  # 目标年化波动率
VOL_TARGET_LOOKBACK = 27   # 波动率计算回溯周数

OUTPUT_EXCEL  = True   # 是否输出 Excel 报告
OUTPUT_PATH   = Path("model_outputs_full")
# ══════════════════════════════════════════════════════════


def _rolling_corr_fast(X: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    T, F = X.shape
    ic_out = np.zeros((T, F), dtype=float)
    if T < window:
        return ic_out
    XW = np.moveaxis(sliding_window_view(X, window, axis=0), -1, 1)
    yW = sliding_window_view(y, window)
    mx = XW.mean(axis=1); my = yW.mean(axis=1, keepdims=True)
    xc = XW - mx[:, None, :]; yc = yW - my
    cov = (xc * yc[:, :, None]).mean(axis=1)
    denom = xc.std(axis=1) * yc.std(axis=1, keepdims=True)
    valid = denom > 1e-12
    ic_out[window-1:] = np.where(valid, cov / np.where(valid, denom, 1.0), 0.0)
    return ic_out


def compute_ic_fund(factor_matrix: pd.DataFrame,
                    weekly_ret: pd.DataFrame,
                    ic_window: int) -> pd.DataFrame:
    common_idx = factor_matrix.index
    X = factor_matrix.values.astype(float)
    fund = pd.DataFrame(0.0, index=common_idx, columns=ASSETS)
    for asset in ASSETS:
        if asset not in weekly_ret.columns:
            continue
        y = weekly_ret[asset].shift(-1).reindex(common_idx).fillna(0.0).values.astype(float)
        ic = _rolling_corr_fast(X, y, ic_window)
        abs_sum = np.abs(ic).sum(axis=1)
        abs_sum[abs_sum < 1e-12] = np.nan
        fund[asset] = np.nan_to_num((ic * X).sum(axis=1) / abs_sum, nan=0.0)
    return fund


def build_factor_matrix(factors: dict, prices: pd.DataFrame) -> pd.DataFrame:
    common_idx = prices.index

    def chg(key):
        monthly = {'pmi', 'ppi', 'cn_pmi', 'cn_ppi', 'real_rate', 'credit_impulse',
                   'gold_oi', 'silver_oi', 'copper_inventory', 'oil_inventory', 'coal_inventory'}
        p = 1 if key in monthly else 4
        s = factors.get(key, pd.Series(dtype=float))
        if s.empty:
            return pd.Series(0.0, index=common_idx)
        return s.reindex(common_idx).ffill().pct_change(p).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def zsc(key, w=52):
        s = factors.get(key, pd.Series(dtype=float))
        if s.empty:
            return pd.Series(0.0, index=common_idx)
        s2 = s.reindex(common_idx).ffill()
        mu = s2.rolling(w, min_periods=w // 2).mean()
        sd = s2.rolling(w, min_periods=w // 2).std(ddof=0)
        return ((s2 - mu) / sd.replace(0, np.nan)).fillna(0.0)

    gs = (prices['黄金'] / prices['白银']).replace([np.inf, -np.inf], np.nan)
    macro_gs = (gs / gs.rolling(52, min_periods=26).mean() - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return pd.DataFrame({
        'real_rate':      -chg('real_rate'),
        'dxy_pos':         chg('dxy'),
        'dxy_neg':        -chg('dxy'),
        'vix_pos':         chg('vix'),
        'vix_neg':        -chg('vix'),
        'pmi':             chg('pmi'),
        'ppi':             chg('ppi'),
        'cn_pmi':          chg('cn_pmi'),
        'cn_pmi_inv':     -chg('cn_pmi'),
        'cn_ppi':          chg('cn_ppi'),
        'fxi':             chg('fxi'),
        'fxi_inv':        -chg('fxi'),
        'ttf':             chg('ttf'),
        'gold_oi':        -chg('gold_oi'),
        'silver_oi':      -chg('silver_oi'),
        'gold_cot':        zsc('gold_cot'),
        'silver_cot':     -zsc('silver_cot'),
        'copper_cot':     -zsc('copper_cot'),
        'oil_cot':         zsc('oil_cot'),
        'gs_ratio':        macro_gs,
        'us10y':           chg('us10y'),
        'yield_spread':    chg('yield_spread'),
        'credit_impulse':  chg('credit_impulse'),
    }, index=common_idx)


def main():
    print("=" * 55)
    print("大宗商品轮动模型（完整版）")
    print(f"  动量权重: 短{MOM_W_SHORT} / 中{MOM_W_MID} / 长{MOM_W_LONG}")
    print(f"  IC={IC_WINDOW}w  IVW={IVW_WEEKS}w  趋势={TREND_WEEKS}w  cash={CASH_THR}")
    print(f"  ML={'ON' if USE_ML else 'OFF'}  IC={'ON' if USE_IC else 'OFF'}  Regime={'ON' if USE_REGIME else 'OFF'}")
    print("=" * 55)

    print("\n[1/5] 加载数据...")
    prices  = load_weekly_prices(WORKBOOK_PATH)
    prices  = prices[prices.index >= "2016-01-01"]
    factors = load_weekly_factors(WORKBOOK_PATH)
    common_idx = prices.index
    weekly_ret = prices.reindex(columns=ASSETS).pct_change()
    print(f"      数据区间: {prices.index[0].date()} ~ {prices.index[-1].date()}  ({len(prices)} 周)")

    print("[2/5] 计算 Regime...")
    regime_df = None
    if USE_REGIME:
        regime_df = get_regime_series(factors, prices)
        cur = regime_df.iloc[-1]
        print(f"      当前: C{int(cur.cluster)} {cur['name']}  仓位={cur.position:.0%}  动量权重={cur.mom_weight:.0%}")

    print("[3/5] 计算基本面信号...")
    ml_fund = None
    ic_fund_df = None

    if USE_ML:
        print("      ML Ridge 回归（约1分钟）...")
        ml_fund = build_ml_fund_scores(prices, factors)

    if USE_IC:
        print(f"      IC 滚动相关（{IC_WINDOW}周窗口，约30秒）...")
        fm = build_factor_matrix(factors, prices)
        ic_fund_df = compute_ic_fund(fm, weekly_ret, IC_WINDOW)

    if ml_fund is not None and ic_fund_df is not None:
        fund = (ml_fund.reindex(index=common_idx, columns=ASSETS).fillna(0.0)
                + ic_fund_df.reindex(columns=ASSETS)) / 2.0
    elif ml_fund is not None:
        fund = ml_fund.reindex(index=common_idx, columns=ASSETS).fillna(0.0)
    elif ic_fund_df is not None:
        fund = ic_fund_df.reindex(columns=ASSETS)
    else:
        fund = pd.DataFrame(0.0, index=common_idx, columns=ASSETS)

    print("[4/5] 构建信号面板并运行回测...")
    # 归一化动量权重
    _t = MOM_W_SHORT + MOM_W_MID + MOM_W_LONG
    ws, wm, wl = MOM_W_SHORT / _t, MOM_W_MID / _t, MOM_W_LONG / _t

    mom_raw = (ws * prices.pct_change(4)
             + wm * prices.pct_change(12)
             + wl * prices.pct_change(52))
    mom_z = zscore_row(mom_raw.reindex(columns=ASSETS).fillna(0.0))

    fund_ri  = fund.reindex(columns=ASSETS)
    fund_std = fund_ri.rolling(52, min_periods=26).std()
    fund_z   = zscore_row(fund_ri.div(fund_std).fillna(0.0))

    if regime_df is not None:
        mw   = regime_df['mom_weight'].reindex(common_idx).ffill().fillna(0.6)
        fw_r = regime_df['ic_weight'].reindex(common_idx).ffill().fillna(0.4)
        score = mom_z.mul(mw, axis=0) + fund_z.mul(fw_r, axis=0)
    else:
        score = 0.6 * mom_z + 0.4 * fund_z
    score = score.where(weekly_ret.notna(), np.nan)

    vol_12w = weekly_ret.rolling(IVW_WEEKS).std() * np.sqrt(52)
    trend_mom = (prices.pct_change(TREND_WEEKS).reindex(columns=ASSETS)
                 if TREND_WEEKS > 0
                 else pd.DataFrame(np.nan, index=common_idx, columns=ASSETS))

    panel = pd.concat({
        'price':      prices.reindex(columns=ASSETS),
        'weekly_ret': weekly_ret.reindex(columns=ASSETS),
        'mom_raw':    mom_raw.reindex(columns=ASSETS),
        'fund_raw':   fund_ri,
        'score':      score.reindex(columns=ASSETS),
        'vol_12w':    vol_12w.reindex(columns=ASSETS),
        'trend_mom':  trend_mom,
    }, axis=1)

    cfg = BacktestConfig(
        cost_bps=COST_BPS,
        no_constraints=True,
        use_ivw=False,
        ivw_weeks=IVW_WEEKS,
        trend_filter_weeks=TREND_WEEKS,
        cash_threshold=CASH_THR,
        mom_w_short=ws, mom_w_mid=wm, mom_w_long=wl,
        use_vol_target=USE_VOL_TARGET,
        vol_target=VOL_TARGET,
        vol_target_lookback=VOL_TARGET_LOOKBACK,
    )
    cfg.max_weight_per_asset = 1.0
    cfg.max_weight_per_sector = 1.0
    cfg.max_turnover = 2.0
    if regime_df is not None:
        cfg.regime_pos_series     = regime_df['position']
        cfg.regime_cluster_series = regime_df['cluster']

    weights, strat_ret = run_backtest(panel, cfg)
    nav = (1.0 + strat_ret.fillna(0.0)).cumprod()

    print("[5/5] 计算绩效指标...")
    p = calc_perf(strat_ret)
    calmar = p['ann_return'] / abs(p['max_drawdown']) if abs(p['max_drawdown']) > 1e-6 else float('nan')

    # ── 分年度收益 ──
    annual = strat_ret.groupby(strat_ret.index.year).apply(lambda x: (1 + x).prod() - 1)

    # ── 最新持仓 ──
    latest = weights.iloc[-1]
    latest_date = weights.index[-1].date()

    print()
    print("═" * 55)
    print("  绩效摘要（全样本）")
    print("═" * 55)
    print(f"  年化收益  : {p['ann_return']:+.2%}")
    print(f"  年化波动  : {p['ann_vol']:.2%}")
    print(f"  Sharpe    : {p['sharpe']:.3f}")
    print(f"  最大回撤  : {p['max_drawdown']:.2%}")
    print(f"  Calmar    : {calmar:.3f}")
    print(f"  胜率      : {p['win_rate']:.1%}")
    print()
    print("  分年度收益:")
    for yr, ret in annual.items():
        bar = "█" * int(abs(ret) * 100 // 5)
        sign = "+" if ret >= 0 else "-"
        print(f"    {yr}: {ret:+.1%}  {sign}{bar}")
    print()
    print(f"  最新调仓日: {latest_date}")
    print(f"  最新入选  : {latest.get('chosen', '')}")
    for a in ASSETS:
        w = float(latest.get(f'w_{a}', 0.0))
        if w > 0.001:
            print(f"    {a}: {w:.1%}")
    print("═" * 55)

    if OUTPUT_EXCEL:
        OUTPUT_PATH.mkdir(exist_ok=True)
        import io
        buf = io.BytesIO()
        etf_df = build_etf_weights(weights)
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            pd.DataFrame([
                ('年化收益',  f"{p['ann_return']:+.2%}"),
                ('年化波动',  f"{p['ann_vol']:.2%}"),
                ('Sharpe',    f"{p['sharpe']:.3f}"),
                ('最大回撤',  f"{p['max_drawdown']:.2%}"),
                ('Calmar',    f"{calmar:.3f}"),
                ('胜率',      f"{p['win_rate']:.1%}"),
                ('最新调仓日', str(latest_date)),
                ('最新入选',  str(latest.get('chosen', ''))),
            ], columns=['指标', '数值']).to_excel(writer, index=False, sheet_name='绩效摘要')
            annual.to_frame('年度收益').to_excel(writer, sheet_name='年度收益')
            nav.to_frame('nav').to_excel(writer, sheet_name='净值曲线')
            weights.to_excel(writer, sheet_name='周度权重')
            strat_ret.to_frame('strategy_ret').to_excel(writer, sheet_name='周收益')
            etf_df.to_excel(writer, index=False, sheet_name='ETF持仓')

        out_file = OUTPUT_PATH / f"轮动模型完整版_{pd.Timestamp.today().date()}.xlsx"
        out_file.write_bytes(buf.getvalue())
        print(f"\n  Excel 已保存: {out_file.resolve()}")


if __name__ == '__main__':
    main()
