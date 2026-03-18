# -*- coding: utf-8 -*-
"""
大宗商品周频多头轮动模型（期货回测 + ETF映射 + 鲁棒性测试）

运行方式:
    python3 rotation_model.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================
# 全局路径与静态配置
# =========================
WORKBOOK_PATH = Path("大宗商品轮动_数据2.xlsx")
OUTPUT_DIR = Path("model_outputs")

ASSETS = ["黄金", "白银", "铜", "原油", "煤炭"]

PRICE_SHEET_POS = {
    "黄金": (0, 1),
    "白银": (2, 3),
    "铜": (4, 5),
    "原油": (6, 7),
    "煤炭": (8, 9),
}

ETF_MAPPING = {
    "黄金": "518880.SH / 159934.SZ（黄金ETF）",
    "白银": "518890.SH（白银LOF，容量较小）",
    "铜": "铜相关主题ETF（大成有色etf）",
    "原油": "162411.SZ / 501018.SH（原油QDII类）",
    "煤炭": "煤炭ETF（515220.SH）",
}

SECTOR_MAP = {
    "黄金": "贵金属",
    "白银": "贵金属",
    "铜": "有色",
    "原油": "能源",
    "煤炭": "能源",
}


@dataclass
class BacktestConfig:
    """策略参数配置。"""
    top_n: int = 2
    momentum_lookback_weeks: int = 12
    cost_bps: float = 0.0
    max_weight_per_asset: float = 0.55
    max_weight_per_sector: float = 0.7
    max_turnover: float = 1.0
    score_momentum_weight: float = 0.6
    score_fundamental_weight: float = 0.4
    start_date: str = "2008-01-01"
    no_constraints: bool = False
    # 多维动量分项参数
    mom_short_weeks: int = 4
    mom_long_weeks: int = 52
    mom_w_short: float = 0.25   # 短期动量（4周）权重
    mom_w_mid: float = 0.50     # 中期动量（12周）权重
    mom_w_long: float = 0.25    # 长期动量（52周）权重
    use_ivw: bool = False        # 是否启用反波动率加权
    ivw_weeks: int = 12          # 反波动率加权的波动率回溯周数
    cash_threshold: float = -99.0  # 最优资产动量低于此值时全仓空仓（-99=禁用）


def parse_args() -> argparse.Namespace:
    """命令行参数入口，便于快速试验不同回测设定。"""
    parser = argparse.ArgumentParser(description="大宗商品周频轮动模型回测")
    parser.add_argument("--cost-bps", type=float, default=0.0, help="单边交易成本（bps），默认0")
    parser.add_argument("--top-n", type=int, default=2, help="每期持仓品种数，默认2")
    parser.add_argument("--start-date", type=str, default="2008-01-01", help="回测起始日期，默认2008-01-01")
    parser.add_argument(
        "--no-constraints",
        action="store_true",
        help="关闭全部组合约束（持仓数量/配比上限/行业上限/换手上限）",
    )
    return parser.parse_args()


def _read_two_col_sheet(path: Path, sheet_name: str) -> pd.Series:
    """读取因子子表（默认前两列为 日期/数值），输出标准化时间序列。"""
    df = pd.read_excel(path, sheet_name=sheet_name)
    if df.empty:
        return pd.Series(dtype=float, name=sheet_name)
    date_col = df.columns[0]
    value_col = df.columns[1]
    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "value"]
    out["date"] = out["date"].map(_parse_mixed_date)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").set_index("date")["value"]
    out.name = sheet_name
    return out


def _weekly_last(series: pd.Series) -> pd.Series:
    """统一到周频（周五），并以前值填充缺口以对齐多源数据。
    截掉周标签 > 今天的不完整周，避免用当周未收盘数据影响信号。
    """
    if series.empty:
        return series
    today = pd.Timestamp.today().normalize()
    return series.sort_index().resample("W-FRI").last().ffill().loc[lambda s: s.index <= today]


def _parse_mixed_date(value) -> pd.Timestamp | pd.NaT:
    """兼容多种日期格式：datetime、Excel序列号、YYYY-MM-DD/ YYYYMMDD 等。"""
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, (datetime, date)):
        return pd.Timestamp(value)

    if isinstance(value, (int, float)):
        # Excel serial date
        if 20000 <= float(value) <= 60000:
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(int(value), unit="D")
        value = str(int(value))

    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return pd.Timestamp(datetime.strptime(s[:10], fmt))
        except Exception:
            continue
    return pd.NaT


def load_weekly_prices(path: Path) -> pd.DataFrame:
    """读取期货价格表，按品种构建周频价格面板。"""
    px = pd.read_excel(path, sheet_name="期货价格")
    out = {}
    for asset, (date_i, price_i) in PRICE_SHEET_POS.items():
        tmp = px.iloc[:, [date_i, price_i]].copy()
        tmp.columns = ["date", "price"]
        tmp["date"] = tmp["date"].map(_parse_mixed_date)
        tmp["price"] = pd.to_numeric(tmp["price"], errors="coerce")
        tmp = tmp.dropna(subset=["date", "price"]).drop_duplicates(subset=["date"], keep="last")
        s = tmp.sort_values("date").set_index("date")["price"]
        out[asset] = _weekly_last(s)
    weekly = pd.concat(out, axis=1).sort_index().dropna(how="all")
    return weekly


def load_weekly_factors(path: Path) -> Dict[str, pd.Series]:
    """
    读取因子子表并统一到周频。
    若某因子缺失，返回空序列并在后续信号中按0处理，保证流程不中断。
    """
    factor_sheet_candidates = {
        "real_rate": ["实际利率"],
        "dxy": ["美元指数DXY", "美元指数"],
        "pmi": ["美国ISM制造业PMI"],
        "ppi": ["美国PPI"],
        "ttf": ["TTF欧洲天然气"],
        "vix": ["VIX恐慌指数"],
        "fxi": ["FXI中国大盘ETF"],
        "gold_oi": ["comex黄金持仓量"],
        "silver_oi": ["comex白银持仓量"],
        "copper_inventory": ["LME铜库存", "铜库存"],
        "oil_inventory": ["原油库存"],
        "coal_inventory": ["煤炭库存"],
        "cn_pmi": ["中国制造业PMI"],
        "cn_ppi": ["中国ppi"],
    }

    xls = pd.ExcelFile(path)
    sheets = set(xls.sheet_names)
    factors: Dict[str, pd.Series] = {}

    for key, candidates in factor_sheet_candidates.items():
        selected = next((s for s in candidates if s in sheets), None)
        if selected is None:
            factors[key] = pd.Series(dtype=float, name=key)
            continue
        s = _read_two_col_sheet(path, selected)
        factors[key] = _weekly_last(s)
    return factors


def zscore_row(df: pd.DataFrame) -> pd.DataFrame:
    """按每一周的截面做标准化，保留“相对强弱”而非绝对水平。"""
    mu = df.mean(axis=1)
    sigma = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sigma, axis=0).fillna(0.0)


def build_signal_panel(weekly_prices: pd.DataFrame, factors: Dict[str, pd.Series], cfg: BacktestConfig) -> pd.DataFrame:
    """
    信号层：
    1) 动量：12-1周（可配置）
    2) 基本面：利率/美元/PMI/库存持仓变化加权
    3) 综合评分：动量与基本面线性组合
    """
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
        """将因子对齐到交易日历后计算变化率。"""
        s = factors.get(key, pd.Series(dtype=float))
        if s.empty:
            return pd.Series(0.0, index=common_idx)
        s2 = s.reindex(common_idx).ffill()
        return s2.pct_change(periods).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 正向因子：上涨→看涨
    macro_pmi = aligned_change("pmi", 4)
    macro_ppi = aligned_change("ppi", 4)
    macro_ttf = aligned_change("ttf", 4)
    macro_fxi = aligned_change("fxi", 4)
    macro_cn_pmi = aligned_change("cn_pmi", 4)
    macro_cn_ppi = aligned_change("cn_ppi", 4)
    # 反向因子：上涨→看跌，统一取反使正值=看涨信号
    macro_real         = -aligned_change("real_rate",  4)  # 实际利率↑→大宗承压
    macro_dxy          = -aligned_change("dxy",        4)  # 美元↑→大宗承压
    macro_vix          = -aligned_change("vix",        4)  # VIX↑→风险偏好↓
    macro_gold_oi_inv  = -aligned_change("gold_oi",    4)  # 黄金持仓↑→空头增加，承压
    macro_silver_oi_inv= -aligned_change("silver_oi",  4)  # 白银持仓↑→空头增加，承压
    macro_copper_fxi_inv = -macro_fxi                       # FXI对铜价IC为负（煤炭保留正向）

    # 金银比均值回归：高于52周均值说明白银相对黄金偏便宜，对白银是正信号
    gs_ratio = (weekly_prices["黄金"] / weekly_prices["白银"]).replace([np.inf, -np.inf], np.nan)
    gs_ma52 = gs_ratio.rolling(52, min_periods=26).mean()
    macro_gs = (gs_ratio / gs_ma52 - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fund = pd.DataFrame(index=common_idx, columns=ASSETS, dtype=float)
    fund["黄金"] = 0.45 * macro_real + 0.35 * macro_dxy + 0.20 * macro_gold_oi_inv
    fund["白银"] = 0.25 * macro_real + 0.25 * macro_dxy + 0.35 * macro_gs + 0.15 * macro_silver_oi_inv
    fund["铜"] = 0.25 * macro_real + 0.20 * macro_dxy + 0.30 * macro_pmi + 0.25 * macro_copper_fxi_inv
    fund["原油"] = 0.40 * macro_dxy + 0.25 * macro_pmi + 0.35 * macro_vix
    fund["煤炭"] = 0.40 * macro_cn_pmi + 0.30 * macro_fxi + 0.30 * macro_cn_ppi

    mom_z = zscore_row(mom_raw.reindex(columns=ASSETS).fillna(0.0))
    fund_reindexed = fund.reindex(columns=ASSETS)
    fund_ts_std = fund_reindexed.rolling(52, min_periods=26).std()
    fund_normalized = fund_reindexed.div(fund_ts_std).fillna(0.0)
    fund_z = zscore_row(fund_normalized)
    score = cfg.score_momentum_weight * mom_z + cfg.score_fundamental_weight * fund_z
    score = score.where(weekly_ret.notna(), np.nan)

    # 12周滚动年化波动率（用于反波动率加权）
    vol_12w = weekly_ret.reindex(columns=ASSETS).rolling(cfg.ivw_weeks).std() * np.sqrt(52)

    panel = pd.concat(
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
    return panel


def _select_top_assets(score_row: pd.Series, cfg: BacktestConfig) -> List[str]:
    """按评分排序选TopN，并施加行业集中度约束。"""
    ranked = score_row.sort_values(ascending=False).dropna().index.tolist()
    chosen: List[str] = []
    sector_weight: Dict[str, float] = {}

    for asset in ranked:
        if len(chosen) >= cfg.top_n:
            break
        sector = SECTOR_MAP.get(asset, "其他")
        # 行业约束用“可实现权重”估算，避免 Top1 时被误判为不可持仓。
        eq_weight_est = min(1.0 / max(cfg.top_n, 1), cfg.max_weight_per_asset)
        sec_w = sector_weight.get(sector, 0.0) + eq_weight_est
        if sec_w > cfg.max_weight_per_sector + 1e-12:
            continue
        chosen.append(asset)
        sector_weight[sector] = sec_w
    return chosen


def _score_to_free_weights(score_row: pd.Series) -> pd.Series:
    """
    无约束模式下的权重生成：
    - 不限制持仓数量（全部可交易品种都可配置）
    - 不限制单品种配比上限
    - 用 softmax 将评分映射成非负且总和为1的权重
    """
    valid = score_row.dropna()
    if valid.empty:
        return pd.Series(0.0, index=ASSETS)
    x = (valid - valid.max()).astype(float)
    ex = np.exp(x)
    w = ex / ex.sum()
    out = pd.Series(0.0, index=ASSETS)
    for a in w.index:
        out[a] = float(w[a])
    return out


def _cap_turnover(target_w: pd.Series, prev_w: pd.Series, max_turnover: float) -> pd.Series:
    """若换手超限，则把目标权重向上期权重线性回拉。"""
    turnover = float((target_w - prev_w).abs().sum())
    if turnover <= max_turnover + 1e-12:
        return target_w
    if turnover == 0:
        return target_w
    alpha = max_turnover / turnover
    blended = prev_w + alpha * (target_w - prev_w)
    total = blended.clip(lower=0).sum()
    if total > 0:
        blended = blended / total
    return blended


def run_backtest(signal_panel: pd.DataFrame, cfg: BacktestConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """
    组合与交易层：
    - 每周生成目标权重
    - 扣减交易成本（bps * 换手）
    - 用下一周收益计算本周调仓后的组合收益
    """
    score = signal_panel["score"].copy()
    rets = signal_panel["weekly_ret"].copy()
    vol = signal_panel["vol_12w"].copy()
    # 最优资产动量（最高的那个），用于大势空仓判断
    best_mom = signal_panel["mom_raw"].max(axis=1)
    score = score[score.index >= pd.Timestamp(cfg.start_date)]
    rets = rets.reindex(score.index)
    vol = vol.reindex(score.index)
    best_mom = best_mom.reindex(score.index).fillna(0.0)

    weight_rows = []
    strat_rets = []
    prev_w = pd.Series(0.0, index=ASSETS)

    for i, dt in enumerate(score.index):
        sc = score.loc[dt]
        if cfg.no_constraints:
            target = _score_to_free_weights(sc)
            chosen = [a for a in ASSETS if target[a] > 1e-10]
        else:
            chosen = _select_top_assets(sc, cfg)
            target = pd.Series(0.0, index=ASSETS)
            if chosen:
                eq_w = 1.0 / len(chosen)
                for c in chosen:
                    target[c] = min(eq_w, cfg.max_weight_per_asset)
            target = _cap_turnover(target, prev_w, cfg.max_turnover)

        # 大势空仓：即使最优资产动量也低于阈值时清空仓位
        if cfg.cash_threshold > -90 and best_mom.loc[dt] < cfg.cash_threshold:
            target = pd.Series(0.0, index=ASSETS)

        # 反波动率加权：将目标权重乘以各资产波动率倒数再归一化
        if cfg.use_ivw:
            vol_row = vol.loc[dt].reindex(ASSETS)
            inv_vol = 1.0 / vol_row.replace(0, np.nan).fillna(0.2)
            inv_vol = inv_vol.where(target > 1e-10, 0.0)
            total_inv = inv_vol.sum()
            if total_inv > 0:
                target = inv_vol / total_inv
        turnover = float((target - prev_w).abs().sum())
        trade_cost = turnover * cfg.cost_bps / 10000.0

        next_ret = rets.iloc[i + 1].fillna(0.0) if i < len(rets.index) - 1 else pd.Series(0.0, index=ASSETS)
        strat_ret = float((target * next_ret).sum() - trade_cost)

        row = {"date": dt, **{f"w_{a}": target[a] for a in ASSETS}, "turnover": turnover, "cost": trade_cost, "chosen": ",".join(chosen)}
        weight_rows.append(row)
        strat_rets.append((dt, strat_ret))
        prev_w = target

    weights = pd.DataFrame(weight_rows).set_index("date")
    nav = pd.Series({d: r for d, r in strat_rets}, name="strategy_ret").sort_index()
    return weights, nav


def calc_perf(returns: pd.Series) -> Dict[str, float]:
    """绩效指标：年化收益/波动、Sharpe、最大回撤、Calmar、胜率。"""
    r = returns.dropna()
    if r.empty:
        return {k: float("nan") for k in ["ann_return", "ann_vol", "sharpe", "max_drawdown", "calmar", "win_rate", "weeks"]}
    nav = (1.0 + r).cumprod()
    dd = nav / nav.cummax() - 1.0
    weeks = len(r)
    ann_return = float(nav.iloc[-1] ** (52 / weeks) - 1.0) if weeks > 0 and nav.iloc[-1] > 0 else float("nan")
    ann_vol = float(r.std(ddof=0) * math.sqrt(52))
    sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")
    max_dd = float(dd.min())
    calmar = ann_return / abs(max_dd) if max_dd < 0 else float("nan")
    win_rate = float((r > 0).mean())
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "weeks": weeks,
    }


def top_drawdowns(nav: pd.Series, top_k: int = 5) -> pd.DataFrame:
    """提取前N大回撤区间（起点、谷底、恢复或截止时间）。"""
    roll_max = nav.cummax()
    dd = nav / roll_max - 1.0
    drawdowns = []
    in_dd = False
    start = peak = trough = None

    for dt, val in dd.items():
        if val < 0 and not in_dd:
            in_dd = True
            start = dt
            peak = roll_max.loc[dt]
            trough = val
            trough_dt = dt
        elif val < 0 and in_dd:
            if val < trough:
                trough = val
                trough_dt = dt
        elif val >= 0 and in_dd:
            drawdowns.append((start, trough_dt, dt, float(trough)))
            in_dd = False
    if in_dd and start is not None:
        drawdowns.append((start, trough_dt, dd.index[-1], float(trough)))

    out = pd.DataFrame(drawdowns, columns=["start", "trough", "recovery_or_end", "drawdown"]).sort_values("drawdown").head(top_k)
    return out


def build_etf_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """将期货权重映射为ETF代理建议，便于实盘参考。"""
    cols = [c for c in weights.columns if c.startswith("w_")]
    out = weights[cols].copy()
    out.columns = [c.replace("w_", "") for c in cols]
    melted = out.reset_index().melt(id_vars="date", var_name="asset", value_name="weight")
    melted["etf_proxy"] = melted["asset"].map(ETF_MAPPING)
    etf = melted[melted["weight"] > 0].sort_values(["date", "weight"], ascending=[True, False])
    return etf


def run_param_sensitivity(signal_panel: pd.DataFrame, base_cfg: BacktestConfig) -> pd.DataFrame:
    """参数敏感性：遍历 TopN 与动量窗口，比较策略稳健性。"""
    rows = []
    for top_n in [1, 2, 3]:
        for lb in [8, 12, 16]:
            cfg = replace(base_cfg, top_n=top_n, momentum_lookback_weeks=lb)
            panel = build_signal_panel(signal_panel["price"], load_weekly_factors(WORKBOOK_PATH), cfg)
            _, ret = run_backtest(panel, cfg)
            perf = calc_perf(ret)
            rows.append({"top_n": top_n, "momentum_lb": lb, **perf})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def run_robust_tests(signal_panel: pd.DataFrame, base_cfg: BacktestConfig) -> pd.DataFrame:
    """鲁棒性测试：成本扰动与单因子化场景。"""
    tests = []
    definitions = [
        ("base", base_cfg, {"mom_w": base_cfg.score_momentum_weight, "fund_w": base_cfg.score_fundamental_weight}),
        ("low_cost", replace(base_cfg, cost_bps=8.0), None),
        ("high_cost", replace(base_cfg, cost_bps=25.0), None),
        ("mom_only", replace(base_cfg, score_momentum_weight=1.0, score_fundamental_weight=0.0), None),
        ("fund_only", replace(base_cfg, score_momentum_weight=0.0, score_fundamental_weight=1.0), None),
    ]

    for name, cfg, _ in definitions:
        panel = build_signal_panel(signal_panel["price"], load_weekly_factors(WORKBOOK_PATH), cfg)
        _, r = run_backtest(panel, cfg)
        perf = calc_perf(r)
        tests.append({"scenario": name, **perf})
    return pd.DataFrame(tests).sort_values("sharpe", ascending=False)


def ensure_output_dir(path: Path) -> None:
    """确保输出目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    主流程：
    数据读取 -> 信号构建 -> 回测 -> 绩效评估 -> 导出报表。
    """
    args = parse_args()
    if not WORKBOOK_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件: {WORKBOOK_PATH}")

    ensure_output_dir(OUTPUT_DIR)

    cfg = BacktestConfig(
        cost_bps=args.cost_bps,
        top_n=args.top_n,
        start_date=args.start_date,
        no_constraints=True,
    )
    cfg.max_weight_per_asset = 1.0
    cfg.max_weight_per_sector = 1.0
    cfg.max_turnover = 2.0
    weekly_prices = load_weekly_prices(WORKBOOK_PATH)
    factors = load_weekly_factors(WORKBOOK_PATH)
    signal_panel = build_signal_panel(weekly_prices, factors, cfg)

    weights, strategy_ret = run_backtest(signal_panel, cfg)
    nav = (1.0 + strategy_ret.fillna(0.0)).cumprod().rename("nav")

    perf_all = calc_perf(strategy_ret)
    split_idx = int(len(strategy_ret) * 0.7)
    ins = calc_perf(strategy_ret.iloc[:split_idx])
    oos = calc_perf(strategy_ret.iloc[split_idx:])
    perf_df = pd.DataFrame(
        [
            {"sample": "full", **perf_all},
            {"sample": "in_sample", **ins},
            {"sample": "out_of_sample", **oos},
        ]
    )

    annual = strategy_ret.groupby(strategy_ret.index.year).apply(lambda x: (1 + x).prod() - 1).rename("annual_return").to_frame()
    dd_df = top_drawdowns(nav, top_k=5)
    etf_df = build_etf_weights(weights)
    sens_df = run_param_sensitivity(signal_panel, cfg)
    robust_df = run_robust_tests(signal_panel, cfg)

    panel_flat = signal_panel.copy()
    panel_flat.columns = [f"{lvl0}_{lvl1}" for lvl0, lvl1 in panel_flat.columns]

    # 把多级列展平为单层，方便Excel/BI工具直接使用。
    panel_flat.to_csv(OUTPUT_DIR / "weekly_panel.csv", encoding="utf-8-sig")
    weights.to_csv(OUTPUT_DIR / "weights.csv", encoding="utf-8-sig")
    strategy_ret.to_frame("strategy_ret").to_csv(OUTPUT_DIR / "strategy_returns.csv", encoding="utf-8-sig")
    nav.to_frame().to_csv(OUTPUT_DIR / "nav_curve.csv", encoding="utf-8-sig")
    perf_df.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUTPUT_DIR / "annual_returns.csv", encoding="utf-8-sig")
    dd_df.to_csv(OUTPUT_DIR / "drawdown_periods.csv", index=False, encoding="utf-8-sig")
    etf_df.to_csv(OUTPUT_DIR / "etf_weights.csv", index=False, encoding="utf-8-sig")
    sens_df.to_csv(OUTPUT_DIR / "param_sensitivity.csv", index=False, encoding="utf-8-sig")
    robust_df.to_csv(OUTPUT_DIR / "robustness_tests.csv", index=False, encoding="utf-8-sig")

    print("模型运行完成，输出目录:", OUTPUT_DIR.resolve())
    print("核心绩效(全样本):")
    for k, v in perf_all.items():
        print(f"  - {k}: {v:.6f}" if isinstance(v, (int, float)) and not math.isnan(v) else f"  - {k}: {v}")
    print("已输出文件: weekly_panel.csv, weights.csv, strategy_returns.csv, nav_curve.csv,")
    print("performance_summary.csv, annual_returns.csv, drawdown_periods.csv, etf_weights.csv,")
    print("param_sensitivity.csv, robustness_tests.csv")


if __name__ == "__main__":
    main()
