# -*- coding: utf-8 -*-
"""
机器学习版大宗商品轮动模型（周频）。

特点：
- 用历史滚动训练模型预测“下一周收益”
- 支持无约束权重（按预测分数softmax映射）
- 输出结构与规则版相近，便于复用报告脚本

示例：
    python3 ml_rotation_model.py --model elasticnet --cost-bps 0 --no-constraints
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rotation_model import (
    ASSETS,
    WORKBOOK_PATH,
    BacktestConfig,
    _cap_turnover,
    build_etf_weights,
    build_signal_panel,
    calc_perf,
    ensure_output_dir,
    load_weekly_factors,
    load_weekly_prices,
    top_drawdowns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="机器学习版大宗商品轮动模型（Ridge / ElasticNet / LightGBM）")
    parser.add_argument("--model", choices=["ridge", "elasticnet", "lgbm"], default="elasticnet", help="机器学习模型类型")
    parser.add_argument("--enet-alpha", type=float, default=0.001, help="ElasticNet 正则强度 alpha")
    parser.add_argument("--enet-l1-ratio", type=float, default=0.5, help="ElasticNet 的 L1 比例（0~1）")
    parser.add_argument("--cost-bps", type=float, default=0.0, help="单边交易成本（bps）")
    parser.add_argument("--start-date", type=str, default="2008-01-01", help="回测起始日期")
    parser.add_argument("--top-n", type=int, default=2, help="有约束模式下每期持仓品种数")
    parser.add_argument("--min-train-weeks", type=int, default=156, help="最少训练周数（默认3年）")
    parser.add_argument("--output-dir", type=str, default="model_outputs_ml", help="输出目录")
    parser.add_argument("--no-constraints", action="store_true", help="关闭持仓数量/比例/换手约束")
    return parser.parse_args()


def _build_model(model_name: str, enet_alpha: float, enet_l1_ratio: float):
    """构建回归模型。"""
    if model_name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model_name == "lgbm":
        return lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=8,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
    return make_pipeline(
        StandardScaler(),
        ElasticNet(alpha=enet_alpha, l1_ratio=enet_l1_ratio, random_state=42, max_iter=5000),
    )


def _softmax_weights(
    score_row: pd.Series,
) -> pd.Series:
    valid = score_row.dropna()
    out = pd.Series(0.0, index=ASSETS)
    if valid.empty:
        return out
    x = (valid - valid.max()).astype(float)
    ex = np.exp(x)
    w = ex / ex.sum()
    for a, v in w.items():
        out[a] = float(v)
    return out


def build_ml_dataset(signal_panel: pd.DataFrame, factors: Dict[str, pd.Series]) -> pd.DataFrame:
    """把多资产面板展开为 long 格式监督学习样本，使用原始因子变化率作为特征。"""
    weekly_ret = signal_panel["weekly_ret"].copy()
    mom_raw = signal_panel["mom_raw"].copy()
    common_idx = weekly_ret.index

    def factor_chg(key: str, periods: int = 4) -> pd.Series:
        s = factors.get(key, pd.Series(dtype=float))
        if s.empty:
            return pd.Series(0.0, index=common_idx)
        s2 = s.reindex(common_idx).ffill()
        return s2.pct_change(periods).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    real_rate_chg = factor_chg("real_rate", 4)
    dxy_chg       = factor_chg("dxy", 4)
    pmi_chg       = factor_chg("pmi", 4)
    gold_oi_chg   = factor_chg("gold_oi", 4)
    silver_oi_chg = factor_chg("silver_oi", 4)
    copper_inv_chg = factor_chg("copper_inventory", 4)
    oil_inv_chg   = factor_chg("oil_inventory", 4)

    frames: List[pd.DataFrame] = []
    for asset in ASSETS:
        sret = weekly_ret[asset]
        df = pd.DataFrame(index=common_idx)
        df["asset"] = asset
        df["mom_raw"] = mom_raw[asset]
        df["real_rate_chg"] = real_rate_chg
        df["dxy_chg"]       = dxy_chg
        df["pmi_chg"]       = pmi_chg
        df["gold_oi_chg"]   = gold_oi_chg
        df["silver_oi_chg"] = silver_oi_chg
        df["copper_inv_chg"] = copper_inv_chg
        df["oil_inv_chg"]   = oil_inv_chg
        df["ret_1w"]  = sret
        df["ret_4w"]  = sret.rolling(4).sum()
        df["ret_12w"] = sret.rolling(12).sum()
        df["vol_12w"] = sret.rolling(12).std(ddof=0)
        df["target_next_ret"] = sret.shift(-1) - weekly_ret.shift(-1).mean(axis=1)
        frames.append(df.reset_index().rename(columns={"index": "date"}))

    long_df = pd.concat(frames, ignore_index=True)
    long_df = pd.get_dummies(long_df, columns=["asset"], prefix="asset", dtype=float)
    long_df = long_df.sort_values("date").reset_index(drop=True)
    return long_df


def rolling_predict_scores(
    long_df: pd.DataFrame,
    model_name: str,
    enet_alpha: float,
    enet_l1_ratio: float,
    min_train_weeks: int,
    start_date: str,
) -> pd.DataFrame:
    """滚动训练并生成每周各资产预测分数（下一周收益预测值）。"""
    feature_cols = [
        "mom_raw",
        "real_rate_chg",
        "dxy_chg",
        "pmi_chg",
        "gold_oi_chg",
        "silver_oi_chg",
        "copper_inv_chg",
        "oil_inv_chg",
        "ret_1w",
        "ret_4w",
        "ret_12w",
        "vol_12w",
        "asset_原油",
        "asset_白银",
        "asset_黄金",
        "asset_铜",
    ]
    target_col = "target_next_ret"

    dates = sorted(long_df["date"].dropna().unique())
    pred_rows: List[Dict[str, object]] = []
    start_ts = pd.Timestamp(start_date)

    for dt in dates:
        dt = pd.Timestamp(dt)
        if dt < start_ts:
            continue
        train = long_df[long_df["date"] < dt].copy()
        train = train.dropna(subset=feature_cols + [target_col])
        if len(train) < min_train_weeks * len(ASSETS):
            continue

        model = _build_model(model_name=model_name, enet_alpha=enet_alpha, enet_l1_ratio=enet_l1_ratio)
        model.fit(train[feature_cols], train[target_col])

        test = long_df[long_df["date"] == dt].copy()
        if test.empty:
            continue
        test_feat = test[feature_cols].fillna(0.0)
        pred = model.predict(test_feat)

        # 从 one-hot 还原资产名
        for j, (_, row) in enumerate(test.iterrows()):
            asset = None
            for a in ASSETS:
                col = f"asset_{a}"
                if col in row and float(row[col]) > 0.5:
                    asset = a
                    break
            if asset is None:
                continue
            pred_rows.append({"date": dt, "asset": asset, "pred_score": float(pred[j])})

    pred_df = pd.DataFrame(pred_rows)
    if pred_df.empty:
        return pd.DataFrame(columns=ASSETS)
    score_wide = pred_df.pivot(index="date", columns="asset", values="pred_score").reindex(columns=ASSETS).sort_index()
    return score_wide


def run_ml_backtest(
    pred_scores: pd.DataFrame,
    weekly_ret: pd.DataFrame,
    cfg: BacktestConfig,
    no_constraints: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    """把模型预测分数转成权重，并计算策略收益。"""
    dates = pred_scores.index
    weight_rows = []
    strat_rows = []
    prev_w = pd.Series(0.0, index=ASSETS)

    for i, dt in enumerate(dates):
        sc = pred_scores.loc[dt]
        if no_constraints:
            target = _softmax_weights(sc)
            chosen = [a for a in ASSETS if target[a] > 1e-10]
        else:
            ranked = sc.sort_values(ascending=False).dropna().index.tolist()[: cfg.top_n]
            target = pd.Series(0.0, index=ASSETS)
            if ranked:
                w = 1.0 / len(ranked)
                for a in ranked:
                    target[a] = w
            target = _cap_turnover(target, prev_w, cfg.max_turnover)
            chosen = ranked

        turnover = float((target - prev_w).abs().sum())
        trade_cost = turnover * cfg.cost_bps / 10000.0
        next_ret = weekly_ret.iloc[i + 1].fillna(0.0) if i < len(dates) - 1 else pd.Series(0.0, index=ASSETS)
        strat_ret = float((target * next_ret).sum() - trade_cost)

        weight_rows.append(
            {
                "date": dt,
                **{f"w_{a}": float(target[a]) for a in ASSETS},
                "turnover": turnover,
                "cost": trade_cost,
                "chosen": ",".join(chosen),
            }
        )
        strat_rows.append((dt, strat_ret))
        prev_w = target

    weights = pd.DataFrame(weight_rows).set_index("date")
    strategy_ret = pd.Series(dict(strat_rows), name="strategy_ret").sort_index()
    return weights, strategy_ret


def main() -> None:
    args = parse_args()
    if not WORKBOOK_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件: {WORKBOOK_PATH}")

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    cfg = BacktestConfig(cost_bps=args.cost_bps, top_n=args.top_n, start_date=args.start_date)
    weekly_prices = load_weekly_prices(WORKBOOK_PATH)
    factors = load_weekly_factors(WORKBOOK_PATH)
    signal_panel = build_signal_panel(weekly_prices, factors, cfg)
    weekly_ret = signal_panel["weekly_ret"].reindex(columns=ASSETS)

    long_df = build_ml_dataset(signal_panel, factors)
    pred_scores = rolling_predict_scores(
        long_df=long_df,
        model_name=args.model,
        enet_alpha=args.enet_alpha,
        enet_l1_ratio=args.enet_l1_ratio,
        min_train_weeks=args.min_train_weeks,
        start_date=args.start_date,
    )
    if pred_scores.empty:
        raise RuntimeError("机器学习预测为空，请降低 --min-train-weeks 或检查输入数据。")

    weekly_ret_for_bt = weekly_ret.reindex(pred_scores.index)
    weights, strategy_ret = run_ml_backtest(
        pred_scores=pred_scores,
        weekly_ret=weekly_ret_for_bt,
        cfg=cfg,
        no_constraints=args.no_constraints,
    )
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

    pred_scores.to_csv(output_dir / "ml_scores.csv", encoding="utf-8-sig")
    weights.to_csv(output_dir / "weights.csv", encoding="utf-8-sig")
    strategy_ret.to_frame("strategy_ret").to_csv(output_dir / "strategy_returns.csv", encoding="utf-8-sig")
    nav.to_frame().to_csv(output_dir / "nav_curve.csv", encoding="utf-8-sig")
    perf_df.to_csv(output_dir / "performance_summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(output_dir / "annual_returns.csv", encoding="utf-8-sig")
    dd_df.to_csv(output_dir / "drawdown_periods.csv", index=False, encoding="utf-8-sig")
    etf_df.to_csv(output_dir / "etf_weights.csv", index=False, encoding="utf-8-sig")

    print("机器学习模型运行完成，输出目录:", output_dir.resolve())
    print(
        f"模型类型: {args.model}, 无约束模式: {args.no_constraints}, "
        f"交易成本bps: {args.cost_bps}"
    )
    print("核心绩效(全样本):")
    for k, v in perf_all.items():
        print(f"  - {k}: {v:.6f}" if isinstance(v, (int, float)) and not math.isnan(v) else f"  - {k}: {v}")


if __name__ == "__main__":
    main()
