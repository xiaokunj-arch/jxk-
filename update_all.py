# -*- coding: utf-8 -*-
"""
一键更新所有可自动化的数据源，按顺序执行：
  1. fetch_factors         — 全量刷新（FRED/EIA/yfinance/akshare）：
                             实际利率、美元指数DXY、VIX、FXI、TTF、美债收益率、
                             原油库存、中国信贷脉冲、中国PMI/PPI、各品种期货价格
  2. update_macro_factors  — 增量追加（FRED + akshare）：
                             PMI、PPI、CPI、LME铜库存、COMEX黄金持仓、中国PMI/PPI
  3. update_remaining      — 增量追加（yfinance + FRED）：
                             美元指数、yfinance价格sheets、工业产出指数、收益率重算
  4. fetch_cot             — CFTC COT 管理资金净头寸（黄金/白银/铜/原油）
  5. fill_futures_yf       — yfinance 补充期货价格缺口

⚠️ 手动更新（不在自动流程中）：
   - 中国制造业PMI / 中国PPI（手动维护）
   - comex白银持仓量（无免费接口）
   - 煤炭库存（需 Wind）
   请参考 Wind取数指引.md 手动操作后再运行 merge_factors.py。

用法：
    python3 update_all.py
"""
from __future__ import annotations
import os
import sys
import traceback
from pathlib import Path

os.chdir(Path(__file__).parent)


def run_step(name: str, module_name: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  [{name}]")
    print(f"{'='*60}")
    try:
        mod = __import__(module_name)
        mod.main()
        print(f"\n  ✅ {name} 完成")
        return True
    except Exception:
        print(f"\n  ❌ {name} 失败：")
        traceback.print_exc()
        return False


def main():
    steps = [
        ("全量刷新（FRED/EIA/yfinance/akshare）", "fetch_factors"),
        ("宏观因子增量追加（FRED + akshare）",      "update_macro_factors"),
        ("价格与收益率增量追加（yfinance + FRED）",  "update_remaining"),
        ("COT 持仓报告（CFTC）",                   "fetch_cot"),
        ("期货价格补齐（yfinance）",                "fill_futures_yf"),
        ("铜展期收益率（SHFE）",                   "fetch_copper_roll"),
    ]

    results = []
    for name, module in steps:
        ok = run_step(name, module)
        results.append((name, ok))

    print(f"\n{'='*60}")
    print("  汇总")
    print(f"{'='*60}")
    all_ok = True
    for name, ok in results:
        status = "✅" if ok else "❌"
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  全部更新完成！")
    else:
        print("\n  部分步骤失败，请检查上方错误信息。")
        print("  失败的步骤不影响其他步骤，已成功的数据已保存。")
    print(f"{'='*60}")
    print()
    print("  ⚠️  以下数据需手动更新：")
    print("      - 中国制造业PMI / 中国PPI（手动维护）")
    print("      - comex白银持仓量（无免费接口）")
    print("      - 煤炭库存（需 Wind）")


if __name__ == "__main__":
    main()
