# -*- coding: utf-8 -*-
"""
从免费数据源自动下载宏观/价格因子，写入 大宗商品轮动_数据.xlsx
数据来源：
  FRED   - 实际利率 / 美元指数
  EIA    - 美国商业原油库存（解析官网 HTML）
  FRED   - 美国工业产出指数（作为 PMI 替代，PMI 需从 Wind 手动补充）
  yfinance - 美元指数DXY / 黄金ETF价格 / 各品种期货价格
"""

import requests
import pandas as pd
import openpyxl
import yfinance as yf
from io import StringIO
from datetime import datetime
import time

MASTER_FILE = "大宗商品轮动_数据2.xlsx"
START = "1990-01-01"


# ══════════════════════════════════════════
# 1. FRED 下载
# ══════════════════════════════════════════
FRED_CONFIGS = [
    # (series_id,  子表名,        指标中文名)
    ("DFII10",   "实际利率",   "美国10年期TIPS收益率(实际利率,%)"),
    ("DTWEXBGS", "美元指数_FRED", "美元指数(美联储广义美元指数)"),
    ("INDPRO",   "工业产出指数", "美国工业生产指数(2017=100)"),   # PMI 替代指标
]


def download_fred(series_id: str, col_name: str) -> pd.DataFrame | None:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = ["日期", col_name]
        df["日期"] = pd.to_datetime(df["日期"])
        df = df[df["日期"] >= START].copy()
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        return df.dropna().sort_values("日期").reset_index(drop=True)
    except Exception as e:
        print(f"    ❌ FRED {series_id} 失败: {e}")
        return None


# ══════════════════════════════════════════
# 2. EIA 原油库存
# ══════════════════════════════════════════
def download_eia_crude_inventory() -> pd.DataFrame | None:
    """解析 EIA 官网 HTML，获取美国商业原油周度库存"""
    url = "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WCRSTUS1&f=W"
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        # 取形状最大的表格（宽格式，按 年月×周）
        raw = max(tables, key=lambda t: t.shape[0] * t.shape[1])
        # 列结构：Year-Month | Week1 EndDate | Week1 Value | Week2 EndDate | Week2 Value | ...
        raw.columns = [f"col{i}" for i in range(raw.shape[1])]
        raw = raw.iloc[2:].reset_index(drop=True)   # 跳过多级表头2行

        rows = []
        for _, r2 in raw.iterrows():
            year_month = str(r2["col0"]).strip()
            if not year_month or year_month in ("nan", "NaN"):
                continue
            # 每周数据占 2 列：EndDate + Value，从 col1 开始
            for j in range(1, raw.shape[1] - 1, 2):
                end_date_raw = str(r2[f"col{j}"]).strip()
                value_raw = r2[f"col{j+1}"]
                if end_date_raw in ("nan", "NaN", "") or pd.isna(value_raw):
                    continue
                try:
                    # end_date_raw 格式 MM/DD，year_month 格式 YYYY-MMM(英) 或 YYYY-Mon
                    year = year_month.split("-")[0]
                    month_day = end_date_raw  # "03/07"
                    d = datetime.strptime(f"{year}/{month_day}", "%Y/%m/%d").date()
                    rows.append((d, float(value_raw)))
                except Exception:
                    continue

        df = pd.DataFrame(rows, columns=["日期", "美国商业原油库存(千桶)"])
        df["日期"] = pd.to_datetime(df["日期"])
        df = df[df["日期"] >= START].drop_duplicates("日期").sort_values("日期").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"    ❌ EIA 原油库存失败: {e}")
        return None


# ══════════════════════════════════════════
# 3. yfinance 价格数据
# ══════════════════════════════════════════
YFINANCE_CONFIGS = [
    # (ticker,       子表名,             指标中文名,                           field)
    ("DX-Y.NYB",  "美元指数DXY",       "美元指数DXY(ICE)",                   "Close"),
    ("^VIX",      "VIX恐慌指数",        "CBOE波动率指数(VIX)",                "Close"),
    ("FXI",       "FXI中国大盘ETF",     "iShares中国大盘ETF收盘价(USD)",       "Close"),
    ("TTF=F",     "TTF欧洲天然气",      "TTF欧洲天然气期货收盘价(EUR/MWh)",    "Close"),
    ("GLD",       "黄金ETF价格",        "SPDR黄金ETF收盘价(USD)",              "Close"),
    ("GC=F",      "黄金期货_YF",        "COMEX黄金期货收盘价(USD/oz)",         "Close"),
    ("SI=F",      "银期货_YF",          "COMEX银期货收盘价(USD/oz)",           "Close"),
    ("HG=F",      "铜期货_YF",          "COMEX铜期货收盘价(USD/lb)",           "Close"),
    ("CL=F",      "WTI原油期货_YF",     "WTI原油期货收盘价(USD/bbl)",          "Close"),
    ("BZ=F",      "布伦特原油期货_YF",  "布伦特原油期货收盘价(USD/bbl)",       "Close"),
]


def download_yfinance(ticker: str, col_name: str, field: str = "Close") -> pd.DataFrame | None:
    try:
        raw = yf.download(ticker, start=START, auto_adjust=True, progress=False)
        if raw.empty:
            print(f"    ❌ {ticker} 无数据")
            return None
        df = raw[[field]].copy()
        # yfinance 1.x MultiIndex columns 兼容
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col_name]
        else:
            df.columns = [col_name]
        df.index.name = "日期"
        df = df.reset_index()
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.dropna().sort_values("日期").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"    ❌ yfinance {ticker} 失败: {e}")
        return None


# ══════════════════════════════════════════
# 4. 写入 Excel
# ══════════════════════════════════════════
def write_sheet(wb: openpyxl.Workbook, sheet_name: str, df: pd.DataFrame):
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)
    headers = list(df.columns)
    for c, h in enumerate(headers, 1):
        ws.cell(1, c, h)
    for r, row in enumerate(df.itertuples(index=False), start=2):
        for c, val in enumerate(row, 1):
            if hasattr(val, "date"):
                ws.cell(r, c, val.date())
            elif pd.isna(val):
                ws.cell(r, c, None)
            else:
                ws.cell(r, c, float(val) if isinstance(val, (int, float)) else val)


# ══════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════
def main():
    if not __import__("os").path.exists(MASTER_FILE):
        print(f"❌ 找不到主文件 {MASTER_FILE}，请先运行 generate_new_excel.py")
        return

    wb = openpyxl.load_workbook(MASTER_FILE)
    results, failed = [], []

    # ── FRED ──
    print("\n【FRED 数据】")
    for series_id, sheet_name, col_name in FRED_CONFIGS:
        print(f"  ↓ {sheet_name} ({series_id})...", end=" ", flush=True)
        df = download_fred(series_id, col_name)
        if df is not None and not df.empty:
            write_sheet(wb, sheet_name, df)
            print(f"✅ {len(df)} 行")
            results.append(sheet_name)
        else:
            failed.append(sheet_name)
        time.sleep(0.8)

    # ── EIA 原油库存 ──
    print("\n【EIA 原油库存】")
    print("  ↓ 原油库存...", end=" ", flush=True)
    df_oil = download_eia_crude_inventory()
    if df_oil is not None and not df_oil.empty:
        write_sheet(wb, "原油库存", df_oil)
        print(f"✅ {len(df_oil)} 行（周度数据）")
        results.append("原油库存")
    else:
        failed.append("原油库存")

    # ── yfinance ──
    print("\n【yfinance 价格数据】")
    for ticker, sheet_name, col_name, field in YFINANCE_CONFIGS:
        print(f"  ↓ {sheet_name} ({ticker})...", end=" ", flush=True)
        df = download_yfinance(ticker, col_name, field)
        if df is not None and not df.empty:
            write_sheet(wb, sheet_name, df)
            print(f"✅ {len(df)} 行")
            results.append(sheet_name)
        else:
            failed.append(sheet_name)
        time.sleep(0.3)

    wb.save(MASTER_FILE)

    print("\n" + "═" * 55)
    print(f"✅ 成功写入 {len(results)} 个子表：{results}")
    if failed:
        print(f"❌ 未能下载：{failed}")
    print("""
【说明】
• 实际利率       → FRED DFII10（10年期TIPS收益率）
• 美元指数       → FRED广义美元指数 + ICE DXY（两个版本）
• 原油库存       → EIA官网，周度，美国商业原油库存（千桶）
• 工业产出指数   → FRED INDPRO，作为 PMI 替代指标
• 各品种价格     → Yahoo Finance（期货近月合约）
• GLD价格        → SPDR黄金ETF收盘价（可作为黄金 ETF 持仓的代理）

【还需要从 Wind 手动补充的数据】
• 美国ISM制造业PMI  → Wind 搜索"美国ISM制造业PMI"或代码 M0000545
• LME铜库存        → Wind 搜索"LME铜库存"
• COMEX银库存      → Wind 搜索"COMEX银库存"
• 黄金ETF实物持仓（吨数）→ Wind 搜索"SPDR黄金持仓"
""")


if __name__ == "__main__":
    main()
