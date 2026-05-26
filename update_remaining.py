# -*- coding: utf-8 -*-
"""
更新所有剩余可自动更新的 sheet：
  美元指数           → 从已更新的美元指数_FRED sheet 复制缺失行
  黄金/银/铜/原油期货价格_YF, 黄金ETF持仓_价格 → yfinance 追加
  工业产出指数        → FRED INDPRO
  日/月/近三月/近六月收益率 → 从期货价格重新计算
"""

import subprocess, io, requests, openpyxl, yfinance as yf, pandas as pd
from io import StringIO
from collections import defaultdict
from datetime import datetime

MASTER_FILE = "大宗商品轮动_数据2.xlsx"


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def last_date_ascending(ws) -> pd.Timestamp | None:
    """升序 sheet：找第一列最后一个非 None 值"""
    last = None
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r and r[0] is not None:
            try: last = pd.Timestamp(r[0])
            except: pass
    return last


def first_date_descending(ws) -> pd.Timestamp | None:
    """降序 sheet（最新在第2行）：找第一行数据的日期"""
    for r in ws.iter_rows(min_row=2, max_row=3, values_only=True):
        if r and r[0] is not None:
            try: return pd.Timestamp(r[0])
            except: pass
    return None


def append_ascending(ws, new_rows: list[tuple]):
    """向升序 sheet 末尾追加 (date, value) 列表"""
    last_row = ws.max_row
    for d, v in new_rows:
        last_row += 1
        ws.cell(row=last_row, column=1, value=d)
        ws.cell(row=last_row, column=2, value=v)
    return len(new_rows)


def prepend_descending(ws, new_rows: list[tuple]):
    """向降序 sheet 最前（第2行）插入新行，旧数据下移"""
    if not new_rows:
        return 0
    # 读取现有数据
    existing = [(r[0], r[1]) for r in ws.iter_rows(min_row=2, values_only=True)
                if r and any(v is not None for v in r)]
    # 清空
    for row in ws.iter_rows(min_row=2):
        for cell in row: cell.value = None
    # 写新 + 旧
    for i, (d, v) in enumerate(new_rows + existing, start=2):
        ws.cell(row=i, column=1, value=d)
        ws.cell(row=i, column=2, value=v)
    return len(new_rows)


def yf_append(ws, ticker: str, after: pd.Timestamp):
    """下载 yfinance Close 数据，追加 after 之后的行"""
    df = yf.download(ticker, start=str(after.date()), auto_adjust=True, progress=False)
    if df.empty:
        return 0
    close = df["Close"]
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).normalize()
    today = pd.Timestamp.today().date()
    new = [(d.date(), float(v)) for d, v in close.items()
           if after.date() < pd.Timestamp(d).date() < today]
    return append_ascending(ws, new), new[-1][0] if new else None


def download_fred_curl(series_id: str) -> pd.DataFrame | None:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = subprocess.run(["curl", "-s", "--max-time", "60", "--retry", "2", url],
                           capture_output=True, text=True, timeout=90)
        text = r.stdout if r.returncode == 0 and r.stdout.startswith("observation") else None
        if not text:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            text = resp.text
        df = pd.read_csv(StringIO(text))
        df.columns = ["日期", "value"]
        df["日期"] = pd.to_datetime(df["日期"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna().sort_values("日期")
    except Exception as e:
        print(f"    FRED {series_id} 失败: {e}")
        return None


# ─────────────────────────────────────────────
# 收益率重计算（来自 fill_returns.py 逻辑）
# ─────────────────────────────────────────────
PRICE_COLS = [(0, 1), (2, 3), (4, 5), (6, 7)]
ASSET_NAMES = ["黄金", "银", "铜", "布伦特原油"]


def build_series(rows, dc, pc):
    data = []
    for row in rows[2:]:
        if row is None: continue
        d = row[dc] if dc < len(row) else None
        p = row[pc] if pc < len(row) else None
        if d is None or p is None: continue
        if not isinstance(p, (int, float)): continue
        dt = d.date() if hasattr(d, "date") else d
        if dt is None: continue
        data.append((dt, float(p)))
    return sorted(data)


def daily_rets(s):
    return [(s[i][0], (s[i][1]-s[i-1][1])/s[i-1][1])
            for i in range(1, len(s)) if s[i-1][1]]


def month_ends(s):
    by_m = defaultdict(list)
    for d, p in s: by_m[(d.year, d.month)].append((d, p))
    return [max(v) for v in sorted(by_m.values())]


def monthly_rets(s):
    ends = month_ends(s)
    return [(ends[i][0], (ends[i][1]-ends[i-1][1])/ends[i-1][1])
            for i in range(1, len(ends)) if ends[i-1][1]]


def quarterly_rets(s):
    by_q = defaultdict(list)
    for d, p in s: by_q[(d.year, (d.month-1)//3+1)].append((d, p))
    out = []
    for vals in sorted(by_q.values()):
        vals = sorted(vals)
        if vals[0][1]: out.append((vals[-1][0], (vals[-1][1]-vals[0][1])/vals[0][1]))
    return out


def semi_rets(s):
    by_h = defaultdict(list)
    for d, p in s: by_h[(d.year, 1 if d.month<=6 else 2)].append((d, p))
    out = []
    for vals in sorted(by_h.values()):
        vals = sorted(vals)
        if vals[0][1]: out.append((vals[-1][0], (vals[-1][1]-vals[0][1])/vals[0][1]))
    return out


def align(series_list):
    all_dates = sorted(set(d for s in series_list for d, _ in s))
    maps = [dict(s) for s in series_list]
    return [[d] + [m.get(d) for m in maps] for d in all_dates]


def write_sheet(wb, name, headers, data):
    if name in wb.sheetnames: del wb[name]
    ws = wb.create_sheet(name)
    for c, h in enumerate(headers, 1): ws.cell(1, c, h)
    for r, row in enumerate(data, 2):
        for c, val in enumerate(row, 1): ws.cell(r, c, val)


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    wb = openpyxl.load_workbook(MASTER_FILE)

    # ── 1. 美元指数（从 美元指数_FRED 复制缺失行，均为 DTWEXBGS 广义美元指数）──
    print("\n【美元指数】")
    ws_dxy = wb["美元指数"]
    ws_src = wb["美元指数_FRED"]
    latest = first_date_descending(ws_dxy)
    print(f"  当前最新: {latest.date() if latest else '无'}")
    # 读取 美元指数_FRED 中更新的数据
    src_rows = [(pd.Timestamp(r[0]), r[1])
                for r in ws_src.iter_rows(min_row=2, values_only=True)
                if r and r[0] is not None and r[1] is not None]
    new = [(d.date(), float(v)) for d, v in src_rows if d > latest] if latest else []
    new.sort(key=lambda x: x[0], reverse=True)  # 降序
    if new:
        n = prepend_descending(ws_dxy, new)
        print(f"  ✅ 追加 {n} 行，最新: {new[0][0]}")
    else:
        print("  ✅ 已是最新")

    # ── 2. 旧版 yfinance 价格 sheets（升序，追加末尾）──
    yf_sheets = [
        ("黄金期货价格_YF",   "GC=F"),
        ("黄金ETF持仓_价格",   "GLD"),
        ("铜期货价格_YF",     "HG=F"),
        ("银期货价格_YF",     "SI=F"),
        ("原油期货价格_YF",   "CL=F"),
    ]
    print("\n【yfinance 价格 sheets】")
    for sheet_name, ticker in yf_sheets:
        ws = wb[sheet_name]
        after = last_date_ascending(ws)
        print(f"  {sheet_name} 当前最新: {after.date() if after else '无'}  ticker={ticker}")
        if after is None:
            print(f"    ⚠️ 跳过")
            continue
        try:
            result = yf_append(ws, ticker, after)
            n, last_d = result if isinstance(result, tuple) else (result, None)
            if n > 0:
                print(f"    ✅ 追加 {n} 行，最新: {last_d}")
            else:
                print(f"    ✅ 已是最新")
        except Exception as e:
            print(f"    ❌ {e}")

    # ── 3. 工业产出指数（FRED INDPRO，升序，月度）──
    print("\n【工业产出指数】")
    ws_ind = wb["工业产出指数"]
    after = last_date_ascending(ws_ind)
    print(f"  当前最新: {after.date() if after else '无'}")
    df_ind = download_fred_curl("INDPRO")
    if df_ind is not None:
        new = [(row["日期"].date(), float(row["value"]))
               for _, row in df_ind.iterrows() if row["日期"] > after]
        if new:
            append_ascending(ws_ind, new)
            print(f"  ✅ 追加 {len(new)} 行，最新: {new[-1][0]}")
        else:
            print("  ✅ 已是最新")
    else:
        print("  ⚠️ FRED 暂时无法访问，跳过")

    # ── 4. 日/月/近三月/近六月收益率（从期货价格重新计算）──
    print("\n【收益率 sheets 重计算】")
    ws_px = wb["期货价格"]
    rows = list(ws_px.iter_rows(values_only=True))
    series_list = [build_series(rows, dc, pc) for dc, pc in PRICE_COLS]
    headers = ["日期"] + [f"{n}收益率" for n in ASSET_NAMES]
    write_sheet(wb, "日收益率",    headers, align([daily_rets(s)    for s in series_list]))
    write_sheet(wb, "月收益率",    headers, align([monthly_rets(s)  for s in series_list]))
    write_sheet(wb, "近三月收益率", headers, align([quarterly_rets(s) for s in series_list]))
    write_sheet(wb, "近六月收益率", headers, align([semi_rets(s)     for s in series_list]))
    print("  ✅ 日/月/近三月/近六月收益率 已重算，基于期货价格最新至 2026-04-26")

    wb.save(MASTER_FILE)
    print("\n✅ 全部保存完成。")


if __name__ == "__main__":
    main()
