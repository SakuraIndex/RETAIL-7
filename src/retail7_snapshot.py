# coding: utf-8
"""
Retail-7 (Japan Retail) equal-dollar-weighted index
- Intraday 1m -> 5min (JST) で当日 "開場からの騰落率(%)" を算出し平均
- 出力:
  - docs/outputs/retail_7_intraday.csv
  - docs/outputs/retail_7_stats.json
  - docs/outputs/retail_7_levels.csv
  - docs/outputs/retail_7_post_intraday.txt
"""

import os, json, math, sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except Exception as e:
    print("yfinance import error:", e)
    sys.exit(1)

OUT_DIR = "docs/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 構成銘柄（yfinance ティッカー）
# ─────────────────────────────────────────────────────────────
TICKERS = {
    "AEON": "8267.T",            # イオン
    "FAST_RETAIL": "9983.T",     # ファーストリテイリング
    "PPIH": "7532.T",            # PPIH（パンパシフィック）
    "SEVEN_I": "3382.T",         # セブン&アイ
    "IMH": "3099.T",             # 三越伊勢丹HD
    "TRIAL": "5885.T",           # トライアルHD
    "TAKASHIMAYA": "8233.T",     # 高島屋
}

INDEX_KEY = "RETAIL_7"
BASE_DATE = "2024-01-04"
JST = timezone(timedelta(hours=9))

def _now_jst():
    return datetime.now(JST)

def _to_jst_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert("Asia/Tokyo")

def _fetch_intraday_1m(ticker: str) -> pd.DataFrame | None:
    """当日分の1分足（JST）だけ返す。"""
    try:
        df = yf.download(ticker, interval="1m", period="7d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        # インデックスをJSTへ
        df.index = _to_jst_index(df.index)
        today = _now_jst().date()
        df = df[df.index.date == today]
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"[warn] 1m fetch failed: {ticker} - {e}")
        return None

def build_intraday_series() -> pd.DataFrame:
    """
    1) 各ティッカーの当日1分足を取得
    2) 最初の有効な価格を寄り値として % vs Open を作成
    3) 5分足へリサンプルし、等金額加重（単純平均）
    """
    series_list = []
    used = []

    for name, tk in TICKERS.items():
        df = _fetch_intraday_1m(tk)
        if df is None or df.empty:
            continue

        # Close列を安全に数値化
        close = pd.to_numeric(df.get("Close"), errors="coerce")
        if close is None or close.empty:
            continue

        first_idx = close.first_valid_index()
        if first_idx is None:
            continue

        # ★ ここが修正ポイント：必ず float スカラー化
        try:
            open_price = float(close.loc[first_idx])
        except Exception:
            continue
        if not np.isfinite(open_price) or open_price <= 0.0:
            continue

        pct = (close / open_price - 1.0) * 100.0
        s = pct.to_frame(name=name)
        series_list.append(s)
        used.append(name)

    if not series_list:
        raise RuntimeError("no prices for today (JST)")

    mat = pd.concat(series_list, axis=1, join="outer").sort_index()
    # 5分足平均（等金額加重）
    mat_5 = mat.resample("5min").mean()
    mat_5 = mat_5.ffill()  # 欠損の前方補完

    out = mat_5[used].mean(axis=1, skipna=True).to_frame("retail7_pct")
    out.index.name = "datetime_jst"
    return out

def build_levels_daily() -> pd.DataFrame:
    """等金額の日足レベル（100起点）。"""
    frames = []
    for name, tk in TICKERS.items():
        try:
            d = yf.download(tk, interval="1d", period="2y", progress=False, auto_adjust=False)
            if d is None or d.empty:
                continue
            d.index = _to_jst_index(d.index)
            s = pd.to_numeric(d["Close"], errors="coerce").rename(name)
            frames.append(s)
        except Exception as e:
            print(f"[warn] daily fetch failed: {tk} - {e}")
            continue

    if not frames:
        raise RuntimeError("daily levels empty")

    mat = pd.concat(frames, axis=1, join="inner").sort_index()
    lvl = mat.mean(axis=1, skipna=True)
    base = pd.to_datetime(BASE_DATE).tz_localize("Asia/Tokyo")
    if base not in lvl.index:
        base = lvl.index[lvl.index.get_indexer([base], method="nearest")][0]
    base_val = float(lvl.loc[base])
    level = lvl / base_val * 100.0
    df_out = level.to_frame("level")
    df_out.index.name = "date_jst"
    return df_out

def main():
    intraday = build_intraday_series()
    os.makedirs(OUT_DIR, exist_ok=True)
    intraday.to_csv(os.path.join(OUT_DIR, "retail_7_intraday.csv"), encoding="utf-8")

    last_pct = float(intraday["retail7_pct"].iloc[-1])
    stats = {
        "key": INDEX_KEY,
        "pct_intraday": round(last_pct, 2),
        "updated_at": _now_jst().strftime("%Y/%m/%d %H:%M"),
        "unit": "pct",
        "last_level": None,
        "tickers": list(TICKERS.values()),
    }
    with open(os.path.join(OUT_DIR, "retail_7_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    try:
        lv = build_levels_daily()
        lv.to_csv(os.path.join(OUT_DIR, "retail_7_levels.csv"), encoding="utf-8")
        if not math.isnan(lv["level"].iloc[-1]):
            stats["last_level"] = round(float(lv["level"].iloc[-1]), 2)
            with open(os.path.join(OUT_DIR, "retail_7_stats.json"), "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[warn] levels build failed:", e)

    post = [
        "【RETAIL-7｜小売業指数】",
        f"本日：{('+' if last_pct>=0 else '')}{last_pct:.2f}%",
        f"構成：{','.join(TICKERS.values())}",
        "#桜Index #Retail7",
    ]
    with open(os.path.join(OUT_DIR, "retail_7_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(post))

    with open(os.path.join(OUT_DIR, "last_run.txt"), "w", encoding="utf-8") as f:
        f.write(_now_jst().strftime("%Y/%m/%d %H:%M (JST)"))

if __name__ == "__main__":
    main()
