# coding: utf-8
"""
Retail-7 (Japan Retail) equal-dollar-weighted index
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

TICKERS = {
    "AEON": "8267.T",
    "FAST_RETAIL": "9983.T",
    "PPIH": "7532.T",
    "SEVEN_I": "3382.T",
    "IMH": "3099.T",
    "TRIAL": "5885.T",
    "TAKASHIMAYA": "8233.T",
}

INDEX_KEY = "RETAIL_7"
BASE_DATE = "2024-01-04"
JST = timezone(timedelta(hours=9))
def _now_jst(): return datetime.now(JST)

def _to_jst_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None: idx = idx.tz_localize("UTC")
    return idx.tz_convert("Asia/Tokyo")

def _pick_close_series(df: pd.DataFrame) -> pd.Series | None:
    """
    df から終値 Series を安全に抜き出す:
    - 'Close' or 'Adj Close' を優先
    - 大文字小文字/スペース差異を吸収
    - MultiIndex 列にフォールバック
    """
    # まず通常列
    for cand in ["Close", "Adj Close", "close", "adj close", "AdjClose"]:
        if cand in df.columns:
            return pd.to_numeric(df[cand], errors="coerce")

    # MultiIndex 列対応（最後のレベルに 'Close' などがいるケース）
    if isinstance(df.columns, pd.MultiIndex):
        for cand in ["Close", "Adj Close"]:
            try:
                sub = df.xs(cand, axis=1, level=-1, drop_level=False)
                if not sub.empty:
                    s = sub.iloc[:, 0]
                    return pd.to_numeric(s, errors="coerce")
            except Exception:
                pass
    return None

def _fetch_intraday_1m(ticker: str) -> pd.DataFrame | None:
    try:
        # group_by='column' で列をフラット化
        df = yf.download(
            ticker, interval="1m", period="7d", progress=False,
            auto_adjust=False, group_by="column"
        )
        if df is None or df.empty: return None
        df.index = _to_jst_index(df.index)
        today = _now_jst().date()
        df = df[df.index.date == today]
        return None if df.empty else df
    except Exception as e:
        print(f"[warn] 1m fetch failed: {ticker} - {e}")
        return None

def build_intraday_series() -> pd.DataFrame:
    series_list, used = [], []
    for name, tk in TICKERS.items():
        df = _fetch_intraday_1m(tk)
        if df is None or df.empty: continue

        close = _pick_close_series(df)
        if close is None or close.empty: continue

        first_idx = close.first_valid_index()
        if first_idx is None: continue

        try:
            open_price = float(close.loc[first_idx])
        except Exception:
            continue
        if not np.isfinite(open_price) or open_price <= 0.0: continue

        pct = (close / open_price - 1.0) * 100.0
        series_list.append(pct.to_frame(name))
        used.append(name)

    if not series_list:
        raise RuntimeError("no prices for today (JST)")

    mat = pd.concat(series_list, axis=1, join="outer").sort_index()
    mat_5 = mat.resample("5min").mean().ffill()
    out = mat_5[used].mean(axis=1, skipna=True).to_frame("retail7_pct")
    out.index.name = "datetime_jst"
    return out

def build_levels_daily() -> pd.DataFrame:
    frames = []
    for name, tk in TICKERS.items():
        try:
            d = yf.download(
                tk, interval="1d", period="2y", progress=False,
                auto_adjust=False, group_by="column"
            )
            if d is None or d.empty: continue
            d.index = _to_jst_index(d.index)
            close = _pick_close_series(d)
            if close is None or close.empty: continue
            frames.append(close.rename(name))
        except Exception as e:
            print(f"[warn] daily fetch failed: {tk} - {e}")

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
    os.makedirs(OUT_DIR, exist_ok=True)
    intraday = build_intraday_series()
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
        last_lvl = float(lv["level"].iloc[-1])
        if np.isfinite(last_lvl):
            stats["last_level"] = round(last_lvl, 2)
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
