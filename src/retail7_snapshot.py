# coding: utf-8
"""
Retail-7 (Japan Retail) equal-dollar-weighted index
- Intraday 1m -> 5min (JST) で当日 "開場からの騰落率(%)" を算出し平均
- 出力:
  - docs/outputs/retail_7_intraday.csv   : 当日の5分足( % vs open )
  - docs/outputs/retail_7_stats.json     : pct_intraday / updated_at(JST) / last_level(任意) など
  - docs/outputs/retail_7_levels.csv     : （Long ワークフロー側でも再利用）日足レベル（100起点）
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
#  構成銘柄（yfinance ティッカー）
#  ※ PPIF は誤記っぽいので「PPIH(7532.T)」にマッピングしています
#  ※ トライアルHDは 5885.T （等金額加重、取得不可なら自動スキップ）
# ─────────────────────────────────────────────────────────────
TICKERS = {
    "AEON": "8267.T",            # イオン
    "FAST_RETAIL": "9983.T",     # ファーストリテイリング
    "PPIH": "7532.T",            # パンパシフィック（ドンキ）
    "SEVEN_I": "3382.T",         # セブン&アイ
    "IMH": "3099.T",             # 三越伊勢丹HD
    "TRIAL": "5885.T",           # トライアルHD
    "TAKASHIMAYA": "8233.T",     # 高島屋
}

INDEX_KEY = "RETAIL_7"
BASE_DATE = "2024-01-04"  # 市場営業日で100起点（必要に応じて調整）

JST = timezone(timedelta(hours=9))


def _now_jst():
    return datetime.now(JST)


def _to_jst(ts: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    # yfinanceは基本UTC。tz情報の有無に応じて安全にJSTへ。
    if hasattr(ts, "tz"):
        if ts.tz is None:
            return ts.tz_localize("UTC").tz_convert("Asia/Tokyo")
        return ts.tz_convert("Asia/Tokyo")
    # Series index
    idx = ts.index if hasattr(ts, "index") else ts
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert("Asia/Tokyo")


def _fetch_intraday_1m(ticker: str) -> pd.DataFrame | None:
    """
    当日分の1分足（最大7日分）からJST当日だけを抽出。
    """
    try:
        df = yf.download(ticker, interval="1m", period="7d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.tz_localize("UTC") if df.index.tz is None else df
        df = df.tz_convert("Asia/Tokyo")
        # 当日のみ
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
    2) 当日寄り付き値(9:00 JST 付近の最初の価格)を基準に「% vs Open」を算出
    3) 5分足へ集計（同時刻の平均＝等金額加重）
    """
    series_list = []
    used = []

    for name, tk in TICKERS.items():
        df = _fetch_intraday_1m(tk)
        if df is None or df.empty:
            continue

        # 開場直後の最初の有効価格を Open_of_Day とする
        open_price = df["Close"].iloc[0]
        if not np.isfinite(open_price) or open_price <= 0:
            continue

        pct = (df["Close"] / open_price - 1.0) * 100.0
        s = pct.to_frame(name=name)
        series_list.append(s)
        used.append(name)

    if not series_list:
        raise RuntimeError("no prices for today (JST)")

    mat = pd.concat(series_list, axis=1, join="outer").sort_index()
    # 5分足リサンプル（平均）
    mat_5 = mat.resample("5min").mean()

    # 等金額加重＝単純平均
    mat_5["EQUAL_AVG"] = mat_5[used].mean(axis=1, skipna=True)

    # Nullの前方補完（板が薄い銘柄の欠損を緩和）
    mat_5 = mat_5.ffill()

    # 出力用整形
    out = mat_5[["EQUAL_AVG"]].rename(columns={"EQUAL_AVG": "retail7_pct"})
    out.index.name = "datetime_jst"
    return out


def build_levels_daily() -> pd.DataFrame:
    """
    日足で等金額加重の「レベル」を生成。
    - 各銘柄の日次終値を同列化
    - 列方向で平均して 100 起点のレベルに
    """
    frames = []
    for name, tk in TICKERS.items():
        try:
            d = yf.download(tk, interval="1d", period="2y", progress=False, auto_adjust=False)
            if d is None or d.empty:
                continue
            d = d.tz_localize("UTC") if d.index.tz is None else d
            d = d.tz_convert("Asia/Tokyo")
            s = d["Close"].rename(name)
            frames.append(s)
        except Exception as e:
            print(f"[warn] daily fetch failed: {tk} - {e}")
            continue

    if not frames:
        raise RuntimeError("daily levels empty")

    mat = pd.concat(frames, axis=1, join="inner").sort_index()
    lvl = mat.mean(axis=1, skipna=True)  # 等金額＝平均
    # 100起点
    base = pd.to_datetime(BASE_DATE).tz_localize("Asia/Tokyo")
    if base not in lvl.index:
        # 最寄り営業日に合わせる
        base = lvl.index[lvl.index.get_indexer([base], method="nearest")]
        base = base[0]
    base_val = float(lvl.loc[base])
    level = lvl / base_val * 100.0
    df_out = level.to_frame("level")
    df_out.index.name = "date_jst"
    return df_out


def main():
    # Intraday 5分足 (%)
    intraday = build_intraday_series()
    intraday.to_csv(os.path.join(OUT_DIR, "retail_7_intraday.csv"), encoding="utf-8")

    # 今日の騰落率（最新値）
    last_pct = float(intraday["retail7_pct"].iloc[-1])
    stats = {
        "key": INDEX_KEY,
        "pct_intraday": round(last_pct, 2),
        "updated_at": _now_jst().strftime("%Y/%m/%d %H:%M"),
        "unit": "pct",
        "last_level": None,     # long 側で上書き可能
        "tickers": list(TICKERS.values()),
    }
    with open(os.path.join(OUT_DIR, "retail_7_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # levels（日足）もついでに生成（Long でも毎回再計算可）
    try:
        lv = build_levels_daily()
        lv.to_csv(os.path.join(OUT_DIR, "retail_7_levels.csv"), encoding="utf-8")
        if not math.isnan(lv["level"].iloc[-1]):
            stats["last_level"] = round(float(lv["level"].iloc[-1]), 2)
            with open(os.path.join(OUT_DIR, "retail_7_stats.json"), "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[warn] levels build failed:", e)

    # X 投稿文（任意）
    post = []
    post.append("【RETAIL-7｜小売業指数】")
    post.append(f"本日：{('+%.2f' % last_pct) if last_pct >= 0 else ('%.2f' % last_pct)}%")
    post.append(f"構成：{','.join(TICKERS.values())}")
    post.append("#桜Index #Retail7")
    with open(os.path.join(OUT_DIR, "retail_7_post_intraday.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(post))

    # 最終実行印
    with open(os.path.join(OUT_DIR, "last_run.txt"), "w", encoding="utf-8") as f:
        f.write(_now_jst().strftime("%Y/%m/%d %H:%M (JST)"))


if __name__ == "__main__":
    main()
