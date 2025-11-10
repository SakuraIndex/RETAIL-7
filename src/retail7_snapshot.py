# src/retail7_snapshot.py  — 完全版（5m→日足フォールバック付き）
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

KEY = "RETAIL-7"
OUTPUT_DIR = Path("docs/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 画像で指定の7社（Yahoo!Finance ティッカー）
TICKERS: Dict[str, str] = {
    "イオン": "8267.T",
    "ファーストリテイリング": "9983.T",
    "PPIH": "7532.T",       # PPIF→PPIH 読み替え
    "セブン＆アイ": "3382.T",
    "三越伊勢丹": "3099.T",
    "トライアルHD": "141A.T",
    "高島屋": "8233.T",
}
TICKER_LIST: List[str] = list(TICKERS.values())

CSV_INTRADAY = OUTPUT_DIR / "retail_7_intraday.csv"
STATS_JSON   = OUTPUT_DIR / "retail_7_stats.json"
POST_TXT     = OUTPUT_DIR / "retail_7_post_intraday.txt"
POST_TXT_LEGACY = OUTPUT_DIR / "post_intraday.txt"
LAST_RUN     = OUTPUT_DIR / "last_run.txt"

TZ_JST = "Asia/Tokyo"


def fmt_jst(ts: Optional[pd.Timestamp | datetime]) -> str:
    if ts is None:
        return ""
    if isinstance(ts, pd.Timestamp):
        ts = ts.tz_convert(TZ_JST) if ts.tzinfo else ts.tz_localize(TZ_JST)
    return ts.strftime("%Y/%m/%d %H:%M")


def _as_series(obj) -> Optional[pd.Series]:
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return None
        return obj.iloc[:, 0]
    return None


def _pick_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """終値列を安全に Series 化（単一列/MultiIndex 両対応）"""
    if df is None or df.empty:
        return None

    if not isinstance(df.columns, pd.MultiIndex):
        for cand in ["Close", "Adj Close", "close", "AdjClose"]:
            if cand in df.columns:
                s = _as_series(df[cand])
                return pd.to_numeric(s, errors="coerce") if s is not None else None

    if isinstance(df.columns, pd.MultiIndex):
        for cand in ["Close", "Adj Close"]:
            mask = np.array([t[-1] == cand for t in df.columns])
            if mask.any():
                sub = df.loc[:, mask]
                s = _as_series(sub)
                if s is not None and not s.empty:
                    return pd.to_numeric(s, errors="coerce")

    return None


def _download_series_5m(ticker: str) -> pd.Series:
    """5分足終値 Series（UTC index）。空なら長さ0のSeries。"""
    df = yf.download(
        ticker,
        period="30d",
        interval="5m",
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    s = _pick_close_series(df)
    if s is None or s.empty:
        return pd.Series(dtype=float)

    idx = pd.to_datetime(s.index, utc=True)
    return pd.Series(s.values, index=idx, name=ticker).sort_index()


def _fallback_daily_series(ticker: str) -> pd.Series:
    """
    5mが取れない場合のフォールバック：
    直近営業日の始値・終値から 2点の“擬似 intraday”を合成（JST 09:00 と 15:00）。
    """
    df = yf.download(
        ticker,
        period="10d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # 列の取り出し（MultiIndex/単一列両対応）
    if isinstance(df.columns, pd.MultiIndex):
        get = lambda col: _as_series(df.loc[:, pd.IndexSlice[:, col]])
    else:
        get = lambda col: _as_series(df[[col]])

    s_open = get("Open")
    s_close = get("Close")
    if s_open is None or s_close is None or s_open.dropna().empty or s_close.dropna().empty:
        return pd.Series(dtype=float)

    last_day = s_close.dropna().index[-1]  # UTC の日付Index
    # JST で 09:00 と 15:00 を作る
    day_jst = pd.Timestamp(last_day).tz_localize("UTC").tz_convert(TZ_JST).normalize()
    t_open = day_jst + pd.Timedelta(hours=9)
    t_close = day_jst + pd.Timedelta(hours=15)

    open_val = float(s_open.loc[last_day])
    close_val = float(s_close.loc[last_day])
    if not (np.isfinite(open_val) and np.isfinite(close_val) and open_val > 0):
        return pd.Series(dtype=float)

    idx = pd.DatetimeIndex([t_open, t_close]).tz_convert("UTC")  # 統一してUTC indexへ
    vals = [open_val, close_val]
    return pd.Series(vals, index=idx, name=ticker)


def _latest_session_slice(df_close_jst: pd.DataFrame) -> pd.DataFrame:
    """当日が空なら直近営業日の1日分を返す。"""
    if df_close_jst.empty:
        return df_close_jst

    today = pd.Timestamp.now(TZ_JST).normalize()
    m_today = (df_close_jst.index >= today) & (df_close_jst.index < today + pd.Timedelta(days=1))
    today_df = df_close_jst.loc[m_today]
    if not today_df.empty:
        return today_df

    g = df_close_jst.groupby(df_close_jst.index.tz_convert(TZ_JST).date)
    eligible = [d for d, x in g if len(x.dropna(how="all")) >= 2]  # フォールバックは2点なので>=2でOK
    if not eligible:
        return pd.DataFrame(index=[], columns=df_close_jst.columns)

    last_date = max(eligible)
    last_day = pd.Timestamp(last_date, tz=TZ_JST)
    return df_close_jst.loc[(df_close_jst.index >= last_day) &
                            (df_close_jst.index < last_day + pd.Timedelta(days=1))]


def build_intraday_series() -> pd.DataFrame:
    """
    等金額加重“対始値騰落率(%)”の時系列。
    5分足を優先し、銘柄ごとに取得不可なら日足から擬似 intraday を合成して補完。
    """
    s_map: Dict[str, pd.Series] = {}
    for tic in TICKER_LIST:
        s5 = _download_series_5m(tic)
        if not s5.empty:
            s_map[tic] = s5
            continue
        sf = _fallback_daily_series(tic)
        if not sf.empty:
            s_map[tic] = sf

    if not s_map:
        raise RuntimeError("no prices at all")

    # JST に揃えて横持ち
    df_close = pd.DataFrame(s_map).sort_index()
    df_close_jst = df_close.tz_convert(TZ_JST)

    ses = _latest_session_slice(df_close_jst)
    if ses.empty:
        raise RuntimeError("no prices for today (ET)")

    # 始値（最初の非NaN）で規格化し、等金額加重の平均％
    open_prices = ses.apply(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
    valid_cols = [c for c in ses.columns if np.isfinite(open_prices.get(c, np.nan)) and open_prices.get(c, np.nan) > 0]
    if not valid_cols:
        raise RuntimeError("no valid open prices")

    df_pct = (ses[valid_cols] / open_prices[valid_cols] - 1.0) * 100.0
    eq_weight = df_pct.mean(axis=1)

    out = pd.DataFrame({"pct": eq_weight})
    out.index.name = "datetime_jst"
    return out


def write_csv(df: pd.DataFrame) -> None:
    df.to_csv(CSV_INTRADAY, float_format="%.6f")


def write_stats_and_post(df: pd.DataFrame) -> None:
    if df.empty:
        pct_now = float("nan")
        ts_last = None
    else:
        pct_now = float(df["pct"].iloc[-1])
        ts_last = df.index[-1]

    last_level = round(1000.0 * (1.0 + (pct_now / 100.0)), 2) if np.isfinite(pct_now) else None
    stats = {
        "key": KEY,
        "pct_intraday": round(pct_now, 2) if np.isfinite(pct_now) else None,
        "updated_at": fmt_jst(ts_last),
        "unit": "pct",
        "last_level": last_level,
        "tickers": TICKER_LIST,
    }
    STATS_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    line = (
        f"{KEY} {pct_now:+.2f}% ({fmt_jst(ts_last)})\n"
        f"指数: {last_level}\n"
        f"構成: {','.join(TICKER_LIST)}\n"
        "#桜Index #Retail7"
    )
    POST_TXT.write_text(line, encoding="utf-8")
    POST_TXT_LEGACY.write_text(line, encoding="utf-8")


def main():
    df = build_intraday_series()
    write_csv(df)
    write_stats_and_post(df)
    LAST_RUN.write_text(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), encoding="utf-8")


if __name__ == "__main__":
    main()
