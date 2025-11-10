# src/retail7_snapshot.py — 完全版（Open/Close をどの列順でも取得・5m→日足フォールバック）

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

KEY = "RETAIL-7"
OUTPUT_DIR = Path("docs/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 構成銘柄（Yahoo!Finance ティッカー）
TICKERS: Dict[str, str] = {
    "イオン": "8267.T",
    "ファーストリテイリング": "9983.T",
    "PPIH": "7532.T",       # PPIF→PPIH に読み替え
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


def _extract_field_anyorder(df: pd.DataFrame, field: str) -> Optional[pd.Series]:
    """
    df.columns が
      - 単一列:  ['Open','High',...]
      - MultiIndex: [(ticker, 'Open'), ...] もしくは [('Open', ticker), ...]
    のいずれでも、指定 field（'Open' / 'Close' / 'Adj Close'）を取り出す。
    """
    if df is None or df.empty:
        return None

    # 単一列
    if not isinstance(df.columns, pd.MultiIndex):
        if field in df.columns:
            s = _as_series(df[[field]])
            return pd.to_numeric(s, errors="coerce") if s is not None else None
        return None

    # MultiIndex: 任意のレベル順で field を含む列を抽出
    mask = []
    for col in df.columns:
        if isinstance(col, tuple) and any(str(level) == field for level in col):
            mask.append(True)
        else:
            mask.append(False)

    if any(mask):
        sub = df.loc[:, mask]
        s = _as_series(sub)
        return pd.to_numeric(s, errors="coerce") if s is not None else None
    return None


def _pick_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """5分足側の終値列抽出（MultiIndex/単一列対応）。"""
    if df is None or df.empty:
        return None

    # 単一列
    if not isinstance(df.columns, pd.MultiIndex):
        for cand in ["Close", "Adj Close", "close", "AdjClose"]:
            if cand in df.columns:
                s = _as_series(df[cand])
                return pd.to_numeric(s, errors="coerce") if s is not None else None

    # MultiIndex
    for cand in ["Close", "Adj Close"]:
        cols = [i for i, c in enumerate(df.columns)
                if isinstance(c, tuple) and any(str(level) == cand for level in c)]
        if cols:
            sub = df.iloc[:, cols]
            s = _as_series(sub)
            if s is not None and not s.empty:
                return pd.to_numeric(s, errors="coerce")
    return None


def _download_series_5m(ticker: str) -> pd.Series:
    df = yf.download(
        ticker,
        period="30d",
        interval="5m",
        auto_adjust=False,
        group_by="column",      # 単一列化を優先（環境差でMultiになる場合もある）
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
    5分足が取れない場合のフォールバック：
      直近営業日の Open/Close から JST 09:00/15:00 の2点時系列を合成。
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

    s_open  = _extract_field_anyorder(df, "Open")
    s_close = _extract_field_anyorder(df, "Close")
    if s_open is None or s_close is None:
        return pd.Series(dtype=float)
    s_open, s_close = s_open.dropna(), s_close.dropna()
    if s_open.empty or s_close.empty:
        return pd.Series(dtype=float)

    # 直近日のインデックス（yfinanceはUTC naive or tzなし→UTC想定）
    last_day = s_close.index[-1]
    last_day = pd.to_datetime(last_day).tz_localize("UTC") if pd.Timestamp(last_day).tzinfo is None else pd.Timestamp(last_day).tz_convert("UTC")

    # JST 09:00 / 15:00
    day_jst = last_day.tz_convert(TZ_JST).normalize()
    t_open  = day_jst + pd.Timedelta(hours=9)
    t_close = day_jst + pd.Timedelta(hours=15)

    try:
        open_val  = float(s_open.loc[last_day])
        close_val = float(s_close.loc[last_day])
    except KeyError:
        # インデックスの微小差異(naive/aware)対策：日付同士で揃える
        s_open_d  = s_open.copy()
        s_close_d = s_close.copy()
        s_open_d.index  = pd.to_datetime(s_open_d.index).tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
        s_close_d.index = pd.to_datetime(s_close_d.index).tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
        open_val  = float(s_open_d.iloc[-1])
        close_val = float(s_close_d.iloc[-1])

    if not (np.isfinite(open_val) and np.isfinite(close_val) and open_val > 0):
        return pd.Series(dtype=float)

    idx_utc = pd.DatetimeIndex([t_open, t_close]).tz_convert("UTC")
    vals = [open_val, close_val]
    return pd.Series(vals, index=idx_utc, name=ticker)


def _latest_session_slice(df_close_jst: pd.DataFrame) -> pd.DataFrame:
    """当日が空なら直近営業日の1日分を返す（2点フォールバックも可）。"""
    if df_close_jst.empty:
        return df_close_jst

    today = pd.Timestamp.now(TZ_JST).normalize()
    today_df = df_close_jst[(df_close_jst.index >= today) &
                            (df_close_jst.index <  today + pd.Timedelta(days=1))]
    if not today_df.empty:
        return today_df

    # 直近で2点以上ある日を使用
    g = df_close_jst.groupby(df_close_jst.index.tz_convert(TZ_JST).date)
    eligible = [d for d, x in g if len(x.dropna(how="all")) >= 2]
    if not eligible:
        return pd.DataFrame(index=[], columns=df_close_jst.columns)
    last_date = max(eligible)
    last_day = pd.Timestamp(last_date, tz=TZ_JST)
    return df_close_jst[(df_close_jst.index >= last_day) &
                        (df_close_jst.index <  last_day + pd.Timedelta(days=1))]


def build_intraday_series() -> pd.DataFrame:
    # 5分足優先で取得、だめな銘柄は日足フォールバック
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

    df_close = pd.DataFrame(s_map).sort_index()
    df_close_jst = df_close.tz_convert(TZ_JST)

    ses = _latest_session_slice(df_close_jst)
    if ses.empty:
        raise RuntimeError("no prices for today (ET)")

    # 始値で正規化 → 等金額加重(%)
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
