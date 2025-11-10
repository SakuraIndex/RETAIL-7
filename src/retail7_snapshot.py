# src/retail7_snapshot.py
# ==========================================
# RETAIL-7（日本小売業7社・等金額加重）インラデイスナップショット
# 生成物:
#   docs/outputs/retail_7_intraday.csv
#   docs/outputs/retail_7_stats.json
#   docs/outputs/retail_7_post_intraday.txt（互換: docs/outputs/post_intraday.txt も出力）
#   docs/outputs/last_run.txt
# ==========================================

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------- 基本設定 ----------
KEY = "RETAIL-7"
OUTPUT_DIR = Path("docs/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 画像指定の7社（Yahoo Finance ティッカー）
TICKERS: Dict[str, str] = {
    "イオン": "8267.T",
    "ファーストリテイリング": "9983.T",
    "PPIH": "7532.T",       # 画像の PPIF を PPIH に読み替え
    "セブン＆アイ": "3382.T",
    "三越伊勢丹": "3099.T",
    "トライアルHD": "141A.T",
    "高島屋": "8233.T",
}
TICKER_LIST: List[str] = list(TICKERS.values())

# 出力ファイル
CSV_INTRADAY = OUTPUT_DIR / "retail_7_intraday.csv"
STATS_JSON = OUTPUT_DIR / "retail_7_stats.json"
POST_TXT = OUTPUT_DIR / "retail_7_post_intraday.txt"
POST_TXT_LEGACY = OUTPUT_DIR / "post_intraday.txt"
LAST_RUN = OUTPUT_DIR / "last_run.txt"

TZ_JST = "Asia/Tokyo"


# ---------- 小道具 ----------
def now_jst() -> datetime:
    return datetime.now(timezone.utc).astimezone(pd.Timestamp.now(tz=TZ_JST).tz)


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

    # 単一レベル
    if not isinstance(df.columns, pd.MultiIndex):
        for cand in ["Close", "Adj Close", "close", "AdjClose"]:
            if cand in df.columns:
                s = _as_series(df[cand])
                return pd.to_numeric(s, errors="coerce") if s is not None else None

    # MultiIndex（最後のレベル名が Close/Adj Close）
    if isinstance(df.columns, pd.MultiIndex):
        for cand in ["Close", "Adj Close"]:
            try:
                mask = np.array([t[-1] == cand for t in df.columns])
                sub = df.loc[:, mask]
                s = _as_series(sub)
                if s is not None and not s.empty:
                    return pd.to_numeric(s, errors="coerce")
            except Exception:
                pass

    return None


def _download_series_5m(ticker: str) -> pd.Series:
    """
    ティッカーの 5分足終値 Series を取得（UTC index）。
    1分足だと「当日データがまだ無い」ケースが出やすいので 5分足を採用。
    """
    df = yf.download(
        ticker,
        period="30d",           # 日本株の 5m は最大60日程度。余裕を持って30d
        interval="5m",
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    s = _pick_close_series(df)
    if s is None:
        return pd.Series(dtype=float)

    idx = pd.to_datetime(s.index, utc=True)
    s = pd.Series(s.values, index=idx, name=ticker).sort_index()
    return s


def _latest_session_slice(df_close_jst: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame(列=銘柄, index=JST) から
    - 当日(JST)にデータがあれば当日を採用
    - 無ければ直近の営業日（行数>10 など最低限のボリューム）を採用
    """
    if df_close_jst.empty:
        return df_close_jst

    today = pd.Timestamp.now(TZ_JST).normalize()
    m_today = (df_close_jst.index >= today) & (df_close_jst.index < today + pd.Timedelta(days=1))
    today_df = df_close_jst.loc[m_today]
    if not today_df.empty:
        return today_df

    # 直近セッション（1日のデータ）を拾う
    g = df_close_jst.groupby(df_close_jst.index.tz_convert(TZ_JST).date)
    # 最小でも10行くらい欲しい（約1時間ぶん）
    eligible = [d for d, x in g if len(x.dropna(how="all")) >= 10]
    if not eligible:
        return pd.DataFrame(index=[], columns=df_close_jst.columns)

    last_date = max(eligible)
    last_day = pd.Timestamp(last_date, tz=TZ_JST)
    return df_close_jst.loc[(df_close_jst.index >= last_day) &
                            (df_close_jst.index < last_day + pd.Timedelta(days=1))]


# ---------- コア処理 ----------
def build_intraday_series() -> pd.DataFrame:
    """
    等金額加重“対始値騰落率(%)”の時系列を返す。
    5分足で各銘柄 Close を結合 → 当日(JST)が無ければ直近営業日を自動採用。
    """
    # 各銘柄 5m Series を UTC で集める
    s_map: Dict[str, pd.Series] = {}
    for _, tic in TICKERS.items():
        s = _download_series_5m(tic)
        if not s.empty:
            s_map[tic] = s

    if not s_map:
        raise RuntimeError("no prices at all")

    # JST に変換して横持ち
    df_close = pd.DataFrame(s_map).sort_index()
    df_close_jst = df_close.tz_convert(TZ_JST)

    # 当日 or 直近営業日を抽出
    ses = _latest_session_slice(df_close_jst)
    if ses.empty:
        raise RuntimeError("no prices for today (ET)")

    # 始値（その日の最初の非NaN）で正規化 → % へ
    open_prices = ses.apply(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
    valid = [c for c in ses.columns if np.isfinite(open_prices.get(c, np.nan)) and open_prices.get(c, np.nan) > 0]
    if not valid:
        raise RuntimeError("no valid open prices")

    df_pct = (ses[valid] / open_prices[valid] - 1.0) * 100.0
    eq_weight = df_pct.mean(axis=1)  # 等金額加重 = 単純平均

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


def heartbeat() -> None:
    LAST_RUN.write_text(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), encoding="utf-8")


def main():
    df = build_intraday_series()
    write_csv(df)
    write_stats_and_post(df)
    heartbeat()


if __name__ == "__main__":
    main()
