# src/retail7_snapshot.py
# ==========================================
# RETAIL-7（日本小売業7社・等金額加重）の当日インラデイ用スナップショット生成
# - docs/outputs/retail_7_intraday.csv        … 5分足等で揃えた対始値騰落率（%）
# - docs/outputs/retail_7_stats.json          … pct_intraday / 更新時刻 など
# - docs/outputs/retail_7_post_intraday.txt   … X（旧Twitter）投稿用テキスト
# - docs/outputs/last_run.txt                 … 心拍確認
#
# ※ yfinance の都合で 1分足は直近7日程度しか取得できません。
# ※ 取引時間は UTC → JST に変換し、当日(JST)のデータのみを対象にします。
# ==========================================

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------- 設定 ----------
KEY = "RETAIL-7"
OUTPUT_DIR = Path("docs/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 構成銘柄（Yahoo Finance 形式）
# 画像で指定いただいた銘柄を一般的なティッカーへマッピングしています。
# 必要に応じて調整してください。
TICKERS: Dict[str, str] = {
    "イオン": "8267.T",                # AEON
    "ファーストリテイリング": "9983.T",  # FAST RETAILING
    "PPIH": "7532.T",                  # Pan Pacific International Holdings（画像の PPIF 想定）
    "セブン＆アイ": "3382.T",             # Seven & i
    "三越伊勢丹": "3099.T",               # Isetan Mitsukoshi
    "トライアルHD": "141A.T",            # Trial Holdings（新コード想定）
    "高島屋": "8233.T",                 # Takashimaya
}
TICKER_LIST: List[str] = list(TICKERS.values())

# 出力ファイル名
CSV_INTRADAY = OUTPUT_DIR / "retail_7_intraday.csv"
STATS_JSON = OUTPUT_DIR / "retail_7_stats.json"
POST_TXT = OUTPUT_DIR / "retail_7_post_intraday.txt"
POST_TXT_LEGACY = OUTPUT_DIR / "post_intraday.txt"  # 互換
LAST_RUN = OUTPUT_DIR / "last_run.txt"

TZ_JST = "Asia/Tokyo"


# ---------- ユーティリティ ----------
def now_jst() -> datetime:
    return datetime.now(timezone.utc).astimezone(pd.Timestamp.now(tz=TZ_JST).tz)


def fmt_jst(ts: Optional[pd.Timestamp | datetime]) -> str:
    if ts is None:
        return ""
    if isinstance(ts, pd.Timestamp):
        ts = ts.tz_convert(TZ_JST) if ts.tzinfo else ts.tz_localize(TZ_JST)
    return ts.strftime("%Y/%m/%d %H:%M")


def _as_series(obj) -> Optional[pd.Series]:
    """DataFrame/Series → Series に正規化"""
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
    """
    df から終値 Series を安全に抽出する。
    - 'Close' / 'Adj Close' を優先
    - DataFrame が返る場合は 1 列だけ選び Series 化
    - 列が MultiIndex の場合でも、最後のレベル名で Close/Adj Close を拾う
    """
    # 1) 単一レベル列
    if not isinstance(df.columns, pd.MultiIndex):
        for cand in ["Close", "Adj Close", "close", "AdjClose", "adj close"]:
            if cand in df.columns:
                s = _as_series(df[cand])
                return pd.to_numeric(s, errors="coerce") if s is not None else None

    # 2) MultiIndex 列
    if isinstance(df.columns, pd.MultiIndex):
        for cand in ["Close", "Adj Close"]:
            try:
                mask = np.array([tup[-1] == cand for tup in df.columns])
                sub = df.loc[:, mask]
                s = _as_series(sub)
                if s is not None and not s.empty:
                    return pd.to_numeric(s, errors="coerce")
            except Exception:
                pass

    return None


def _download_minutely(ticker: str) -> pd.Series:
    """
    単一ティッカーの 1分足終値 Series（UTC index）を返す。
    見つからなければ空 Series。
    """
    df = yf.download(
        ticker,
        period="7d",
        interval="1m",
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

    # UTC に整える（yfinance 1m は tz-naive になりがち）
    idx = pd.to_datetime(s.index, utc=True)
    s = pd.Series(s.values, index=idx, name=ticker).sort_index()
    return s


def _today_jst_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    # 当日 00:00 ~ 23:59:59（JST）
    today = pd.Timestamp.now(TZ_JST).normalize()
    tomorrow = today + pd.Timedelta(days=1)
    return today, tomorrow


def build_intraday_series() -> pd.DataFrame:
    """
    等金額加重の“対始値騰落率”の時系列を返す（列: pct）
    - 横持ち: 各ティッカーの Close
    - 先頭値を“当日最初の値”として騰落率を算出
    - 最後に全銘柄の平均を取り “等金額加重” を実現
    """
    t0_jst, t1_jst = _today_jst_range()

    series_map: Dict[str, pd.Series] = {}
    for name, tic in TICKERS.items():
        s_utc = _download_minutely(tic)
        if s_utc.empty:
            continue
        # JST に変換し、当日分だけに限定
        s_jst = s_utc.tz_convert(TZ_JST)
        s_jst = s_jst[(s_jst.index >= t0_jst) & (s_jst.index < t1_jst)]
        if s_jst.empty:
            continue
        series_map[tic] = s_jst

    if not series_map:
        raise RuntimeError("no prices for today (ET)")

    # インデックスの和集合で結合 → 1分 → 5分（ノイズ・ファイルサイズ抑制）
    df_close = pd.DataFrame(series_map).sort_index()
    df_close = df_close.resample("5min").last()

    # 各銘柄の“当日最初の値”で正規化 → 騰落率(%) を計算
    open_prices = df_close.apply(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
    # 異常値防止（ゼロ / NaN は除外）
    valid_cols = [c for c in df_close.columns if np.isfinite(open_prices.get(c, np.nan)) and open_prices.get(c, np.nan) > 0]
    if not valid_cols:
        raise RuntimeError("no valid open prices")

    df_pct = (df_close[valid_cols] / open_prices[valid_cols] - 1.0) * 100.0
    intraday_pct = df_pct.mean(axis=1)  # 等金額加重 = 単純平均

    # 出力用 DataFrame
    out = pd.DataFrame({"pct": intraday_pct})
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
        ts_last = df.index[-1].tz_convert(TZ_JST)

    stats = {
        "key": KEY,
        "pct_intraday": round(pct_now, 2) if np.isfinite(pct_now) else None,
        "updated_at": fmt_jst(ts_last) if ts_last is not None else None,
        "unit": "pct",
        "last_level": round(1000.0 * (1.0 + (pct_now / 100.0)), 2) if np.isfinite(pct_now) else None,
        "tickers": TICKER_LIST,
    }
    STATS_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # X（旧Twitter）用文面
    line = f"{KEY} {pct_now:+.2f}% ({fmt_jst(ts_last)})\n指数: {stats['last_level']}\n構成: " + ",".join(TICKER_LIST) + "\n#桜Index #Retail7"
    POST_TXT.write_text(line, encoding="utf-8")
    # 互換（既存フローが post_intraday.txt を参照する場合）
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
