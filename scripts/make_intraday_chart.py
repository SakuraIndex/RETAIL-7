# scripts/make_intraday_chart.py  — 完全版（列名自動検出 & ダークスタイル）

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("docs/outputs")
CSV_PATH   = OUTPUT_DIR / "retail_7_intraday.csv"
PNG_PATH   = OUTPUT_DIR / "retail_7_intraday.png"

TITLE = "RETAIL-7 Intraday Snapshot (JST)"

def load_series() -> pd.Series:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    # 列名を自動検出（古いスクリプト互換）
    cand_cols = ["retail7_pct", "pct"]
    col = next((c for c in cand_cols if c in df.columns), None)
    if col is None:
        raise KeyError(f"No target column found in {CSV_PATH}. tried={cand_cols}, actual={list(df.columns)}")

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    # index は JST（retail7_snapshot.py で JST を保存）→ そのまま使用
    s.index.name = "datetime_jst"
    return s

def plot(s: pd.Series) -> None:
    # スタイル（黒ベース）
    plt.rcParams.update({
        "figure.figsize": (16, 9),
        "figure.dpi": 160,
        "axes.facecolor": "#0b1420",
        "figure.facecolor": "#0b1420",
        "axes.edgecolor": "#2a3a4a",
        "text.color": "#d4e9f7",
        "axes.labelcolor": "#cfe6f3",
        "xtick.color": "#9fb6c7",
        "ytick.color": "#9fb6c7",
        "grid.color": "#1c2a3a",
        "grid.linestyle": "-",
        "grid.alpha": 0.35,
    })

    fig, ax = plt.subplots()

    y = s.copy()
    x = y.index

    color_up   = "#22d3ee"   # ティール
    color_down = "#fb7185"   # サーモン
    line_color = color_up if y.iloc[-1] >= 0 else color_down
    fill_color = (*matplotlib.colors.to_rgba(line_color)[:3], 0.18)

    ax.plot(x, y.values, linewidth=2.2, color=line_color)
    ax.fill_between(x, np.minimum(y.values, 0), y.values, where=y.values>=0,
                    color=fill_color, interpolate=True)
    ax.fill_between(x, y.values, np.minimum(y.values, 0), where=y.values<0,
                    color=fill_color, interpolate=True)

    ax.axhline(0, color="#2a3a4a", linewidth=1.2)
    ax.grid(True, axis="both")

    ax.set_title(TITLE, fontsize=16, weight="bold")
    ax.set_ylabel("Change vs Open (%)")
    ax.set_xlabel("")

    # 余白
    fig.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, bbox_inches="tight")
    plt.close(fig)

def main():
    s = load_series()
    if s.empty:
        raise RuntimeError("No data to plot.")
    plot(s)

if __name__ == "__main__":
    main()
