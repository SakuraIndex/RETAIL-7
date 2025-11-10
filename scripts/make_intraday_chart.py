# coding: utf-8
import os, json
from datetime import timezone, timedelta
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "docs/outputs"
CSV = os.path.join(OUT_DIR, "retail_7_intraday.csv")
STATS = os.path.join(OUT_DIR, "retail_7_stats.json")
PNG = os.path.join(OUT_DIR, "retail_7_intraday.png")

JST = timezone(timedelta(hours=9))

def load():
    df = pd.read_csv(CSV)
    # 既にJST文字列 → pandasに食わせる
    df["datetime_jst"] = pd.to_datetime(df["datetime_jst"])
    df = df.set_index("datetime_jst")
    with open(STATS, "r", encoding="utf-8") as f:
        s = json.load(f)
    return df, s

def main():
    df, s = load()
    pct = float(s.get("pct_intraday") or 0.0)
    # 線の色（上げ=ややエメラルド / 下げ=ややピンク）
    color = "#34d399" if pct >= 0 else "#fb7185"

    plt.figure(figsize=(14, 6), dpi=140)
    ax = plt.gca()
    ax.set_facecolor("#0b1420")
    plt.rcParams["savefig.facecolor"] = "#0b1420"

    x = df.index
    y = df["retail7_pct"]

    ax.plot(x, y, linewidth=2.0, color="#60f0e0")  # 線は青緑で統一
    ax.fill_between(x, y, 0, where=(y>=0), alpha=0.18, color="#10b981")
    ax.fill_between(x, y, 0, where=(y<0),  alpha=0.18, color="#ef4444")

    ax.grid(color="#1c2a3a", alpha=0.6)
    ax.spines[:].set_color("#1c2a3a")
    ax.tick_params(colors="#cfe6f3")
    ax.set_ylabel("Change vs Open (%)", color="#cfe6f3")
    ax.set_title(f"RETAIL-7 Intraday Snapshot ({pd.Timestamp.now(tz=JST).strftime('%Y/%m/%d %H:%M JST')})",
                 color="#d4e9f7", fontweight="bold")

    plt.savefig(PNG, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
