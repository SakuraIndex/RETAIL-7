# coding: utf-8
import json, os

OUT_DIR = "docs/outputs"
STATS = os.path.join(OUT_DIR, "retail_7_stats.json")
POST = os.path.join(OUT_DIR, "retail_7_post_intraday.txt")

def main():
    s = json.load(open(STATS, "r", encoding="utf-8"))
    pct = s.get("pct_intraday", 0.0)
    post = []
    post.append("【RETAIL-7｜小売業指数】")
    sign = "+" if float(pct) >= 0 else ""
    post.append(f"本日：{sign}{pct:.2f}%")
    post.append("等金額加重・JPN")
    post.append("#桜Index #Retail7")
    open(POST, "w", encoding="utf-8").write("\n".join(post))

if __name__ == "__main__":
    main()
