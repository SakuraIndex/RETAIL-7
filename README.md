# RETAIL-7 Sakura Index（小売業指数）

- 構成（全て東証、等金額加重）:
  - 8267.T（イオン）, 9983.T（ファーストリテイリング）, 7532.T（PPIH）
  - 3382.T（セブン＆アイ）, 3099.T（三越伊勢丹HD）, 5885.T（トライアルHD）
  - 8233.T（高島屋）

## 出力
- `docs/outputs/retail_7_intraday.csv` … 当日5分足（% vs open, JST）
- `docs/outputs/retail_7_intraday.png` … 黒背景スナップショット
- `docs/outputs/retail_7_stats.json` … `pct_intraday`, `updated_at` など
- `docs/outputs/retail_7_levels.csv` … 日足レベル（100起点）
- `docs/outputs/retail_7_post_intraday.txt` … X投稿文

## ワークフロー
- `RETAIL-7 Intraday` … 平日 9:00–15:00 JST をカバーして準リアルタイム更新
- `RETAIL-7 Long Charts` … 前日データ確定後に日次レベル再計算
