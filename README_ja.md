# SkiSense

[English](README.md) | **日本語**

スキー滑走映像から滑走者の姿勢を推定・可視化し、フォームを定量的に評価するコンピュータビジョンツールである。

![SkiSense preview](images/skier.gif)
![SkiSense preview](images/skier2.gif)

---

## 概要

SkiSense は映像中のスキーヤーを検出し、姿勢を推定して、主要な関節角度（膝・股関節・足首・肩の傾き）を基礎スキーの理想範囲と照合してスコア化する。自動採点ではなく、可視化と定量フィードバックのためのツールである。注釈付き動画・最高スコアのフレーム・関節ごとの数値が出力される。

開発の背景・設計判断・評価ロジック・苦労した点といった詳細は [`docs/project_details_ja.md`](docs/project_details_ja.md) にまとめている。

## 主な機能

- **人物検出** — YOLOv8x
- **骨格推定** — 既定は YOLO26-Pose（2D COCO-17）。`--pose-backend` で YOLO11-Pose または SAM 3D Body（3D MHR-21）に切替可能。[姿勢推定バックエンド](#姿勢推定バックエンド)を参照
- **関節角度評価** — YOLO は膝・股関節を 2D、SAM 3D Body は膝・股関節・足首を 3D で評価。肩の水平傾きは 2D で評価
- **総合スコア** — 算出可能な項目を 0〜100 点で評価
- **複数人トラッキング** — Deep SORT によるフレーム間の ID 一貫性
- **自動ズーム・センタリング** — 胴体中心を画面中央に固定
- **ベストショット抽出** — 最高スコアのフレームを自動保存

## クイックスタート

先に使用デバイス向けの PyTorch を導入する。以下は CUDA の例で、CPU・MPS の
導入手順は PyTorch の公式サイトを参照する。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# SAM 3D Body backend を使う場合のみ（docs/pose_backends.md 参照）:
pip install -r requirements-sam3d.txt
```

実行:

```bash
python run.py video.mp4                          # 動画を処理（既定 input/video.mp4）
python run.py --fast video.mp4                   # フレーム単位の検出/追跡を省略し 1 回の推論に集約
python run.py --pose-backend sam3d video.mp4     # 3D エンジン（CUDA 必須）
python run.py --pose-backend yolo11 video.mp4    # 従来の 2D エンジン
python run.py skier.jpg --image                  # 静止画を処理
```

出力は `output/YYYYMMDD_HHMMSS/`（`video_pose.mp4` / `best_shot.jpg` / 入力のコピー）。

同じ滑走者のフレームで複数の姿勢推定モデルを比較する場合は、
[姿勢推定モデルの比較](docs/pose_comparison.md)を参照。

```bash
python compare.py video.mp4 --frames 30,90
```

比較結果は `output/comparison_YYYYMMDD_HHMMSS/` に、モデルの出力を並べた
`index.html`、フレーム別の画像、`results.json` として保存される。通常処理で使う
backend の設定は変更しない。

> `sam3d` を使う場合は、`facebook/sam-3d-body-dinov3` のアクセス申請と
> `hf auth login` が初回に必要。既定の `yolo26` は公開重みを使い、CPU でも動作する。

## 姿勢推定バックエンド

通常実行では 3 種類の姿勢推定エンジンを選べる。既定の `yolo26` は 2D COCO-17、
`yolo11` は従来の 2D 出力、`sam3d` は 3D の体と足先を扱う。

実行ごとに切り替える場合はコマンドラインで指定する。

```bash
python run.py --pose-backend yolo26    # 2D エンジン（既定）
python run.py --pose-backend yolo11    # 従来の 2D エンジン
python run.py --pose-backend sam3d     # 3D エンジン（CUDA 必須）
```

恒久的な既定値は `.env` で設定する。

```bash
SKISENSE_POSE_BACKEND=yolo26   # 必要に応じて yolo11 または sam3d に変更
```

CLI フラグが `.env` より優先される。有効なエンジンは起動時バナーの `- Pose:` 行に
表示される。

- **SAM 3D Body** — 3D MHR キーポイント + 人体メッシュ。視点不変の関節角度（足首含む）。CUDA 必須、~1–2 秒/フレーム。
- **YOLO26-Pose** — 既定の 2D COCO-17 エンジン。公開重みを使用し、CPU に対応する。
- **YOLO11-Pose** — 2D COCO-17。高速で CPU でも動くが、足先ランドマークがないため足首角度は `N/A`。

YOLO26 も足先ランドマークがないため、足首角度は `N/A` となる。2D と 3D の
スコアは**直接比較できない**。角度の測り方が異なり、YOLO は 5 項目、
`sam3d` は 7 項目の平均を取る。比較表・設定・使い分けの詳細は
[`docs/pose_backends_ja.md`](docs/pose_backends_ja.md) を参照。

## アーキテクチャ

フレーム単位で 3 ステップを実行する。

1. **検出・トラッキング** — YOLOv8x で検出、Deep SORT で持続 ID を付与
2. **骨格推定** — 選択された backend がキーポイントを返し（SAM 3D Body は 3D + 2D、YOLO11/YOLO26 は 2D）、`pose_analyzer` が関節角度を評価
3. **描画** — `ZoomTracker` のズームを適用し、骨格・bbox を `transform_point_to_zoom()` 経由で描画、情報パネルを重畳

主要モジュール: `config.py` / `pose_topology.py` / `backends/` / `pose_analyzer.py` / `zoom_tracker.py` / `main.py` / `image_processor.py`。

## License

MIT License. 詳細は [LICENSE](LICENSE) を参照。
