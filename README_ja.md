# SkiSense

[English](README.md) | **日本語**

スキー滑走映像から滑走者の姿勢を推定・可視化し、フォームを定量的に評価するコンピュータビジョンツールである。

![SkiSense preview](images/skier.gif)
![SkiSense preview](images/skier2.gif)

---

## 概要

SkiSense は映像中のスキーヤーを検出し、姿勢を推定して、主要な関節角度（膝・股関節・足首・肩の傾き）を基礎スキーの理想範囲と照合してスコア化する。自動採点ではなく、可視化と定量フィードバックのためのツールである。注釈付き動画・最高スコアのフレーム・関節ごとの数値が出力される。

## 主な機能

- **人物検出** — YOLOv8x
- **骨格推定** — 既定は SAM 3D Body（3D MHR-21）、`.env` で YOLO11-Pose（2D COCO-17）に切替可能。[姿勢推定バックエンド](#姿勢推定バックエンド)を参照
- **関節角度評価** — 膝・股関節・足首を 3D、肩の水平傾きを 2D で評価
- **総合スコア** — 算出可能な項目を 0〜100 点で評価
- **複数人トラッキング** — Deep SORT によるフレーム間の ID 一貫性
- **自動ズーム・センタリング** — 胴体中心を画面中央に固定
- **ベストショット抽出** — 最高スコアのフレームを自動保存

## クイックスタート

先に CUDA ビルドに合わせて PyTorch を導入する。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# SAM 3D Body backend を使う場合のみ（docs/pose_backends.md 参照）:
pip install -r requirements-sam3d.txt
```

実行:

```bash
python run.py video.mp4            # 動画を処理（既定 input/video.mp4）
python run.py --fast video.mp4     # フレーム単位の検出/追跡を省略し 1 回の推論に集約
python run.py skier.jpg --image    # 静止画を処理
```

出力は `output/YYYYMMDD_HHMMSS/`（`video_pose.mp4` / `best_shot.jpg` / 入力のコピー）。

> 既定の `sam3d` backend は HuggingFace のゲート重みを使う。初回のみ
> `facebook/sam-3d-body-dinov3` のアクセス申請と `hf auth login` が必要。
> これを避けたい場合や CPU で動かしたい場合は `yolo11` backend を使う。

## 姿勢推定バックエンド

姿勢推定エンジンは `.env` で選択する。

```bash
SKISENSE_POSE_BACKEND=sam3d    # 既定: SAM 3D Body（3D・CUDA 必須）
SKISENSE_POSE_BACKEND=yolo11   # YOLO11-Pose（2D・CPU/MPS/CUDA 可）
```

- **SAM 3D Body** — 3D MHR キーポイント + 人体メッシュ。視点不変の関節角度（足首含む）。CUDA 必須、~1–2 秒/フレーム。
- **YOLO11-Pose** — 2D COCO-17。高速で CPU でも動くが、足先ランドマークがないため足首角度は `N/A`。

比較表・設定・使い分けの詳細は [`docs/pose_backends_ja.md`](docs/pose_backends_ja.md) を参照。

## アーキテクチャ

フレーム単位で 3 ステップを実行する。

1. **検出・トラッキング** — YOLOv8x で検出、Deep SORT で持続 ID を付与
2. **骨格推定** — 選択された backend がキーポイントを返し（SAM 3D Body は 3D + 2D、YOLO11-Pose は 2D）、`pose_analyzer` が関節角度を評価
3. **描画** — `ZoomTracker` のズームを適用し、骨格・bbox を `transform_point_to_zoom()` 経由で描画、情報パネルを重畳

主要モジュール: `config.py` / `pose_topology.py` / `backends/` / `pose_analyzer.py` / `zoom_tracker.py` / `main.py` / `image_processor.py`。

## 詳細

設計思想・処理パイプラインの詳細・評価ロジック・開発で苦労した点・今後の改善予定は
[`docs/project_details_ja.md`](docs/project_details_ja.md) にまとめている。

## License

MIT License. 詳細は [LICENSE](LICENSE) を参照。
