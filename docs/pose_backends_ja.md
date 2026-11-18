# ポーズ推定バックエンド

[English](pose_backends.md) | **日本語**

通常実行では姿勢推定エンジンを 3 種類から選べる。既定の `yolo26` は公開重みを
使い、CPU・MPS・CUDA で実行できる。`yolo11` は従来の 2D 出力を再現し、
`sam3d` は 3D の体と足先を推定する。

**実行ごとの切り替え — CLI フラグ**（最優先）:

```bash
python run.py --pose-backend yolo26 video.mp4     # 既定の 2D エンジン
python run.py --pose-backend yolo11 video.mp4     # 従来の 2D エンジン
python run.py --pose-backend sam3d video.mp4      # 3D エンジン、CUDA 必須
python run.py --pose-backend yolo26 skier.jpg --image
```

**恒久的な既定値 — `.env`:**

```bash
SKISENSE_POSE_BACKEND=yolo26   # 必要に応じて yolo11 または sam3d に変更
```

優先順位は `--pose-backend` → `SKISENSE_POSE_BACKEND` → `yolo26` である。
選択されたエンジンは起動時バナーの `- Pose:` 行に出力されるため、長時間の
処理を始める前に有効なモデルを確認できる。

人物検出（YOLOv8x）・トラッキング（Deep SORT）・ズーム・スコアリングの
パイプラインは共通で、姿勢推定部だけが差し替わる。各バックエンドは自身の
トポロジ（`pose_topology.py`）を宣言し、`analyze_ski_pose` と描画ヘルパーが
それに自動追従する。`--fast` / `--high` / `--target-mode` は 3 種類のエンジン
でも動作する。

## 比較表

| 項目 | YOLO26-Pose | YOLO11-Pose | SAM 3D Body |
|---|---|---|---|
| バックエンド名 | `yolo26`（既定） | `yolo11` | `sam3d` |
| トポロジ | COCO-17 | COCO-17 | MHR-21（体 + 足先 + 手首） |
| 次元 | 2D 画像座標 | 2D 画像座標 | 3D（カメラ座標）+ 2D 投影 |
| 関節角度 | 膝・股関節を 2D で算出 | 膝・股関節を 2D で算出 | 膝・股関節・足首を 3D で算出 |
| 足首角度 | N/A | N/A | 取得可 |
| 肩の傾き | 2D 画像平面 | 2D 画像平面 | 2D 画像平面 |
| デバイス | CPU / MPS / CUDA | CPU / MPS / CUDA | **CUDA 必須** |
| 重み入手 | Ultralytics 自動 DL | Ultralytics 自動 DL | Hugging Face ゲート（要申請 + `hf auth login`） |
| ライセンス | Ultralytics AGPL-3.0 | Ultralytics AGPL-3.0 | SAM License |
| メッシュ出力 | なし | なし | あり（`pred_vertices`） |

既定値は導入の容易さと少数のサンプルで観察した結果を踏まえた選択であり、
スキー姿勢推定の精度順位を検証したものではない。手元の映像では
[`compare.py`](pose_comparison.md) で出力を確かめられる。

## YOLO26-Pose（`yolo26`）

既定の通常実行モデルは `yolo26x-pose.pt` を使い、COCO-17 の体の点を推定する。
別の YOLO26-Pose 重みを使う場合は `SKISENSE_YOLO26_POSE_MODEL` で指定する。
重みは初回に Ultralytics から自動取得される。YOLO11 と同じ 2D 採点経路を
使うため、視点の影響を受け、足首角度は算出できない。`ultralytics>=8.4.215`
が必要である。

## SAM 3D Body（`sam3d`）

Meta が 2025-11-19 に公開した単一画像 3D 人体メッシュ復元モデル。MHR
（Momentum Human Rig）パラメトリックモデルで、視点に依存しない真の関節角度が
得られる。SkiSense は MHR70 キーポイントの先頭 63（体・足先・手首）を使用する。

- **強み**: 斜め視点でも膝・股関節・足首角度が歪まない。内脚の遮蔽に強い。
  足首の前圧（knee→ankle→toe）まで評価でき、人体メッシュも取得できる。
- **弱み**: CUDA 必須。1 フレーム数秒と重く、動画はオフライン処理前提。
  重みがゲート公開のため初回に HuggingFace のアクセス申請が要る。
- **設定**:
  - `SKISENSE_SAM3D_HF_REPO`: `facebook/sam-3d-body-dinov3`（既定, 840M）/
    `facebook/sam-3d-body-vith`（631M, 軽量・低 VRAM）
  - `SKISENSE_SAM3D_USE_HAND_REFINE`: 手指デコーダ。スキー採点では不要のため
    既定 `false`（VRAM・レイテンシをほぼ半減）
- **セットアップ**: [`../notes/sam3d_setup.md`](../notes/sam3d_setup.md) を参照
  （detectron2 / MoGe は不要）。

## YOLO11-Pose（`yolo11`）

Ultralytics の 2D 姿勢推定モデルであり、SAM 3D Body 移行前に SkiSense が使って
いたエンジンでもある。COCO-17 キーポイントを ROI 単位で推定する。旧パイプライン
の結果を再現できるようにし、また CUDA 非搭載機でも動かせるようにするため、
非推奨扱いにせず意図的に維持している。

- **強み**: CPU でも動く。1 フレーム数 ms と高速で、リアルタイム寄りの確認や
  CUDA 非搭載機での利用に向く。重みは自動ダウンロード。
- **弱み**: 2D のため斜め視点で関節角度に投影歪みが出る。COCO-17 に足先
  ランドマークがないため**足首角度は N/A**（スコアから除外）。
- **設定**:
  - `SKISENSE_YOLO_POSE_MODEL`: `yolo11x-pose.pt`（既定）他 n/s/m/l/x
  - `SKISENSE_YOLO_POSE_CONFIDENCE`: キーポイント信頼度しきい値（既定 0.25）
  - `SKISENSE_CLAHE_ENABLED`: ROI への CLAHE 適用（既定 false）
  - `SKISENSE_FLIP_TTA_ENABLED`: 水平反転 TTA（既定 false, 推論コスト 2 倍）

## 使い分けの指針

- **モデルのアクセス申請なしで始める** → `yolo26`。
- **3D の体と足先を見る** → CUDA とゲート重みを用意して `sam3d` を使う。
  評価対象の映像でも推定結果を確認する。
- **移行前の出力を再現したい場合** → `yolo11`。SAM 3D Body 移行前と同一の
  エンジン・同一の COCO-17 トポロジである。

## バックエンドをまたいだスコアは比較できない

バックエンドを切り替えると、精度だけでなく測定値そのものが変わる。したがって
`sam3d` のスコアと YOLO のスコアを直接比べてはならない。

- 膝・股関節の角度は、`sam3d` ではカメラ空間の 3D ベクトルから、`yolo11` では
  `yolo11` と `yolo26` では画像平面に投影された座標から算出する。同じ姿勢でも斜め視点では度数が異なる。
- 総合スコアは測定できた関節の平均である。YOLO は左右の足首角度を測定できない
  ため、7 項目ではなく 5 項目の平均になる。

シーズンを通したクリップ群を見比べるときなど、比較を行う場面ではバックエンドを
固定すること。

## 既知の制約

- YOLO11 と YOLO26 は COCO-17 トポロジのため足首角度を出せない（info panel は `N/A`）。
- `sam3d` は CUDA 必須。`SKISENSE_DEVICE` が CUDA に解決されない環境で `sam3d` を
  選ぶと、バックエンド構築時に `RuntimeError` を送出する。エラーメッセージは
  `--pose-backend yolo26` への切り替えを案内する。
- `sam3d` は関節ごとの信頼度を公開しないため、使用する関節の visibility は一律
  `1.0` で埋めている。その結果 `SKISENSE_POSE_VISIBILITY_THRESHOLD` および
  `SKISENSE_POSE_VISIBILITY_THRESHOLD_LEGS` は `sam3d` では効果を持たず、遮蔽された
  関節もモデルの推定値のまま採点される。YOLO11 と YOLO26 では両設定とも有効である。

RTMW3D は [`compare.py`](pose_comparison.md) で出力を見比べられる。
3D 座標系と採点への適用は検証中のため、通常実行の `run.py` では選べない。
