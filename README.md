# asrlance

## 概要

asrlance は、複数の音声認識モデルの性能を定量的に比較するためのベンチマークツールです。音声ファイルと正解テキストを入力として、各モデルの認識精度（CER）、処理時間、CPU使用率を計測・比較できます。

**対象ユーザー:** 音声認識モデルの選定・評価を行う開発者・研究者

## 特徴

- **複数モデル対応**: MLX Whisper、OpenAI Whisper、ReazonSpeech k2 を統一インターフェースで利用可能
- **定量評価**: CER（文字誤り率）による認識精度の自動計算
- **パフォーマンス計測**: 処理時間・CPU使用率（平均/最大）を記録
- **Apple Silicon 最適化**: MLX Whisper により Mac での高速推論に対応

## クイックスタート

### 前提条件

- Python 3.10 以上
- macOS（Apple Silicon 推奨）または Linux
- 約2GB以上の空きメモリ（モデルにより異なる）

### インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd asrlance

# 仮想環境を作成・有効化
python -m venv .venv
source .venv/bin/activate

# 依存パッケージをインストール
pip install -r requirement.txt
```

**uv を使用する場合:**

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirement.txt
```

### 基本的な使い方

```bash
python benchmark.py <音声ファイル> <正解テキスト> <モデル> [出力ファイル]
```

**例:**

```bash
# MLX Whisper でベンチマーク
python benchmark.py ./audio.wav ./ground_truth.txt mlx mlx_result.txt

# OpenAI Whisper でベンチマーク
python benchmark.py ./audio.wav ./ground_truth.txt whisper whisper_result.txt

# ReazonSpeech でベンチマーク
python benchmark.py ./audio.wav ./ground_truth.txt reazonspeech reazonspeech_result.txt
```

## 対応モデル

| エイリアス | モデル名 | 説明 |
|-----------|---------|------|
| `mlx` | MLX Whisper Large V3 Turbo | Apple Silicon 向け最適化版 |
| `whisper` | OpenAI Whisper Large V3 Turbo | オリジナル OSS 版 |
| `reazonspeech` | ReazonSpeech k2 | 日本語特化モデル (159M params) |

## 出力形式

ベンチマーク実行後、以下の形式で結果がファイルに保存されます:

```
ファイル: /path/to/audio.wav
モデル: MLX Whisper Large V3 Turbo (mlx)
呼称: large-v3-turbo (MLX)
処理時間: 2.40秒
平均CPU: 12.20% 最大CPU: 24.40%
CER（文字誤り率）: 12.42%
認識テキスト:
（認識された文章）
```

## 評価指標

### CER（Character Error Rate / 文字誤り率）

正解テキストと認識結果の文字単位での誤り率を示す指標です。レーベンシュタイン距離を用いて計算されます。

```
CER = (挿入 + 削除 + 置換) / 正解文字数 × 100 [%]
```

- **0%**: 完全一致
- **低いほど良い**: 認識精度が高い
- 空白文字は比較前に除去されます

## プロジェクト構成

```
asrlance/
├── benchmark.py       # ベンチマーク実行スクリプト（メイン）
├── fileRecognizer.py  # 音声認識処理モジュール
├── main.py            # エントリーポイント（未使用）
├── requirement.txt    # 依存パッケージ一覧
├── pyproject.toml     # プロジェクト設定
└── README.md          # このファイル
```

### 主要モジュール

- **benchmark.py**: CLI エントリーポイント。引数解析、CER 計算、結果出力を担当
- **fileRecognizer.py**: 各音声認識モデルの統一インターフェースを提供

## トラブルシューティング

### Mac 固有の注意点

1. **MPS (Metal Performance Shaders)**
   Apple Silicon Mac では PyTorch が MPS を使用する場合があります。CPU で実行したい場合:
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```

2. **メモリ要件**
   ReazonSpeech k2 は約 1.5GB のメモリを使用します。

3. **サンプリングレート**
   入力音声は **16kHz** が推奨されます。異なるレートの場合、自動リサンプリングされます。

### インストールエラー

**ESPnet インストールエラーの場合:**

```bash
pip install espnet --no-build-isolation
```

**torch 関連エラーの場合:**

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**ReazonSpeech インストールエラーの場合:**

```bash
# GitHubから直接インストール
pip install "git+https://github.com/reazon-research/ReazonSpeech#subdirectory=pkg/espnet-asr"

# または、リポジトリをクローンしてインストール
git clone https://github.com/reazon-research/ReazonSpeech
pip install ./ReazonSpeech/pkg/espnet-asr
```
