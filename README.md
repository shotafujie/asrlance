# asrlance

## 概要

asrlance は、複数の音声認識モデルの性能を定量的に比較するためのベンチマークツールです。音声ファイルと正解テキストを入力として、各モデルの認識精度（CER）、処理時間、CPU使用率を計測・比較できます。

**対象ユーザー:** 音声認識モデルの選定・評価を行う開発者・研究者

## 特徴

- **複数モデル対応**: MLX Whisper、OpenAI Whisper、ReazonSpeech k2、FunASR SenseVoiceSmall、Moonshine (Japanese Base)、Kotoba-Whisper v2.0、rinna/nue-asr、Qwen3-ASR 0.6B、IBM granite-4.0-1b-speech を統一インターフェースで利用可能
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

# FunASR SenseVoiceSmall でベンチマーク
python benchmark.py ./audio.wav ./ground_truth.txt funasr funasr_result.txt

# Moonshine (Japanese Base) でベンチマーク
python benchmark.py ./audio.wav ./ground_truth.txt moonshine moonshine_result.txt

# Kotoba-Whisper v2.0 でベンチマーク
python benchmark.py ./audio.wav ./ground_truth.txt kotoba kotoba_result.txt

# rinna/nue-asr でベンチマーク
python benchmark.py ./audio.wav ./ground_truth.txt nue nue_result.txt

# Qwen3-ASR 0.6B でベンチマーク
python benchmark.py ./audio.wav ./ground_truth.txt qwen qwen_result.txt

# IBM granite-4.0-1b-speech でベンチマーク
python benchmark.py ./audio.wav ./ground_truth.txt granite granite_result.txt
```

## 対応モデル

| エイリアス | モデル名 | 説明 |
|-----------|---------|------|
| `mlx` | MLX Whisper Large V3 Turbo | Apple Silicon 向け最適化版 |
| `whisper` | OpenAI Whisper Large V3 Turbo | オリジナル OSS 版 |
| `reazonspeech` | ReazonSpeech k2 | 日本語特化モデル (159M params) |
| `funasr` | FunASR SenseVoiceSmall | 多言語対応モデル (234M params, 日本語含む) |
| `moonshine` | Moonshine Japanese Base | Flavors of Moonshine の日本語特化モデル (58M params) |
| `kotoba` | Kotoba-Whisper v2.0 | ReazonSpeech で蒸留した日本語特化 Whisper (large-v3 より 6.3x 高速) |
| `nue` | rinna/nue-asr | HuBERT + GPT-NeoX ハイブリッド, ReazonSpeech v1 (19,000時間) で学習 |
| `qwen` | Qwen3-ASR 0.6B | Alibaba Qwen の多言語 ASR (52言語対応, 日本語含む) |
| `granite` | IBM granite-4.0-1b-speech | IBM の多言語 ASR (1B, 日本語含む6言語, OpenASR leaderboard 平均 WER 5.52) |

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
- **batch_benchmark.py**: 1モデルを1回だけロードして複数音声を一括評価するバッチ版（`kotoba` / `nue` / `qwen` 対応）

## ベンチマーク実績

### 自声 20文での検証結果（2026-04）

16kHz / 1ch / 3.5〜10.3秒の日本語音声 20文（短文・技術用語・数字・カタカナ固有名詞・疑問文・滑舌チェック等バランス配分）で、セッションで追加したモデルを比較。

| モデル | 平均 CER | 平均推論時間 | モデルロード時間 | 実行環境 |
|---|---|---|---|---|
| kotoba-whisper v2.0 | 19.65% | 0.76s | 69s | MPS |
| rinna/nue-asr (yky-h) | 19.17% | 1.17s | 11s | MPS |
| IBM granite-4.0-1b-speech | 17.25% | 1.01s | 8s | MPS |
| **Qwen3-ASR 0.6B** | **8.44%** | 0.78s | 127s | MPS |

**主な観察**
- Qwen3-ASR は依然最良。英単語/アルファベット（Python, GitHub, OpenAI, ChatGPT, iPhone 等）をそのまま大文字混じりで出力でき、他モデルのカタカナ化（パイソン、ギットハブ）や小文字化（python, github）による CER 増加を回避している。
- granite-4.0-1b は 4モデル中最速でロード（8s）。英単語は残すが全小文字化（python, gpu, json, pull request）する傾向があり、ground truth が大文字のケースでペナルティを受けている。アンソロピックなど固有名詞は他モデル同様崩れる。
- 4モデル共通で苦手なのは早口言葉（「赤パジャマ青パジャマ黄パジャマ」）、難読地名（「東京特許許可局」）。
- CER は表記ゆれ（漢数字 vs 算用数字、大文字 vs 小文字、カタカナ vs アルファベット）を一律ペナルティとして数える点に注意。

※ 録音データ・結果CSV（`my_voice/`）はリポジトリに含めず `.gitignore` で除外。

## 今後の追加候補モデル（2026-04 調査）

以下は調査済みで未実装。優先度順。

### S ランク（推奨）

1. ~~**IBM granite-4.0-1b-speech**~~ — 検証済み（上記テーブル）。自声20文では 17.25%。英単語の大小文字扱いが特徴的で、OpenASR の公称 WER とは別系統の CER ペナルティが効く。
2. **Voxtral-Mini-3B-2507** (Mistral) — 3B / Apache 2.0 / Audio-LLM（ASR + Q&A 同時）。Neosophie ベンチで WER 24%。
3. **Cohere-transcribe-03-2026** — 2B / 現時点 SOTA（OpenASR WER 5.42）。日本語実測公開なしのため自ら測る価値高。ライセンス条項要確認。

### A ランク

- **Phi-4-multimodal-instruct** (MS, 5.6B, MIT) — 日本語含む8言語学習。
- **Qwen3-Omni-30B-A3B** (Alibaba, MoE active 3B) — Audio/Video/Text フルマルチモーダル。ローカル実行は重い。
- **Meta Omnilingual ASR** (最大 7B) — 1,690 言語対応、fairseq2 必須。
- **litagin/anime-whisper** (Kotoba-Whisper 派生) — アニメ調音声 5,300時間で fine-tune。ドメイン特化事例。
- **sbintuitions/nest-ja-0.1b** — 日本語 35K 時間 SSL、ASR fine-tune のベースとして。

### B ランク（紹介のみ）

Meta SeamlessM4T-v2-Large（CC-BY-NC）、Meta MMS、ESPnet OWSM v4、Gemma 3n E2B/E4B、Fun-ASR-Nano-2512、既存の Japanese HuBERT / wav2vec2 系。

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

**Moonshine 初回実行時:**

`moonshine` エイリアスで初回実行する際、日本語モデル（約 58M params）を自動ダウンロードします。ダウンロードはキャッシュ（`~/Library/Caches/moonshine_voice/` 等）に保存されます。

**FunASR 初回実行時:**

`funasr` エイリアスで初回実行する際、ModelScope から SenseVoiceSmall（約 234M params）と fsmn-vad モデルを自動ダウンロードします。

**Kotoba-Whisper 初回実行時:**

`kotoba` エイリアスで初回実行する際、HuggingFace Hub から `kotoba-tech/kotoba-whisper-v2.0`（約 1.5GB）を自動ダウンロードします。`transformers` と `accelerate` が必要です。

**rinna/nue-asr インストール:**

`nue` エイリアスは GitHub からの直接インストールが必要です:

```bash
pip install git+https://github.com/rinnakk/nue-asr.git
```

**Qwen3-ASR インストール:**

`qwen` エイリアスは `qwen-asr` パッケージが必要です:

```bash
pip install qwen-asr
```

初回実行時に HuggingFace Hub から `Qwen/Qwen3-ASR-0.6B`（約 1.2GB）を自動ダウンロードします。MPS/CPU でも動きますが、パフォーマンスは CUDA GPU が最良です。

**granite-4.0-1b-speech インストール:**

`granite` エイリアスは `transformers>=4.52.1` と `peft` が必要です:

```bash
pip install transformers peft soundfile
```

初回実行時に HuggingFace Hub から `ibm-granite/granite-4.0-1b-speech`（約 1GB）を自動ダウンロードします。

なお torchaudio 2.9.x は内部で torchcodec を呼ぶため、torch 2.9 との ABI 不整合で音声読込に失敗することがあります。本ツールでは `soundfile` で wav を直接読み込んで回避しています。
