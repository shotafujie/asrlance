# トラブルシューティング

## Mac 固有の注意点

1. **MPS (Metal Performance Shaders)**
   Apple Silicon Mac では PyTorch が MPS を使用する場合があります。CPU で実行したい場合:
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```

2. **メモリ要件**
   ReazonSpeech k2 は約 1.5GB のメモリを使用します。

3. **サンプリングレート**
   入力音声は **16kHz** が推奨されます。異なるレートの場合、自動リサンプリングされます。

## macOS 更新後の環境トラブル（2026-10 実測）

- **scipy の `dlopen ... _spropack ... __thread_bss` エラー**: scipy 1.15.x の wheel が新しい macOS のローダーに弾かれる。`uv pip install --python <venv>/bin/python scipy==1.14.1` で解消（`.venv` / `~/cohere-venv` / `~/gemma-venv` すべて）。
- **`Could not load libtorchcodec` / `libavutil.56.dylib`**: torchcodec が Homebrew の ffmpeg 版数と合わず読めない。使っていなければ `uv pip uninstall --python .venv/bin/python torchcodec` で回避（kotoba で発生）。
- **gemma-venv で `Segmentation fault`（exit 139）**: torch 2.14.1 + transformers 5.18.0 の組み合わせでモデルロード中に発生。`torch==2.12.1 transformers==5.12.1`（cohere-venv と同じ版）に固定すると動く。

## インストールエラー

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

初回実行時に HuggingFace Hub から `Qwen/Qwen3-ASR-0.6B`（約 1.2GB）を自動ダウンロードします。`qwen17b` エイリアスでは `Qwen/Qwen3-ASR-1.7B`（約 3.5GB）を使用します。MPS/CPU でも動きますが、パフォーマンスは CUDA GPU が最良です。

**granite-4.0-1b-speech インストール:**

`granite` エイリアスは `transformers>=4.52.1` と `peft` が必要です:

```bash
pip install transformers peft soundfile
```

初回実行時に HuggingFace Hub から `ibm-granite/granite-4.0-1b-speech`（約 1GB）を自動ダウンロードします。

なお torchaudio 2.9.x は内部で torchcodec を呼ぶため、torch 2.9 との ABI 不整合で音声読込に失敗することがあります。本ツールでは `soundfile` で wav を直接読み込んで回避しています。

**Gemma 4 E2B (audio) インストール — 専用 venv 必須:**

`gemma` は Google 公式の transformers 実装で動かします。Gemma4 サポートには **transformers>=5.1** が
必要で、本リポジトリの他モデル（transformers 4.57.x 前提）と**同一環境に同居できません**。
そのため gemma だけ別 venv を作ります:

```bash
# 1) 専用 venv を作成し依存を導入
uv venv ~/gemma-venv --python 3.10
uv pip install --python ~/gemma-venv/bin/python \
    "transformers>=5.1" torch torchvision torchaudio soundfile accelerate psutil

# 2) モデル取得（xet はストールしやすいので無効化推奨, 約 10.6GB）
HF_HUB_DISABLE_XET=1 ~/gemma-venv/bin/hf download unsloth/gemma-4-E2B-it-qat-q4_0-unquantized

# 3) この venv の python で benchmark / batch_benchmark を実行
~/gemma-venv/bin/python benchmark.py ./audio.wav ./ground_truth.txt gemma
```

- **モデル**: `unsloth/gemma-4-E2B-it-qat-q4_0-unquantized`（ungated・フル精度・音声対応）。本家
  `google/gemma-4-E2B-it` は gated（HF 認証＋ライセンス同意が必要）なので ungated ミラーを使用。
- **MLX(mlx-vlm) では動かさないこと**: mlx-vlm 0.4.3 の Gemma4 音声経路は実装が未成熟で、音声
  グラウンディングが劣化し CER が約8倍悪化する（自声20文で公式 152%→誤、実体は 20.5%）。詳細は
  「ベンチマーク実績 > Gemma 4 E2B (audio) の検証結果」を参照。
- `torchvision` は transformers の Gemma4 プロセッサ（画像処理を含む統合プロセッサ）の import に必要。
- 検証結果: 日本語逐語 ASR は**実用圏**（平均 CER 20.5%）。英→日 音声翻訳も良好。

**Cohere Transcribe インストール — 専用 venv 必須:**

`cohere` は transformers 公式の `CohereAsrForConditionalGeneration` で動かします。**transformers>=5.4** が
必要で、本リポジトリの他モデル（transformers 4.57.x 前提）と**同一環境に同居できません**。
そのため cohere だけ別 venv を作ります:

```bash
# 1) 専用 venv を作成し依存を導入
uv venv ~/cohere-venv --python 3.10
uv pip install --python ~/cohere-venv/bin/python \
    "transformers>=5.4" torch torchaudio soundfile librosa sentencepiece protobuf accelerate psutil

# 2) gated リポなので HF 認証（モデルページで規約同意のうえ、gated を読める Read トークンで login）
#    https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 で "Agree and access repository"
~/cohere-venv/bin/hf auth login   # 既にログイン済みなら --force で再ログイン

# 3) この venv の python で benchmark / batch_benchmark を実行（xet は無効化推奨）
HF_HUB_DISABLE_XET=1 ~/cohere-venv/bin/python batch_benchmark.py cohere my_voice my_voice/result_cohere.csv
```

- **モデル**: `CohereLabs/cohere-transcribe-03-2026`（Apache 2.0・フル精度 safetensors）。**gated** なので
  規約同意（即時付与）＋ gated 読み取り可能なトークンが必要。fine-grained トークンで `canReadGatedRepos:false`
  だと 403 になるため、Read（classic）トークンか gated 読み取りを有効化したトークンを使う。
- **`language="ja"` を必ず指定**: このモデルは言語の自動判定を持たない。processor に明示しないと言語が振れる。
- ungated ミラーは ONNX（`onnx-community/...`, transformers.js 前提）・GGUF（量子化）のみ。他モデルと同条件の
  フル精度ローカル比較には公式リポを使う。
- 検証結果: 日本語逐語 ASR で**平均 CER 11.81%（Qwen3-ASR に次ぐ2位）**・最速級。詳細は
  「ベンチマーク実績 > Cohere Transcribe の検証結果」を参照。
