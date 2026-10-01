# asrlance

## 概要

asrlance は、複数の音声認識モデルの性能を定量的に比較するためのベンチマークツールです。音声ファイルと正解テキストを入力として、各モデルの認識精度（CER）、処理時間、CPU使用率を計測・比較できます。

**対象ユーザー:** 音声認識モデルの選定・評価を行う開発者・研究者

## 特徴

- **複数モデル対応**: MLX Whisper、OpenAI Whisper、ReazonSpeech k2、FunASR SenseVoiceSmall、Moonshine (Japanese Base)、Kotoba-Whisper v2.0、rinna/nue-asr、Qwen3-ASR 0.6B/1.7B、IBM granite-4.0-1b-speech、Gemma 4 E2B (audio)、Cohere Transcribe (cohere-transcribe-03-2026) を統一インターフェースで利用可能
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

# Gemma 4 E2B (audio) でベンチマーク
python benchmark.py ./audio.wav ./ground_truth.txt gemma gemma_result.txt

# Cohere Transcribe (cohere-transcribe-03-2026) でベンチマーク（専用 venv で実行・下記参照）
python benchmark.py ./audio.wav ./ground_truth.txt cohere cohere_result.txt
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
| `qwen17b` | Qwen3-ASR 1.7B | Qwen3-ASR の 1.7B 版。0.6B と同一インターフェース (`qwen-asr` パッケージ) |
| `granite` | IBM granite-4.0-1b-speech | IBM の多言語 ASR (1B, 日本語含む6言語, OpenASR leaderboard 平均 WER 5.52) |
| `gemma` | Gemma 4 E2B (audio) | Google のネイティブ音声入力マルチモーダル LLM (E2B = effective 2B)。**Google 公式の transformers 実装**で実行（transformers>=5.1 必須・専用 venv）|
| `cohere` | Cohere Transcribe (cohere-transcribe-03-2026) | Cohere Labs の専用 ASR (2B, Fast-Conformer encoder-decoder, Apache 2.0, 14言語)。OpenASR(英語) 平均 WER 5.42 で公開当時 SOTA。**transformers 公式実装**で実行（transformers>=5.4 必須・専用 venv、gated 要 HF 認証）|

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
- **batch_benchmark.py**: 1モデルを1回だけロードして複数音声を一括評価するバッチ版（`mlx` / `kotoba` / `nue` / `qwen` / `qwen17b` / `granite` / `gemma` / `cohere` 対応）

## ベンチマーク実績

### 自声 20文での検証結果（2026-04）

16kHz / 1ch / 3.5〜10.3秒の日本語音声 20文（短文・技術用語・数字・カタカナ固有名詞・疑問文・滑舌チェック等バランス配分）で、セッションで追加したモデルを比較。

| モデル | 平均 CER | 平均推論時間 | モデルロード時間 | 実行環境 |
|---|---|---|---|---|
| Gemma 4 E2B (audio) ※1 | 20.52% | 0.71s | 10.1s | transformers (MPS) |
| kotoba-whisper v2.0 | 19.65% | 0.76s | 3.5s | MPS |
| rinna/nue-asr (yky-h) | 19.17% | 1.17s | 11s | MPS |
| IBM granite-4.0-1b-speech | 17.25% | 1.01s | 8s | MPS |
| Cohere Transcribe (cohere-transcribe-03-2026) ※4 | 11.81% | 0.25s | 6.6s | transformers (MPS) |
| Whisper large-v3-turbo (MLX) ※2 | 11.80% | 0.26s | 0.8s | MLX |
| Qwen3-ASR 0.6B | 8.44% | 0.78s | 4.9s | MPS |
| **Qwen3-ASR 1.7B** ※5 | **6.48%** | 0.55s | 6.8s | MPS |

> 2026-10 に別の30文を追加して50文で測り直した結果、平均 CER は Qwen3-ASR 1.7B（8.67%）が1位を維持、Cohere Transcribe（10.27%）が2位。30文単独では Cohere が1位。下記「自声 50文での統合評価」「自声 30文での追加検証」を参照。

> **推奨（2026-07 更新）**: 最高精度が要るなら **Qwen3-ASR 1.7B**（CER 6.48%・中央値 0.00%）。Qwen3-ASR 1.7B は 0.55s/文と十分速く、**精度と速度の両取りでも実運用の既定候補**になった。純粋な速度最優先（0.25s/文・実時間の約26倍速）なら **Cohere Transcribe**（精度 11.81% は 1.7B/0.6B/Whisper turbo に次ぐ4位、Apache 2.0・後処理不要・英単語の大小文字保持）も引き続き有力。

※1 Gemma 4 E2B は 2026-06 追加。Google 公式 transformers 実装で測定（中央値 CER は 14.09%）。MLX(mlx-vlm) 経路は音声品質が約8倍悪化するため不採用。詳細は下記「Gemma 4 E2B (audio) の検証結果」。

※2 Whisper large-v3-turbo（`mlx-community/whisper-large-v3-turbo`）は専用 ASR のベースライン。2026-06 に同条件で実測（中央値 CER 9.09%、CER≤30% 19/20、メモリ ~1.5GB）。誤差の多くは表記揺れ（算用数字↔漢数字）で意味は正しい。純粋な日本語転写では本表で最良の Qwen3-ASR に次ぐ精度かつ最速・最軽量。MLX でも正常動作する（mlx-vlm の Gemma4 音声実装のみが未成熟だった）。

※3 ロード時間はウォーム（モデルキャッシュ済）での値。旧 2026-04 表の Qwen3-ASR **127s**・kotoba **69s** は初回ダウンロード込みの計測だったため、2026-06 にウォーム再計測して **4.9s・3.5s** に訂正。nue(11s)/granite(8s)/Gemma(10.1s)/Whisper(0.8s) は妥当なウォーム値。

※4 Cohere Transcribe は 2026-06 追加。transformers 公式実装 (`CohereAsrForConditionalGeneration`, MPS) で測定（中央値 CER **8.57%**、CER≤30% **19/20**、完璧(0%) 6/20）。英語 OpenASR では SOTA(WER 5.42) だが、自声20文では **Qwen3-ASR(8.44%) に次ぐ2位**で Whisper turbo(11.80%) とほぼ同率、かつ **ローカル勢で最速（0.25s/文・実時間の約26倍速・メモリ ~4GB・CPU平均46%）**。失点の大半は漢字↔仮名の表記揺れ（今日→きょう、明日→あした、家族→かぞ）と英小文字化（api / json）。公式リポは gated（Apache 2.0, 規約同意で即時アクセス付与）で要 HF 認証。詳細は下記「Cohere Transcribe の検証結果」。

※5 Qwen3-ASR 1.7B は 2026-07 追加。`qwen-asr` パッケージ（0.6B と同一インターフェース, MPS, fp16）で測定。**平均 CER 6.48%・中央値 0.00%（完璧 11/20・CER≤30% 19/20）で本表の最高精度を更新**し、推論も 0.55s/文と 0.6B(0.78s) より速かった。詳細は下記「Qwen3-ASR 1.7B の検証結果」。

**主な観察**
- Qwen3-ASR は依然最良。英単語/アルファベット（Python, GitHub, OpenAI, ChatGPT, iPhone 等）をそのまま大文字混じりで出力でき、他モデルのカタカナ化（パイソン、ギットハブ）や小文字化（python, github）による CER 増加を回避している。
- granite-4.0-1b は 4モデル中最速でロード（8s）。英単語は残すが全小文字化（python, gpu, json, pull request）する傾向があり、ground truth が大文字のケースでペナルティを受けている。アンソロピックなど固有名詞は他モデル同様崩れる。
- **Cohere Transcribe** は英語 OpenASR の SOTA(WER 5.42) を引っさげての登場だが、日本語自声20文では Qwen3-ASR に届かず2位（11.81%）。ただし英単語の大小文字保持（GPU, GitHub, iPhone）は Qwen 同様得意で、**他モデルが全滅した早口言葉「赤パジャマ青パジャマ黄パジャマ」を唯一 0% で正解**し、難読地名「東京特許許可局」も「東京特許著書」(22%) と相対的に健闘。失点源は漢字を仮名に開く癖（今日→きょう、明日→あした、家族→かぞ）と英小文字化（api, json）で、いずれも意味は正しい表記揺れ。
- 早口言葉（「赤パジャマ青パジャマ黄パジャマ」）・難読地名（「東京特許許可局」）は kotoba/nue/qwen/granite では共通して苦手だったが、Cohere は早口言葉を 0% で攻略した（ただし「東京特許許可局」は全モデル未到達）。
- CER は表記ゆれ（漢数字 vs 算用数字、大文字 vs 小文字、カタカナ vs アルファベット、漢字 vs 仮名）を一律ペナルティとして数える点に注意。

※ 録音データ・結果CSV（`my_voice/`）はリポジトリに含めず `.gitignore` で除外。

### 自声 50文（20文＋30文）での統合評価（2026-10）

20文（`my_voice/`）と30文（`my_voice_30/`）を `my_voice_50/` にまとめ、**8モデルすべてを同一環境・同一条件で測り直した**（旧20文の結果を流用していない。20文だけの平均は旧 README の値と一致し、環境差による変動がないことを確認済み）。

| モデル | 平均 CER | 中央値 | 完全一致 | CER≤30% | 句読点除外の平均 | 平均推論時間 | ロード |
|---|---|---|---|---|---|---|---|
| **Qwen3-ASR 1.7B** | **8.67%** | 7.41% | 11/50 | 48/50 | 6.63% | 1.64s | 13.4s |
| Cohere Transcribe | 10.27% | 7.29% | 6/50 | 49/50 | 7.83% | 0.38s | 8.6s |
| Qwen3-ASR 0.6B | 11.02% | 8.71% | 10/50 | 47/50 | 8.92% | 1.14s | 8.2s |
| Whisper large-v3-turbo (MLX) | 12.38% | 10.79% | 0/50 | 49/50 | 6.77% | **0.28s** | 1.4s |
| IBM granite-4.0-1b-speech | 13.31% | 9.76% | 3/50 | 46/50 | 9.75% | 1.53s | 8.4s |
| rinna/nue-asr (yky-h) | 13.85% | 7.90% | 3/50 | 44/50 | 10.86% | 1.40s | 6.8s |
| kotoba-whisper v2.0 | 17.12% | 13.18% | 0/50 | 46/50 | 11.67% | 0.77s | 3.9s |
| Gemma 4 E2B (audio) ※ | 28.84% | 17.88% | 0/50 | 43/50 | 25.08% | 1.70s | 13.1s |

※ 30文側 utt_027 の繰り返し暴走（CER 480.8%）を含む。除くと平均 約 19.2%。

内訳（平均 CER、20文 / 30文）: Qwen 1.7B 6.48 / 10.13、Cohere 11.81 / 9.25、Qwen 0.6B 8.44 / 12.75、Whisper turbo 11.80 / 12.78、granite 17.25 / 10.69、nue 19.17 / 10.30、kotoba 19.65 / 15.44、Gemma 20.52 / 34.40。

**観察**
- 50文では **Qwen3-ASR 1.7B が最良（8.67%）で、Cohere が2位（10.27%）**。ただし文の種類で逆転する（20文の短文・技術用語は Qwen 1.7B、30文の長い会話調は Cohere）。差は 1.6pt で、中央値は Cohere（7.29%）の方が低い。
- **Qwen 1.7B は平均では最良だが、中央値では Cohere とほぼ同じ**。完全一致（11/50）は Qwen 系だけが多く、英単語・固有名詞を含む短文で強い。
- **句読点を除くと Whisper turbo (MLX) が 6.77% で Qwen 1.7B（6.63%）に肉薄**する。Whisper は句読点を多めに付ける・付けない癖が CER に響いており、句読点を気にしない用途では実質互角。速度は最速（0.28s）。
- nue・granite は短文（20文）が苦手で、30文では 10% 台に改善する。モデルによって得意な文種が違うため、評価文の構成で順位が動く。
- 速度×精度の実用面では、精度重視は Qwen3-ASR 1.7B、速度重視は Whisper turbo (MLX) か Cohere（0.3〜0.4s/文）。

### 自声 30文での追加検証（2026-10）

別の30文（日常会話調の長め・フィラー入り、6.89〜11.60秒、16kHz/mono/PCM16）で、上記 8 モデルを同条件で再測定。正解テキスト（`my_voice_30/utt_NNN.txt`）は録音者本人が発話との一致を確認済み。

| モデル | 平均 CER | 中央値 CER | CER≤30% | 平均推論時間 | モデルロード | 実行環境 |
|---|---|---|---|---|---|---|
| **Cohere Transcribe** | **9.25%** | 7.29% | 30/30 | 0.46s | 102s ※a | transformers (MPS) |
| Qwen3-ASR 1.7B | 10.13% | 8.33% | 29/30 | 1.35s | 10.4s | MPS |
| rinna/nue-asr (yky-h) | 10.30% | 8.31% | 29/30 | 2.77s | 7.6s | MPS |
| IBM granite-4.0-1b-speech | 10.69% | 10.00% | 30/30 | 3.49s | 9.5s | MPS |
| Qwen3-ASR 0.6B | 12.75% | 11.27% | 29/30 | 1.06s | 9.1s | MPS |
| Whisper large-v3-turbo (MLX) | 12.78% | 12.00% | 30/30 | **0.41s** | 1.1s | MLX |
| kotoba-whisper v2.0 | 15.44% | 14.13% | 30/30 | 0.90s | 4.1s | MPS |
| Gemma 4 E2B (audio) ※b | 34.40% | 18.84% | 27/30 | 2.24s | 16.5s | transformers (MPS) |

※a 初回ダウンロード込みの値（ウォームなら数秒〜10秒台）。他モデルはキャッシュ済み。
※b utt_027 で同じ文字列を繰り返す暴走（CER 480.77%）が出て平均を押し上げている。除くと平均 約19.0%。

**観察**
- **20文とは順位が変わった**: 20文では Qwen3-ASR 1.7B（6.48%）が最良だったが、30文では Cohere Transcribe が1位（9.25%）、Qwen3-ASR 1.7B は2位（10.13%）。上位4モデル（9.25〜10.69%）の差は 1.5pt 程度で、30文では順位を確定できる差ではない。「Qwen 1.7B が常に最高精度」とは言えず、**文の種類（今回は長めの会話調・フィラー入り）で順位が入れ替わる**。
- **句読点が CER を大きく左右する**: 本ツールの CER は句読点（、。？）も誤りとして数えるため、今回は完全一致（0%）が全モデルで0文。参考に句読点を除いて再計算すると下表のとおりで、平均が約4pt 下がり、完全一致も出る。**順位も入れ替わる**（nue が2位、Whisper turbo が4位に上がる）ので、モデル間の優劣は句読点の扱い次第で変わる。

  | モデル | 平均 CER（句読点除外） | 完全一致 |
  |---|---|---|
  | Cohere Transcribe | 5.33% | 6/30 |
  | rinna/nue-asr | 5.73% | 11/30 |
  | Whisper large-v3-turbo (MLX) | 6.48% | 9/30 |
  | IBM granite-4.0-1b-speech | 6.74% | 7/30 |
  | Qwen3-ASR 1.7B | 6.94% | 6/30 |
  | kotoba-whisper v2.0 | 9.40% | 4/30 |
  | Qwen3-ASR 0.6B | 9.55% | 3/30 |

  （Gemma は未再計算。）上表の主表（句読点込み）が `batch_benchmark.py` の出力そのもの。
- 全モデル共通で崩れやすかったのは utt_027（「説明が長くなると途中で追えなくなる…」）と utt_009・utt_028・utt_023 で、聞き取りの難しさは文に依存する。誤りの多くは同音・近音の置換（用事→用紙、何がよくて→何がよとて 等）。
- 速度は Whisper turbo (MLX) 0.41s と Cohere 0.46s が突出して速い。精度と速度の両立では Cohere が最もバランスが良い。
- Gemma 4 E2B は平均を暴走が大きく悪化させる。実運用では生成長の上限（`max_new_tokens`）や繰り返し抑制が必要。

**再現手順**

```bash
# my_voice_30/ に utt_NNN.wav と utt_NNN.txt を置く
HF_HUB_DISABLE_XET=1 .venv/bin/python batch_benchmark.py qwen17b my_voice_30 my_voice_30/result_qwen17b.csv
# mlx / qwen / granite / kotoba / nue も同様（.venv）
HF_HUB_DISABLE_XET=1 ~/cohere-venv/bin/python batch_benchmark.py cohere my_voice_30 my_voice_30/result_cohere.csv
HF_HUB_DISABLE_XET=1 ~/gemma-venv/bin/python batch_benchmark.py gemma my_voice_30 my_voice_30/result_gemma.csv
```

※ 録音データ・結果CSV（`my_voice_30/*.wav` / `result_*.csv`）は `.gitignore` で除外。

### Gemma 4 E2B (audio) の検証結果（2026-06）

同じ自声20文を、Google の Gemma 4 E2B（ネイティブ音声入力マルチモーダル LLM）で測定。
**実行系の違いで結果が激変した**ため、両者を併記する（重要な教訓）。

| 実行系 | 平均 CER | 中央値 CER | ≤30% CER の文数 |
|---|---|---|---|
| **transformers 公式実装（正）** | **20.52%** | **14.09%** | **16/20** |
| MLX (mlx-vlm 0.4.3, 8bit) — 不採用 | 151.97% | 92.52% | 2/20 |

**結論: Gemma 4 E2B は日本語の逐語 ASR で実用圏。** 平均 CER 20.5%/中央値 14% は、既存の
kotoba-whisper(19.65%)・nue(19.17%)・granite(17.25%) と**ほぼ互角**（Qwen3-ASR 8.44% には及ばない）。
残差の多くは表記揺れ（三時↔3時、二〇二五↔2025）で意味は概ね正しく、「機械学習のモデルをGPUで…」
「最新のiPhoneは…」等は CER 4〜5% とほぼ完璧。

**当初 MLX で誤った結論に至った経緯（教訓）**
- 最初 MLX(mlx-vlm 0.4.3) で動かし、平均 CER 152%/中央値 92% という壊滅的な数値が出て
  「モデルが逐語 ASR に不向き」と早合点した。
- だが clean な英語 LibriSpeech で A/B したところ、**同じ音声で MLX は WER 69.8%、Google 公式の
  transformers 実装は WER 7.6%** と約9倍差。日本語20文も公式実装で測り直すと上表の通り回復した。
- 原因は **mlx-vlm 0.4.3 の Gemma4 音声経路の実装** にあり、音声グラウンディングが劣化していた
  （mel 特徴に NaN は無いので「壊れて」はいないが、PLE/音声トークン整合あたりが微妙にズレ、
  モデルに歪んだ音が届いて「冒頭の要旨だけ掴んで後は作文」する挙動になっていた）。**モデルの実力ではなかった。**

**コツ/ハマりどころ**
- **MLX(mlx-vlm) では動かさないこと。** Google 公式の transformers `Gemma4ForConditionalGeneration` を使う。
- Gemma4 サポートには **transformers>=5.1** が必要。本リポジトリの他モデル(4.57.x)とは非互換なので
  **gemma だけ専用 venv** を作る（トラブルシューティング参照）。
- モデルは ungated・フル精度・音声対応の `unsloth/gemma-4-E2B-it-qat-q4_0-unquantized`（~10.6GB）。
  本家 `google/gemma-4-E2B-it` は gated（要 HF 認証）。
- HF からの初回 DL は xet バックエンドがストールしやすい。`HF_HUB_DISABLE_XET=1` を付けると安定。

**おまけ: 英→日 音声翻訳(S2TT)** も公式実装なら良好。LibriSpeech の英語音声を「日本語に翻訳して」と
指示すると、内容に忠実な日本語を生成する（例:「クリスマスとローストビーフが迫る時期であり、食事から
連想されるシンボルが最も頭に浮かぶ」）。逐語 ASR より LLM の翻訳能力が活きる用途。

### Cohere Transcribe の検証結果（2026-06）

同じ自声20文を、Cohere Labs の専用 ASR `cohere-transcribe-03-2026`（2B, Fast-Conformer encoder + Transformer decoder, Apache 2.0, 日本語含む14言語）で測定。英語 OpenASR(8セット平均) WER 5.42 で公開当時 SOTA のモデル。

| 指標 | 値 |
|---|---|
| 平均 CER | **11.81%** |
| 中央値 CER | **8.57%** |
| CER≤30% の文数 | 19/20 |
| 完璧(0%) の文数 | 6/20 |
| 平均推論時間 | **0.25s/文**（中央値 0.23s、0.17〜0.51s） |
| リアルタイム係数 (RTF) | **0.038 ≈ 実時間の約26倍速**（音声総尺 132s を 5.08s で処理） |
| モデルロード（ウォーム） | 6.6s |
| メモリ（fp16 重み・実効占有） | **約3.9〜4GB**（プロセス RSS は ~1.3GB、残りは MPS ユニファイドメモリ） |
| CPU使用率（MPS 実行） | 平均 約46% / サンプル最大 約79%（重い処理は GPU 側） |
| 実行デバイス | MPS（Apple Silicon GPU） |

**結論: 速度×精度のバランスで本リポジトリの推奨筆頭。** 日本語逐語 ASR の精度は Qwen3-ASR(8.44%) に次ぐ2位（11.81%、Whisper large-v3-turbo 11.80% とほぼ同率）だが、**推論はローカル勢で最速（0.25s/文・実時間の約26倍速）**で、英単語の大小文字保持や早口言葉も強い。最高精度が要るなら Qwen3-ASR、**精度と速度・実装の素直さ（専用 ASR で後処理不要・Apache 2.0）を両取りする実運用の既定候補としては Cohere Transcribe** を推す。

**強み**
- 英単語/アルファベットを大小文字込みで保持（`GPU`, `GitHub`, `iPhone`, `Python` → CER 0%）。Qwen と同じ美点。
- **早口言葉「赤パジャマ、青パジャマ、黄パジャマ」を唯一 0% で正解**（kotoba/nue/qwen/granite は 40〜63% と全滅していた難所）。
- 長文・複合文に強い（「補聴器の性能は…」「このベンチマークツールでは…」が CER 0〜2%）。

**失点の傾向（いずれも意味は正しい表記揺れ）**
- 漢字を仮名に開く癖: 今日→きょう、明日→あした、家族→かぞ（聞き取り由来の脱落あり）。
- 英語の小文字化: `API`→`api`、`JSON`→`json`（ground truth が大文字でペナルティ）。
- 固有名詞の崩れ: Anthropic / Claude →「アンスロピッツの鶴堂」(CER 50%) が最大の失点。「東京特許許可局」→「東京特許著書」(22%) も未到達だが他モデルより軽微。

**コツ/ハマりどころ**
- transformers 公式実装の `CohereAsrForConditionalGeneration` を使う。**transformers>=5.4** が必要で本リポジトリの他モデル(4.57.x)とは非互換のため、**cohere だけ専用 venv** を作る（トラブルシューティング参照）。
- 言語の自動判定を持たないため、processor に **`language="ja"` を明示**する（未指定だと言語が振れる）。
- 公式リポ `CohereLabs/cohere-transcribe-03-2026` は **gated**（Apache 2.0・規約同意で即時アクセス付与）。HF にログインし、モデルページで規約同意のうえ **gated を読めるトークン**（Read トークン等）で認証する。ungated ミラーは ONNX/GGUF（量子化・別ランタイム）のみで、フル精度ローカル比較には公式リポを使う。

### Qwen3-ASR 1.7B の検証結果（2026-07）

同じ自声20文を、Qwen3-ASR の 1.7B 版 `Qwen/Qwen3-ASR-1.7B`（約 3.5GB, 52言語対応, 0.6B と同一の `qwen-asr` パッケージ）で測定。

| 指標 | 値 |
|---|---|
| 平均 CER | **6.48%** |
| 中央値 CER | **0.00%** |
| 完璧(0%) の文数 | **11/20** |
| CER≤30% の文数 | 19/20 |
| 平均推論時間 | 0.55s/文（ウォーム、初回ロード直後の1文目のみ 0.73s） |
| モデルロード（ウォーム） | 6.8s |
| 実行デバイス | MPS（Apple Silicon GPU, fp16） |

**結論: 本リポジトリの最高精度を更新（8.44% → 6.48%）。** 20文中11文を完全一致で認識し、推論も 0.6B（0.78s/文）より速い 0.55s/文。0.6B と同一インターフェースなので導入コストはモデルサイズ差（1.2GB → 3.5GB）のみ。

**強み**
- 0.6B の美点（英単語/アルファベットの大小文字保持: Python, API, JSON, GitHub, OpenAI, ChatGPT, Anthropic, Claude → すべて CER 0%）をそのまま継承。
- **難読地名「東京特許許可局」を CER 3.7% でほぼ攻略**（全モデルが未到達だった難所。Cohere でも 22%）。
- 早口言葉「赤パジャマ青パジャマ黄パジャマ」も 14.81% と健闘（0.6B 含む従来勢は 40〜63%、0% は Cohere のみ）。

**失点の傾向**
- 最大の失点は日付文（CER 45%）: 「二〇二五年十二月二十八日」→「2025年10月28日」。漢数字→算用数字の表記揺れに加え「十二月」→「10月」の聞き取り誤りを含む。
- カタカナ化: iPhone →「アイフォン」(25%)。0.6B が得意だった箇所でむしろ揺れた。
- 語彙置換: 補聴器→「ホットオーキ」(16.67%)、ベンチマーク→「ベンチマート」。

**コツ/ハマりどころ**
- HF からの初回 DL（~3.5GB）は xet バックエンドがストールしやすい（実測で 450MB 地点から 30分無進捗）。**`HF_HUB_DISABLE_XET=1` を付けると安定**（Gemma の項と同じ既知問題）。
- それ以外は 0.6B と完全に同じ手順で動く（`pip install qwen-asr`、transformers 4.57.x 環境で OK、専用 venv 不要）。

## 今後の追加候補モデル（2026-04 調査）

以下は調査済みで未実装。優先度順。

### S ランク（推奨）

1. ~~**IBM granite-4.0-1b-speech**~~ — 検証済み（上記テーブル）。自声20文では 17.25%。英単語の大小文字扱いが特徴的で、OpenASR の公称 WER とは別系統の CER ペナルティが効く。
2. **Voxtral-Mini-3B-2507** (Mistral) — 3B / Apache 2.0 / Audio-LLM（ASR + Q&A 同時）。Neosophie ベンチで WER 24%。
3. ~~**Cohere-transcribe-03-2026**~~ — 検証済み（上記テーブル / 「Cohere Transcribe の検証結果」）。自声20文では **11.81%**（Qwen3-ASR に次ぐ2位）。英語 OpenASR の SOTA(WER 5.42) は日本語1位には直結しなかったが実用圏。ライセンスは Apache 2.0、リポは gated（規約同意で即時）。

### A ランク

- **Phi-4-multimodal-instruct** (MS, 5.6B, MIT) — 日本語含む8言語学習。
- **Qwen3-Omni-30B-A3B** (Alibaba, MoE active 3B) — Audio/Video/Text フルマルチモーダル。ローカル実行は重い。
- **Meta Omnilingual ASR** (最大 7B) — 1,690 言語対応、fairseq2 必須。
- **litagin/anime-whisper** (Kotoba-Whisper 派生) — アニメ調音声 5,300時間で fine-tune。ドメイン特化事例。
- **sbintuitions/nest-ja-0.1b** — 日本語 35K 時間 SSL、ASR fine-tune のベースとして。

### B ランク（紹介のみ）

Meta SeamlessM4T-v2-Large（CC-BY-NC）、Meta MMS、ESPnet OWSM v4、Fun-ASR-Nano-2512、既存の Japanese HuBERT / wav2vec2 系。

※ Gemma 3n E2B/E4B は後継の **Gemma 4 E2B (audio)** として実装・検証済み（上記「対応モデル」「ベンチマーク実績」を参照）。

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

### macOS 更新後の環境トラブル（2026-10 実測）

- **scipy の `dlopen ... _spropack ... __thread_bss` エラー**: scipy 1.15.x の wheel が新しい macOS のローダーに弾かれる。`uv pip install --python <venv>/bin/python scipy==1.14.1` で解消（`.venv` / `~/cohere-venv` / `~/gemma-venv` すべて）。
- **`Could not load libtorchcodec` / `libavutil.56.dylib`**: torchcodec が Homebrew の ffmpeg 版数と合わず読めない。使っていなければ `uv pip uninstall --python .venv/bin/python torchcodec` で回避（kotoba で発生）。
- **gemma-venv で `Segmentation fault`（exit 139）**: torch 2.14.1 + transformers 5.18.0 の組み合わせでモデルロード中に発生。`torch==2.12.1 transformers==5.12.1`（cohere-venv と同じ版）に固定すると動く。

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
