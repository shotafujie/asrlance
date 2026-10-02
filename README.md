# asrlance

複数の音声認識（ASR）モデルを、同じ音声・同じ指標で比較するベンチマークツールです。音声と正解テキストを入力に、文字誤り率（CER）・処理時間・CPU 使用率を計測します。対象は音声認識モデルを選定・評価する開発者・研究者です。

## 何を測ったか

| 項目 | 内容 |
|---|---|
| 音声 | 自分の声の日本語 **50文**（16kHz / mono、3.5〜11.6秒）。20文（短文・技術用語・数字・固有名詞・滑舌）＋30文（長めの会話調・フィラー入り） |
| 正解 | 各文の正解テキスト（発話との一致を本人が確認済み） |
| モデル | 下記 8 モデル（Apple Silicon の MPS / MLX で実行） |
| 指標 | **CER**（低いほど良い。句読点も誤りとして数える）、1文あたり推論時間 |
| 条件 | 8モデルすべて同一環境で 2026-10 に再測定 |

## 結果（50文・2026-10）

| 順位 | モデル | 平均 CER | 中央値 | 推論時間/文 |
|---|---|---|---|---|
| 1 | **Qwen3-ASR 1.7B** | **8.67%** | 7.41% | 1.64s |
| 2 | Cohere Transcribe | 10.27% | 7.29% | 0.38s |
| 3 | Qwen3-ASR 0.6B | 11.02% | 8.71% | 1.14s |
| 4 | Whisper large-v3-turbo (MLX) | 12.38% | 10.79% | **0.28s** |
| 5 | IBM granite-4.0-1b-speech | 13.31% | 9.76% | 1.53s |
| 6 | rinna/nue-asr | 13.85% | 7.90% | 1.40s |
| 7 | Kotoba-Whisper v2.0 | 17.12% | 13.18% | 0.77s |
| 8 | Gemma 4 E2B (audio) | 28.84%※ | 17.88% | 1.70s |

※ 1文で同じ文字列を繰り返す暴走（CER 480%）を含む。除くと約 19.2%。

**わかったこと**

- **精度重視なら Qwen3-ASR 1.7B**。50文で最良（8.67%）。短文・英単語・固有名詞に強い。
- **速度重視なら Whisper turbo (MLX) か Cohere**（0.28〜0.38s/文）。Cohere は精度も2位（中央値は 1.7B と同等）。
- **上位2つの差は小さく、文の種類で逆転する**。20文のみなら Qwen 1.7B が圧勝（6.48% vs 11.81%）、30文のみなら Cohere が1位（9.25% vs 10.13%）。
- **句読点を除くと Whisper turbo が Qwen 1.7B に並ぶ**（6.77% vs 6.63%）。表記の癖で順位が動くので、用途に合わせて評価する。
- Gemma 4 E2B は実用圏だが暴走が出る。MLX(mlx-vlm) 経路は精度が約8倍悪化するので使わない。

20文／30文の個別結果・モデル別の検証メモ・句読点除外の表は **[docs/benchmark-results.md](docs/benchmark-results.md)**。

## セットアップ

**前提**

- Python 3.10 以上
- macOS（Apple Silicon 推奨）または Linux
- 約2GB以上の空きメモリ（モデルにより異なる）

**インストール**

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

## 使い方

```bash
# 1ファイル: python benchmark.py <音声> <正解テキスト> <モデル> [出力ファイル]
python benchmark.py ./audio.wav ./ground_truth.txt qwen17b result.txt

# ディレクトリ一括（<stem>.wav と <stem>.txt が対）。1モデルを1回だけロード
HF_HUB_DISABLE_XET=1 python batch_benchmark.py qwen17b my_voice_50 my_voice_50/result_qwen17b.csv
```

`gemma` と `cohere` は transformers のバージョン要件が異なるため**専用 venv** で実行します（[手順](docs/troubleshooting.md)）。

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
├── batch_benchmark.py # 1モデルを1回ロードして複数音声を一括評価
├── pyproject.toml     # プロジェクト設定
├── my_voice/ my_voice_30/ my_voice_50/  # 自声データ（音声・結果CSVは git 管理外）
└── docs/              # 詳細な結果・調査・トラブルシューティング
```

## ドキュメント

- [docs/benchmark-results.md](docs/benchmark-results.md) — 全ベンチマーク結果とモデル別の検証メモ
- [docs/future-candidates.md](docs/future-candidates.md) — 今後の追加候補モデル
- [docs/troubleshooting.md](docs/troubleshooting.md) — 環境構築・専用 venv・トラブル対処
