# Nemotron評価と通常CLI（2026-10-30公開）

Apple M4 Max / 128GB、本人確認済み日本語50発話・413.1753秒を、2026-10-29に4構成で全文入力・逐次測定した結果です。入力・参照と既存scorerの一致をローカルで確認しています。音声・参照・認識本文・個別入力hashは公開しません。公開するのは集計値と実行コードです。

| 構成 | macro CER | micro CER | 平均推論/文 | RTF | GPUメモリ指標 |
|---|---:|---:|---:|---:|---:|
| Nemotron参照 BF16/MPS | 6.96% | 7.24% | 3.309s | 0.4005 | 64.57GiB |
| Nemotron改良 MLX8 | **6.52%** | 6.93% | 0.498s | 0.0603 | 33.08GiB |
| Whisper large-v3-turbo MLX | 12.38% | 12.40% | **0.288s** | 0.0349 | 2.31GiB |
| Qwen3-ASR1.7B MPS | 8.67% | 9.11% | 0.902s | 0.1092 | 5.62GiB |

[集計CSV](nemotron-results-20261029.csv)。全4構成50件完走。Nemotronの256生成token上限到達なし。Whisper/Qwenの上限判定は元runnerから取得不能。PyTorchメモリは100ms標本のMPS driver観測ピーク、MLXはallocator peak。RSSとの加算や厳密な削減率比較には使えません。MLX8は参照より約6.64倍速いが、Whisperより約1.73倍遅い。量子化・backend・promptが同時に違うため、量子化だけの効果ではありません。

**これは既存ローカル評価commit8416652の結果を再利用した掲載です。新CLIで50件を再測定した値ではありません。** 当該50件は調整にも利用した再利用データで、独立holdoutの汎化性能ではありません。未再測定の旧8モデルすべてとの新しい総合順位は作りません。

## 評価条件

モデルは公式[NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)、revision `9d8cd6b79bfdd427b6558a5fdbf76fcaf8c75341`。重みはOpenMDW-1.1、重みそのものは本リポジトリで配布しません。元モデルのLICENSE/THIRD_PARTY_NOTICESを保持してください。

- `nemotron-stock`: 元BF16、公式TransformersコードをMPS/eagerで実行。ロード時の大きなparameter配置だけ同期・cache解放し、推論前に元処理へ戻す。grouped_mm・MoE/decode上書きなし。公式CUDA無変更構成とは呼びません。
- `nemotron-mlx8`: mlx-audio0.6.3 / mlx0.32.4 / mlx-metal0.32.4。公式量子化predicateによる8bit affine/group64。知覚encoderは非量子化。元checkpointの未使用CTC timestamp41キーを形状allowlistで確認して除外しstrict load。単語timestampは非対応。speaker encoderは保持。
- stockはLanguage: as spoken、MLX8はLanguage: Japaneseの逐語prompt。MLX8の32000sample以下は短入力promptへ切替。全文50件は全てこの長さを超えるので、短入力枝の検証とは扱いません。greedy、temperature0、最大256token。
- 元のMLX8評価には限定割合formatterがあったが、50件に割合はなく全件raw/formatted CERが同じでした。公開CLIは**後処理なし**。任意の新しい音声に対する既存アプリとの出力同等性は主張しません。
- `benchmark.calculate_cer`を使用。ASCII/全角空白だけ除去し、句読点・大小文字・数字表記は保持。macroは文平均、microは総編集数/総参照文字数。
- batch1、モデルロードと合成1秒無音2回の暖機は別。音声読込・前処理・生成・decodeを含む同期済み全文推論時間。RTF=総処理秒/原音声秒。発話終了後遅延や起動完了時間ではありません。
- 旧endpoint/VAD付き5.66%やGijilog end-to-endは入力・区間・整形・採点条件が別。この表の6.52%と混ぜません。

## インストールとモデル準備

Apple Silicon、Python3.11、十分なRAMが必要です。実測は128GB。実行は16GiB空きとswap増分512MiBの停止条件を持ち、MLX8でも約33GiB allocatorを使います。共通requirement.txtの古いモデル環境と混ぜず、各環境を分けます。

```sh
python3.11 -m venv .venv-nemotron-mlx8
.venv-nemotron-mlx8/bin/pip install -r nemotron/requirements-mlx8.txt
# BF16参照を使う場合だけ別環境
python3.11 -m venv .venv-nemotron-stock
.venv-nemotron-stock/bin/pip install -r nemotron/requirements-stock.txt
```

stockのTransformersは公式commit `144ecf007b8b4f2910da104c84a79801db6985a6`、測定版5.22.0.dev0、torch2.15.0。MLX AudioはMITの公式release0.6.3、commit `8c244a1faff44cfa69e3151c195f52a4c1c90be5`。未知のremote codeやpickleを実行しません。

元モデルは公式リポジトリの上記revisionをHugging Face CLI等で任意のローカルディレクトリへ取得してください。重みは大容量です。実行時の自動downloadはありません。取得例（ライセンスを確認し、必要容量を確保した場合のみ）:

```sh
hf download nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
  --revision 9d8cd6b79bfdd427b6558a5fdbf76fcaf8c75341 --local-dir ./models/nemotron-bf16
export ASRLANCE_NEMOTRON_STOCK_MODEL="$PWD/models/nemotron-bf16"
# 他のローカルASRアプリがある場合、そのアプリと同一のlockパスを指定する。
# 独立した環境だけなら以下の専用lockを使用できる。
mkdir -p .asrlance
export ASRLANCE_MODEL_LOCK="$PWD/.asrlance/model.lock"
.venv-nemotron-mlx8/bin/python nemotron/convert_mlx8.py \
  "$ASRLANCE_NEMOTRON_STOCK_MODEL" ./models/nemotron-mlx8
export ASRLANCE_NEMOTRON_MLX8_MODEL="$PWD/models/nemotron-mlx8"
```

変換出力は約35.06GB、実測変換時MLX peak約73.5GiB。元モデル保存容量も別途必要。変換器は出力先の68GiB空き（出力＋reserve）を要求し、既存出力を上書きしません。変換レシピは測定時と同じAPI・設定から分離したものですが、公開時に大容量変換を再実行したわけではありません。変換後は掲載済み全ファイルSHA256との一致を要求し、不一致なら実行しません。差を確認せずmanifestを書き換えて回避しないでください。

元BF16はconfig・shard header/sizeのidentityを確認（全shardデータhash保証ではない）。MLX8は全ファイルhash・inventory・quantizationを検証。manifestに音声・個人情報は含めていません。モデルのJSONに異なるrevisionを混ぜないでください。

## 通常コマンド

`batch_benchmark.py`の一覧・実行と、`benchmark.py`/`fileRecognizer`の単一ファイル経路に2aliasを登録済みです。VoiceTypeのコード・ユーザー固有パスには依存しません。

```sh
.venv-nemotron-mlx8/bin/python batch_benchmark.py --list
# dataには16kHz/mono/PCM16の.wavと同名.txtを用意。任意件数、各20ms以上。
.venv-nemotron-mlx8/bin/python batch_benchmark.py nemotron-mlx8 ./data .asrlance/mlx8.csv
.venv-nemotron-stock/bin/python batch_benchmark.py nemotron-stock ./data .asrlance/stock.csv
.venv-nemotron-mlx8/bin/python benchmark.py ./data/example.wav ./data/example.txt nemotron-mlx8
```

Nemotronのbatch出力はcase番号・CER・処理秒・音声秒だけのCSV、stdoutは集計JSONです。参照/認識本文は保存せず、既存結果を上書きしません。token上限・ロード失敗・推論失敗は非zero終了で、空文字の認識成功として採点しません。単一ファイルbenchmarkの従来ログは本文を含むため、手元に置き公開しないでください。

`ASRLANCE_MODEL_LOCK`は必須。ロード前から解放まで同じlockを保持し、既知のVoiceType/Nemotron workerが稼働中なら拒否します。非協調アプリまで排他できる保証はありません。アプリを勝手に終了せず、利用者が空けた時間に実行してください。資源guardは評価プロセス自身だけを終了します。

## 検証と制限

`python -m unittest test_nemotron_runtime -v`で登録・dispatch・短入力境界・model hash・lock・数値出力・失敗伝搬・WAV形式を確認。新CLIの合成短発話による両実モデルsmoke結果は[公開検証記録](nemotron-verification.md)を参照。50件の再測定や新しい精度改善とは扱いません。データは非公開なので第三者が同一50件のCERを完全再現することはできませんが、コード・モデル設定と自身の音声での比較は再現できます。
