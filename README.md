# asrlance のセットアップ・必要モジュール一括導入

## 1. 仮想環境（.venv）作成＆有効化

```
python -m venv .venv
source .venv/bin/activate
```
（uv利用の場合は `uv venv .venv` → `source .venv/bin/activate` でも可）

## モジュールインストール

実行に必要なモジュールを以下のコマンドでインストールしてください．
`pip install -r requirements.txt`

## 実行

mlx-whisperまたはOpenAIのlarge-v3-turboモデルを使うことを想定しています．
ローカルで実行することを想定しているため，あらかじめ必要なスペックはご確認ください．
`python benchmark.py <音声ファイルパス> <正解テキストファイル> <モデルエイリアス（mlx or whisper）> [出力ログファイル名]`

## 結果

実行が成功すると下記のような文章が表示されます．
```
OpenAI Whisper Large V3 Turbo OSS の結果は whisper_log.txt に保存されました。

```

保存されたファイルを確認してみると，下記のように結果が格納されています．
```
ファイル: (使用音声ファイルパス)
モデル: OpenAI Whisper Large V3 Turbo OSS (whisper)
呼称: large-v3-turbo（ローカル）
処理時間: x.xx秒
平均CPU: x.xx% 最大CPU: xx.xx%
CER（文字誤り率）: x.xx%
認識テキスト:
(実際の認識結果)
```
以上．
