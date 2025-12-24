#!/bin/bash
# setup_reazonspeech.sh
# Mac で ReazonSpeech ESPnet v2 をセットアップするスクリプト (uv版)

set -e

echo "=== ReazonSpeech ESPnet v2 セットアップ (uv) ==="

# uvの確認
if ! command -v uv &> /dev/null; then
    echo "エラー: uvがインストールされていません"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "依存パッケージをインストール中..."

# 依存パッケージのインストール
uv add torch torchaudio espnet espnet_model_zoo psutil

# ReazonSpeechをGitHubからインストール
echo "ReazonSpeechをGitHubからインストール中..."
uv pip install "git+https://github.com/reazon-research/ReazonSpeech#subdirectory=pkg/espnet-asr"

echo ""
echo "=== セットアップ完了 ==="
echo ""
echo "使用方法:"
echo "  uv run python benchmark.py <音声ファイル> <正解テキスト> reazonspeech"
echo ""
echo "初回実行時はモデルのダウンロード（約500MB）が行われます。"