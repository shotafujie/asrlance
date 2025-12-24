#!/usr/bin/env python3
"""
benchmark.py - 音声認識ベンチマークツール
MLX Whisper, OpenAI Whisper, ReazonSpeech ESPnet v2 に対応

使用方法:
    python benchmark.py <音声ファイルパス> <正解テキストファイル> <モデルエイリアス> [出力ログファイル名]

モデルエイリアス:
    - mlx: MLX Whisper Large V3 Turbo
    - whisper: OpenAI Whisper Large V3 Turbo OSS
    - reazonspeech: ReazonSpeech ESPnet v2
"""

import sys
import os
from typing import Optional

from fileRecognizer import recognize, get_model_info, get_available_models


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    CER（文字誤り率）を計算
    
    Args:
        reference: 正解テキスト
        hypothesis: 認識結果テキスト
    
    Returns:
        float: CER（パーセント）
    """
    # 空白を除去して文字単位で比較
    ref_chars = list(reference.replace(" ", "").replace("　", ""))
    hyp_chars = list(hypothesis.replace(" ", "").replace("　", ""))
    
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 100.0
    
    # レーベンシュタイン距離を計算
    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
    
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # 削除
                    d[i][j - 1] + 1,      # 挿入
                    d[i - 1][j - 1] + 1   # 置換
                )
    
    cer = (d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)) * 100
    return cer


def load_ground_truth(filepath: str) -> str:
    """正解テキストファイルを読み込む"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()


def save_results(
    output_path: str,
    audio_path: str,
    model_info: dict,
    processing_time: float,
    avg_cpu: float,
    max_cpu: float,
    cer: float,
    recognized_text: str
):
    """結果をファイルに保存"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"ファイル: {audio_path}\n")
        f.write(f"モデル: {model_info['name']} ({model_info['alias']})\n")
        f.write(f"呼称: {model_info['display_name']}\n")
        f.write(f"処理時間: {processing_time:.2f}秒\n")
        f.write(f"平均CPU: {avg_cpu:.2f}% 最大CPU: {max_cpu:.2f}%\n")
        f.write(f"CER（文字誤り率）: {cer:.2f}%\n")
        f.write(f"認識テキスト:\n{recognized_text}\n")


def print_usage():
    """使用方法を表示"""
    print("使用方法:")
    print("  python benchmark.py <音声ファイルパス> <正解テキストファイル> <モデルエイリアス> [出力ログファイル名]")
    print()
    print("モデルエイリアス:")
    for alias in get_available_models():
        info = get_model_info(alias)
        print(f"  - {alias}: {info['name']}")
    print()
    print("例:")
    print("  python benchmark.py ./audio.wav ./ground_truth.txt reazonspeech result.txt")


def main():
    # 引数チェック
    if len(sys.argv) < 4:
        print_usage()
        sys.exit(1)
    
    audio_path = sys.argv[1]
    ground_truth_path = sys.argv[2]
    model_alias = sys.argv[3].lower()
    output_path = sys.argv[4] if len(sys.argv) > 4 else f"{model_alias}_log.txt"
    
    # ファイル存在チェック
    if not os.path.exists(audio_path):
        print(f"エラー: 音声ファイルが見つかりません: {audio_path}")
        sys.exit(1)
    
    if not os.path.exists(ground_truth_path):
        print(f"エラー: 正解テキストファイルが見つかりません: {ground_truth_path}")
        sys.exit(1)
    
    # モデルエイリアスチェック
    available_models = get_available_models()
    if model_alias not in available_models:
        print(f"エラー: 未対応のモデルエイリアス: {model_alias}")
        print(f"利用可能なモデル: {', '.join(available_models)}")
        sys.exit(1)
    
    # モデル情報取得
    model_info = get_model_info(model_alias)
    
    print(f"音声認識を開始します...")
    print(f"  音声ファイル: {audio_path}")
    print(f"  モデル: {model_info['name']}")
    
    # 正解テキスト読み込み
    ground_truth = load_ground_truth(ground_truth_path)
    
    # 音声認識実行
    try:
        recognized_text, processing_time, avg_cpu, max_cpu = recognize(audio_path, model_alias)
    except ImportError as e:
        print(f"エラー: 必要なモジュールがインストールされていません: {e}")
        if model_alias == "reazonspeech":
            print("  pip install reazonspeech-espnet torch torchaudio espnet espnet_model_zoo")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: 音声認識中にエラーが発生しました: {e}")
        sys.exit(1)
    
    # CER計算
    cer = calculate_cer(ground_truth, recognized_text)
    
    # 結果保存
    save_results(
        output_path,
        audio_path,
        model_info,
        processing_time,
        avg_cpu,
        max_cpu,
        cer,
        recognized_text
    )
    
    print(f"\n{model_info['name']} の結果は {output_path} に保存されました。")
    print(f"  処理時間: {processing_time:.2f}秒")
    print(f"  CER: {cer:.2f}%")


if __name__ == "__main__":
    main()