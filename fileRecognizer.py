"""
fileRecognizer.py - 音声認識処理モジュール
MLX Whisper, OpenAI Whisper, ReazonSpeech ESPnet v2 に対応
"""

import os
import time
import psutil
from typing import Tuple, Optional

# ============================================================
# MLX Whisper 認識
# ============================================================
def recognize_with_mlx_whisper(audio_path: str) -> Tuple[str, float, float, float]:
    """
    MLX Whisperを使用して音声認識を行う
    
    Returns:
        Tuple[str, float, float, float]: (認識テキスト, 処理時間, 平均CPU, 最大CPU)
    """
    import mlx_whisper
    
    process = psutil.Process(os.getpid())
    cpu_samples = []
    
    start_time = time.time()
    
    # CPU使用率のサンプリング開始
    cpu_samples.append(process.cpu_percent(interval=None))
    
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        language="ja"
    )
    
    cpu_samples.append(process.cpu_percent(interval=None))
    end_time = time.time()
    
    processing_time = end_time - start_time
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    max_cpu = max(cpu_samples) if cpu_samples else 0
    
    return result["text"], processing_time, avg_cpu, max_cpu


# ============================================================
# OpenAI Whisper 認識
# ============================================================
def recognize_with_whisper(audio_path: str) -> Tuple[str, float, float, float]:
    """
    OpenAI Whisperを使用して音声認識を行う
    
    Returns:
        Tuple[str, float, float, float]: (認識テキスト, 処理時間, 平均CPU, 最大CPU)
    """
    import whisper
    
    process = psutil.Process(os.getpid())
    cpu_samples = []
    
    # モデルロード
    model = whisper.load_model("large-v3-turbo")
    
    start_time = time.time()
    cpu_samples.append(process.cpu_percent(interval=None))
    
    result = model.transcribe(audio_path, language="ja")
    
    cpu_samples.append(process.cpu_percent(interval=None))
    end_time = time.time()
    
    processing_time = end_time - start_time
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    max_cpu = max(cpu_samples) if cpu_samples else 0
    
    return result["text"], processing_time, avg_cpu, max_cpu


# ============================================================
# ReazonSpeech k2 認識
# ============================================================
def recognize_with_reazonspeech(audio_path: str) -> Tuple[str, float, float, float]:
    """
    ReazonSpeech k2を使用して音声認識を行う
    """
    from reazonspeech.k2.asr import load_model, transcribe, audio_from_path
    
    process = psutil.Process(os.getpid())
    cpu_samples = []
    
    print("ReazonSpeech k2 モデルをロード中...")
    model = load_model()
    
    audio = audio_from_path(audio_path)
    
    start_time = time.time()
    cpu_samples.append(process.cpu_percent(interval=None))
    
    result = transcribe(model, audio)
    
    cpu_samples.append(process.cpu_percent(interval=None))
    end_time = time.time()
    
    processing_time = end_time - start_time
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    max_cpu = max(cpu_samples) if cpu_samples else 0
    
    return result.text, processing_time, avg_cpu, max_cpu

# ============================================================
# 統合認識関数
# ============================================================
def recognize(audio_path: str, model_alias: str) -> Tuple[str, float, float, float]:
    """
    指定されたモデルで音声認識を行う
    
    Args:
        audio_path: 音声ファイルのパス
        model_alias: モデルエイリアス ('mlx', 'whisper', 'reazonspeech')
    
    Returns:
        Tuple[str, float, float, float]: (認識テキスト, 処理時間, 平均CPU, 最大CPU)
    """
    if model_alias == "mlx":
        return recognize_with_mlx_whisper(audio_path)
    elif model_alias == "whisper":
        return recognize_with_whisper(audio_path)
    elif model_alias == "reazonspeech":
        return recognize_with_reazonspeech(audio_path)
    else:
        raise ValueError(f"未対応のモデル: {model_alias}")


# モデル情報の定義
MODEL_INFO = {
    "mlx": {
        "name": "MLX Whisper Large V3 Turbo",
        "alias": "mlx",
        "display_name": "large-v3-turbo (MLX)"
    },
    "whisper": {
        "name": "OpenAI Whisper Large V3 Turbo OSS",
        "alias": "whisper", 
        "display_name": "large-v3-turbo（ローカル）"
    },
    "reazonspeech": {
        "name": "ReazonSpeech k2",
        "alias": "reazonspeech",
        "display_name": "reazonspeech-k2-asr（Next-gen Kaldi, 159M params）"
    }
}


def get_model_info(model_alias: str) -> dict:
    """モデル情報を取得"""
    if model_alias not in MODEL_INFO:
        raise ValueError(f"未対応のモデル: {model_alias}")
    return MODEL_INFO[model_alias]


def get_available_models() -> list:
    """利用可能なモデルのリストを取得"""
    return list(MODEL_INFO.keys())