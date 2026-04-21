"""
fileRecognizer.py - 音声認識処理モジュール
MLX Whisper, OpenAI Whisper, ReazonSpeech k2, FunASR SenseVoiceSmall,
Moonshine (Japanese Base), Kotoba-Whisper v2.0, rinna/nue-asr, Qwen3-ASR 0.6B,
IBM granite-4.0-1b-speech に対応
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
# FunASR SenseVoiceSmall 認識
# ============================================================
def recognize_with_funasr(audio_path: str) -> Tuple[str, float, float, float]:
    """
    FunASR SenseVoiceSmall を使用して音声認識を行う（多言語, 日本語対応）
    """
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    process = psutil.Process(os.getpid())
    cpu_samples = []

    print("FunASR SenseVoiceSmall モデルをロード中...")
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="mps",
        disable_update=True,
    )

    start_time = time.time()
    cpu_samples.append(process.cpu_percent(interval=None))

    res = model.generate(
        input=audio_path,
        cache={},
        language="ja",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
    )

    cpu_samples.append(process.cpu_percent(interval=None))
    end_time = time.time()

    text = rich_transcription_postprocess(res[0]["text"])

    processing_time = end_time - start_time
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    max_cpu = max(cpu_samples) if cpu_samples else 0

    return text, processing_time, avg_cpu, max_cpu


# ============================================================
# Moonshine (Japanese Base) 認識
# ============================================================
def recognize_with_moonshine(audio_path: str) -> Tuple[str, float, float, float]:
    """
    Moonshine の日本語特化モデル（Flavors of Moonshine, Base 58M）を使用して音声認識を行う
    """
    from moonshine_voice import (
        Transcriber,
        TranscriptEventListener,
        download_model,
        load_wav_file,
    )

    process = psutil.Process(os.getpid())
    cpu_samples = []

    print("Moonshine 日本語モデルをロード中...")
    model_path, model_arch = download_model("ja")

    # 日本語など非ラテン言語ではハルシネーション抑制のため max_tokens_per_second=13.0 が推奨
    transcriber = Transcriber(
        model_path=model_path,
        model_arch=model_arch,
        max_tokens_per_second=13.0,
    )

    collected: list = []

    class _Collector(TranscriptEventListener):
        def on_line_completed(self, event):
            collected.append(event.line.text)

    transcriber.add_listener(_Collector())

    audio_data, sample_rate = load_wav_file(audio_path)

    start_time = time.time()
    cpu_samples.append(process.cpu_percent(interval=None))

    transcriber.transcribe_without_streaming(audio_data, sample_rate)

    cpu_samples.append(process.cpu_percent(interval=None))
    end_time = time.time()

    text = "".join(collected)

    processing_time = end_time - start_time
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    max_cpu = max(cpu_samples) if cpu_samples else 0

    return text, processing_time, avg_cpu, max_cpu


# ============================================================
# Kotoba-Whisper v2.0 認識
# ============================================================
def recognize_with_kotoba_whisper(audio_path: str) -> Tuple[str, float, float, float]:
    """
    Kotoba-Whisper v2.0（ReazonSpeech で蒸留した日本語特化 Whisper）で音声認識を行う
    """
    import torch
    from transformers import pipeline

    process = psutil.Process(os.getpid())
    cpu_samples = []

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print(f"Kotoba-Whisper v2.0 モデルをロード中... (device={device})")
    pipe = pipeline(
        "automatic-speech-recognition",
        model="kotoba-tech/kotoba-whisper-v2.0",
        torch_dtype=torch.float32,
        device=device,
    )

    start_time = time.time()
    cpu_samples.append(process.cpu_percent(interval=None))

    result = pipe(
        audio_path,
        chunk_length_s=15,
        generate_kwargs={"language": "ja", "task": "transcribe"},
    )

    cpu_samples.append(process.cpu_percent(interval=None))
    end_time = time.time()

    processing_time = end_time - start_time
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    max_cpu = max(cpu_samples) if cpu_samples else 0

    return result["text"], processing_time, avg_cpu, max_cpu


# ============================================================
# rinna/nue-asr 認識
# ============================================================
def recognize_with_nue_asr(audio_path: str) -> Tuple[str, float, float, float]:
    """
    rinna/nue-asr（HuBERT + GPT-NeoX, ReazonSpeech v1 で学習）で音声認識を行う
    """
    import nue_asr

    process = psutil.Process(os.getpid())
    cpu_samples = []

    print("rinna/nue-asr モデルをロード中...")
    model = nue_asr.load_model("rinna/nue-asr")
    tokenizer = nue_asr.load_tokenizer("rinna/nue-asr")

    start_time = time.time()
    cpu_samples.append(process.cpu_percent(interval=None))

    result = nue_asr.transcribe(model, tokenizer, audio_path)

    cpu_samples.append(process.cpu_percent(interval=None))
    end_time = time.time()

    processing_time = end_time - start_time
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    max_cpu = max(cpu_samples) if cpu_samples else 0

    return result.text, processing_time, avg_cpu, max_cpu


# ============================================================
# Qwen3-ASR 0.6B 認識
# ============================================================
def recognize_with_qwen3_asr(audio_path: str) -> Tuple[str, float, float, float]:
    """
    Qwen3-ASR 0.6B（Alibaba Qwen, 52言語対応, 日本語含む）で音声認識を行う
    """
    import torch
    from qwen_asr import Qwen3ASRModel

    process = psutil.Process(os.getpid())
    cpu_samples = []

    if torch.cuda.is_available():
        device_map = "cuda:0"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device_map = "mps"
        dtype = torch.float16
    else:
        device_map = "cpu"
        dtype = torch.float32

    print(f"Qwen3-ASR 0.6B モデルをロード中... (device_map={device_map})")
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        dtype=dtype,
        device_map=device_map,
        max_inference_batch_size=32,
        max_new_tokens=256,
    )

    start_time = time.time()
    cpu_samples.append(process.cpu_percent(interval=None))

    results = model.transcribe(audio=audio_path, language="Japanese")

    cpu_samples.append(process.cpu_percent(interval=None))
    end_time = time.time()

    processing_time = end_time - start_time
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    max_cpu = max(cpu_samples) if cpu_samples else 0

    return results[0].text, processing_time, avg_cpu, max_cpu


# ============================================================
# IBM granite-4.0-1b-speech 認識
# ============================================================
def recognize_with_granite(audio_path: str) -> Tuple[str, float, float, float]:
    """
    IBM granite-4.0-1b-speech（多言語 ASR, 日本語対応）で音声認識を行う
    """
    import soundfile as sf
    import torch
    import torchaudio
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    process = psutil.Process(os.getpid())
    cpu_samples = []

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"granite-4.0-1b-speech モデルをロード中... (device={device})")
    model_name = "ibm-granite/granite-4.0-1b-speech"
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, device_map=device, torch_dtype=dtype
    )

    # torchaudio 2.9.x の torchaudio.load は torchcodec 経由となり ABI 互換性問題があるため soundfile で読み込み
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    chat = [{
        "role": "user",
        "content": "<|audio|>can you transcribe the speech into a written format?",
    }]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    start_time = time.time()
    cpu_samples.append(process.cpu_percent(interval=None))

    model_inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)
    outputs = model.generate(
        **model_inputs, max_new_tokens=200, do_sample=False, num_beams=1
    )

    cpu_samples.append(process.cpu_percent(interval=None))
    end_time = time.time()

    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = outputs[0, num_input_tokens:].unsqueeze(0)
    text = tokenizer.batch_decode(
        new_tokens, add_special_tokens=False, skip_special_tokens=True
    )[0]

    processing_time = end_time - start_time
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    max_cpu = max(cpu_samples) if cpu_samples else 0

    return text, processing_time, avg_cpu, max_cpu


# ============================================================
# 統合認識関数
# ============================================================
def recognize(audio_path: str, model_alias: str) -> Tuple[str, float, float, float]:
    """
    指定されたモデルで音声認識を行う

    Args:
        audio_path: 音声ファイルのパス
        model_alias: モデルエイリアス ('mlx', 'whisper', 'reazonspeech', 'funasr', 'moonshine')

    Returns:
        Tuple[str, float, float, float]: (認識テキスト, 処理時間, 平均CPU, 最大CPU)
    """
    if model_alias == "mlx":
        return recognize_with_mlx_whisper(audio_path)
    elif model_alias == "whisper":
        return recognize_with_whisper(audio_path)
    elif model_alias == "reazonspeech":
        return recognize_with_reazonspeech(audio_path)
    elif model_alias == "funasr":
        return recognize_with_funasr(audio_path)
    elif model_alias == "moonshine":
        return recognize_with_moonshine(audio_path)
    elif model_alias == "kotoba":
        return recognize_with_kotoba_whisper(audio_path)
    elif model_alias == "nue":
        return recognize_with_nue_asr(audio_path)
    elif model_alias == "qwen":
        return recognize_with_qwen3_asr(audio_path)
    elif model_alias == "granite":
        return recognize_with_granite(audio_path)
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
    },
    "funasr": {
        "name": "FunASR SenseVoiceSmall",
        "alias": "funasr",
        "display_name": "SenseVoiceSmall（多言語, 234M params）"
    },
    "moonshine": {
        "name": "Moonshine Japanese Base",
        "alias": "moonshine",
        "display_name": "moonshine-base-ja（Flavors of Moonshine, 58M params）"
    },
    "kotoba": {
        "name": "Kotoba-Whisper v2.0",
        "alias": "kotoba",
        "display_name": "kotoba-whisper-v2.0（ReazonSpeech 蒸留, 日本語特化）"
    },
    "nue": {
        "name": "rinna/nue-asr",
        "alias": "nue",
        "display_name": "nue-asr（HuBERT + GPT-NeoX, ReazonSpeech v1 学習）"
    },
    "qwen": {
        "name": "Qwen3-ASR 0.6B",
        "alias": "qwen",
        "display_name": "Qwen3-ASR-0.6B（Alibaba Qwen, 52言語対応）"
    },
    "granite": {
        "name": "IBM granite-4.0-1b-speech",
        "alias": "granite",
        "display_name": "granite-4.0-1b-speech（IBM, 1B, 多言語, 日本語対応）"
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