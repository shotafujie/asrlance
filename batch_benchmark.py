#!/usr/bin/env python3
"""
batch_benchmark.py - 複数音声ファイルに対して1モデルを1回だけロードして一括ベンチマーク

使用方法:
    python batch_benchmark.py <モデルエイリアス> <音声ディレクトリ> [出力CSVパス]

音声ディレクトリには <stem>.wav と同名の <stem>.txt（正解テキスト）が対で存在する想定。
モデルエイリアス: kotoba, nue, qwen, granite, gemma, cohere
"""

import csv
import glob
import os
import sys
import time
from typing import Callable, List, Tuple

import psutil

from benchmark import calculate_cer, load_ground_truth
from fileRecognizer import get_model_info


def _cpu_sample(process: psutil.Process, samples: list) -> None:
    samples.append(process.cpu_percent(interval=None))


# ============================================================
# モデルごとの load / infer 分離実装
# ============================================================
def build_kotoba_runner() -> Callable[[str], str]:
    import torch
    from transformers import pipeline

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print(f"[kotoba] loading model on device={device} ...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model="kotoba-tech/kotoba-whisper-v2.0",
        torch_dtype=torch.float32,
        device=device,
    )

    def _run(audio_path: str) -> str:
        result = pipe(
            audio_path,
            chunk_length_s=15,
            generate_kwargs={"language": "ja", "task": "transcribe"},
        )
        return result["text"]

    return _run


def build_nue_runner() -> Callable[[str], str]:
    import nue_asr

    # rinna/nue-asr はリポジトリが削除されたため yky-h/nue-asr を使用
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[nue] loading model (yky-h/nue-asr) on device={device} ...")
    model = nue_asr.load_model("yky-h/nue-asr", device=device, fp16=False)
    tokenizer = nue_asr.load_tokenizer("yky-h/nue-asr")

    def _run(audio_path: str) -> str:
        result = nue_asr.transcribe(model, tokenizer, audio_path)
        return result.text

    return _run


def build_qwen_runner() -> Callable[[str], str]:
    import torch
    from qwen_asr import Qwen3ASRModel

    if torch.cuda.is_available():
        device_map = "cuda:0"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device_map = "mps"
        dtype = torch.float16
    else:
        device_map = "cpu"
        dtype = torch.float32

    print(f"[qwen] loading model on device_map={device_map} ...")
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        dtype=dtype,
        device_map=device_map,
        max_inference_batch_size=32,
        max_new_tokens=256,
    )

    def _run(audio_path: str) -> str:
        results = model.transcribe(audio=audio_path, language="Japanese")
        return results[0].text

    return _run


def build_granite_runner() -> Callable[[str], str]:
    import soundfile as sf
    import torch
    import torchaudio
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"[granite] loading model on device={device} ...")
    model_name = "ibm-granite/granite-4.0-1b-speech"
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, device_map=device, torch_dtype=dtype
    )

    chat = [{
        "role": "user",
        "content": "<|audio|>can you transcribe the speech into a written format?",
    }]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    def _run(audio_path: str) -> str:
        # torchaudio 2.9.x の torchaudio.load は torchcodec 経由となり ABI 互換性問題があるため soundfile で読み込み
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        # soundfile returns (T, C) — convert to (C, T)
        wav = torch.from_numpy(data.T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        model_inputs = processor(
            prompt, wav, device=device, return_tensors="pt"
        ).to(device)
        outputs = model.generate(
            **model_inputs, max_new_tokens=200, do_sample=False, num_beams=1
        )
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = outputs[0, num_input_tokens:].unsqueeze(0)
        text = tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )[0]
        return text

    return _run


def build_gemma_runner() -> Callable[[str], str]:
    # Gemma 4 E2B は Google 公式の transformers 実装で動かす（MLX 経路は音声品質が劣化）。
    # transformers>=5.1 が必要で他モデル(4.57.x)と非互換のため専用 venv 推奨。
    import soundfile as sf
    import torch
    import torchaudio
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    from fileRecognizer import _GEMMA_MODEL_ID, _GEMMA_PROMPT, clean_gemma_transcription

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    print(f"[gemma] loading model ({_GEMMA_MODEL_ID}) on device={device} ...")
    processor = AutoProcessor.from_pretrained(_GEMMA_MODEL_ID)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        _GEMMA_MODEL_ID, dtype=dtype, device_map=device
    )
    model.eval()

    messages = [{
        "role": "user",
        "content": [{"type": "audio"}, {"type": "text", "text": _GEMMA_PROMPT}],
    }]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    def _run(audio_path: str) -> str:
        data, sr = sf.read(audio_path, dtype="float32")
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            data = torchaudio.functional.resample(torch.from_numpy(data), sr, 16000).numpy()
        inputs = processor(text=prompt, audio=[data], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        gen = outputs[0, inputs["input_ids"].shape[1]:]
        return clean_gemma_transcription(processor.decode(gen, skip_special_tokens=True))

    return _run


def build_cohere_runner() -> Callable[[str], str]:
    # Cohere Transcribe (cohere-transcribe-03-2026) は transformers 公式実装で動かす。
    # transformers>=5.4 が必要で他モデル(4.57.x)と非互換のため専用 venv 推奨。
    # 言語自動判定を持たないため processor に language="ja" を明示する。
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    from fileRecognizer import _COHERE_MODEL_ID

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    print(f"[cohere] loading model ({_COHERE_MODEL_ID}) on device={device} ...")
    processor = AutoProcessor.from_pretrained(_COHERE_MODEL_ID)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        _COHERE_MODEL_ID, torch_dtype=dtype
    ).to(device)
    model.eval()

    def _run(audio_path: str) -> str:
        data, sr = sf.read(audio_path, dtype="float32")
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        inputs = processor(data, sampling_rate=sr, return_tensors="pt", language="ja")
        inputs = inputs.to(device, dtype=dtype)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)
        return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    return _run


RUNNERS = {
    "kotoba": build_kotoba_runner,
    "nue": build_nue_runner,
    "qwen": build_qwen_runner,
    "granite": build_granite_runner,
    "gemma": build_gemma_runner,
    "cohere": build_cohere_runner,
}


# ============================================================
# メイン
# ============================================================
def main():
    if len(sys.argv) < 3:
        print("使用方法: python batch_benchmark.py <モデルエイリアス> <音声ディレクトリ> [出力CSVパス]")
        print("モデル: kotoba | nue | qwen | granite | gemma | cohere")
        sys.exit(1)

    model_alias = sys.argv[1].lower()
    audio_dir = sys.argv[2]
    output_csv = sys.argv[3] if len(sys.argv) > 3 else f"batch_{model_alias}_result.csv"

    if model_alias not in RUNNERS:
        print(f"エラー: 未対応モデル: {model_alias}")
        sys.exit(1)

    if not os.path.isdir(audio_dir):
        print(f"エラー: ディレクトリが存在しません: {audio_dir}")
        sys.exit(1)

    wav_paths = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    if not wav_paths:
        print(f"エラー: wav ファイルが見つかりません: {audio_dir}")
        sys.exit(1)

    # モデル情報 & ランナー構築
    model_info = get_model_info(model_alias)
    print(f"=== {model_info['name']} ===")
    print(f"音声ファイル数: {len(wav_paths)}")

    load_start = time.time()
    runner = RUNNERS[model_alias]()
    load_time = time.time() - load_start
    print(f"[{model_alias}] model load: {load_time:.2f}s\n")

    process = psutil.Process(os.getpid())

    rows: List[dict] = []
    total_proc_time = 0.0

    for wav in wav_paths:
        stem = os.path.splitext(os.path.basename(wav))[0]
        gt_path = os.path.join(audio_dir, f"{stem}.txt")
        if not os.path.exists(gt_path):
            print(f"SKIP {stem}: 正解テキストなし ({gt_path})")
            continue
        gt = load_ground_truth(gt_path)

        cpu_samples: list = []
        t0 = time.time()
        _cpu_sample(process, cpu_samples)

        try:
            hyp = runner(wav)
        except Exception as e:
            print(f"ERROR {stem}: {e}")
            hyp = ""

        _cpu_sample(process, cpu_samples)
        proc_time = time.time() - t0
        total_proc_time += proc_time

        cer = calculate_cer(gt, hyp) if hyp else 100.0
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
        max_cpu = max(cpu_samples) if cpu_samples else 0

        print(f"{stem}: {proc_time:.2f}s  CER={cer:.2f}%  => {hyp[:60]}")

        rows.append({
            "file": stem,
            "processing_time_s": round(proc_time, 3),
            "cer_percent": round(cer, 2),
            "avg_cpu_percent": round(avg_cpu, 2),
            "max_cpu_percent": round(max_cpu, 2),
            "ground_truth": gt,
            "recognized": hyp,
        })

    # CSV 出力
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "processing_time_s",
                "cer_percent",
                "avg_cpu_percent",
                "max_cpu_percent",
                "ground_truth",
                "recognized",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    # サマリー
    if rows:
        n = len(rows)
        mean_cer = sum(r["cer_percent"] for r in rows) / n
        mean_time = sum(r["processing_time_s"] for r in rows) / n
        print("\n=== サマリー ===")
        print(f"モデル: {model_info['name']} ({model_alias})")
        print(f"ファイル数: {n}")
        print(f"モデルロード時間: {load_time:.2f}s")
        print(f"合計認識時間: {total_proc_time:.2f}s")
        print(f"平均 CER: {mean_cer:.2f}%")
        print(f"平均 処理時間/ファイル: {mean_time:.2f}s")
        print(f"結果 CSV: {output_csv}")


if __name__ == "__main__":
    main()
