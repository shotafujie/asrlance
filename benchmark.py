import sys
import time
import psutil
import jiwer
from fileRecognizer import recognize_audio

def load_ground_truth(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()

def measure_performance(audio_path, ground_truth_path, model_alias, model_name, out_file):
    ground_truth = load_ground_truth(ground_truth_path)
    cpu_log = []
    start = time.time()
    def cpu_mon():
        import threading
        flag = [False]
        def mon():
            while not flag[0]:
                cpu_log.append(psutil.cpu_percent(interval=0.1))
        t = threading.Thread(target=mon)
        t.start()
        return flag, t
    flag, t = cpu_mon()
    text = recognize_audio(audio_path, model_alias)
    flag[0] = True
    t.join()
    elapsed = time.time() - start
    avg_cpu = sum(cpu_log)/len(cpu_log) if cpu_log else 0
    max_cpu = max(cpu_log) if cpu_log else 0
    cer_raw = jiwer.cer(ground_truth, text)
    cer_percent = cer_raw * 100  # パーセント表記
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"ファイル: {audio_path}\n")
        f.write(f"モデル: {model_name} ({model_alias})\n")
        f.write(f"呼称: large-v3-turbo（ローカル）\n")
        f.write(f"処理時間: {elapsed:.2f}秒\n")
        f.write(f"平均CPU: {avg_cpu:.2f}% 最大CPU: {max_cpu:.2f}%\n")
        f.write(f"CER（文字誤り率）: {cer_percent:.2f}%\n")
        f.write(f"認識テキスト:\n{text}\n")
    print(f"{model_name} の結果は {out_file} に保存されました。")
    return {
        "model": model_name, "alias": model_alias,
        "cer_raw": cer_raw,
        "cer_percent": cer_percent,
        "sec": elapsed, "cpu_avg": avg_cpu, "cpu_max": max_cpu,
        "text": text, "log": out_file
    }

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python benchmark.py <audio_file> <ground_truth_file> <model_alias> [output_log_file]")
        sys.exit(1)
    audio_file = sys.argv[1]
    ground_truth_file = sys.argv[2]
    model_alias = sys.argv[3]
    if model_alias == "mlx":
        model_name = "MLX Whisper Large V3 Turbo"
    elif model_alias == "whisper":
        model_name = "OpenAI Whisper Large V3 Turbo OSS"
    else:
        model_name = f"UnknownModel({model_alias})"
    output_log_file = sys.argv[4] if len(sys.argv) > 4 else f"{model_alias}_log.txt"
    measure_performance(audio_file, ground_truth_file, model_alias, model_name, output_log_file)

