"""Reproduce the recorded MLX8 conversion; explicit local source, no downloads."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import threading
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from nemotron_runtime import preflight,acquire_lease,verify_model,resource_watch
from formal_compat import install

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('source',type=Path);ap.add_argument('destination',type=Path);a=ap.parse_args()
    if not os.environ.get('ASRLANCE_MODEL_LOCK'):raise ValueError('Set shared ASRLANCE_MODEL_LOCK')
    if a.destination.exists():raise ValueError('Destination already exists')
    if not a.destination.parent.is_dir() or shutil.disk_usage(a.destination.parent).free < 68*2**30:raise ValueError('Need68GiB free including reserve')
    preflight()
    with acquire_lease(os.environ['ASRLANCE_MODEL_LOCK']):
        preflight();verify_model(a.source,'nemotron-stock')
        os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
        import psutil
        swap=psutil.swap_memory().used;stop=threading.Event()
        thread=threading.Thread(target=resource_watch,args=(stop,swap),daemon=True);thread.start()
        try:
            import mlx.core as mx
            from mlx_audio.stt import load
            from mlx_audio.convert import build_quant_predicate
            from mlx_audio.lm.convert import quantize_model,save_model,save_config
            mx.set_cache_limit(2**30);mx.set_memory_limit(90*2**30);install()
            model=load(str(a.source),strict=True);config=json.loads((a.source/'config.json').read_text())
            model,config=quantize_model(model,config,64,8,quant_predicate=build_quant_predicate(model))
            a.destination.mkdir()
            for name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja','processor_config.json','generation_config.json','LICENSE','THIRD_PARTY_NOTICES.md']:
                if (a.source/name).exists():shutil.copy2(a.source/name,a.destination/name)
            save_model(a.destination,model,donate_model=True);config['model_type']='nemotron_transcribe';save_config(config,a.destination/'config.json')
            verify_model(a.destination,'nemotron-mlx8')
            print('MLX8 conversion matches published full-file hashes')
        finally:stop.set();thread.join(timeout=1)
if __name__=='__main__':main()
