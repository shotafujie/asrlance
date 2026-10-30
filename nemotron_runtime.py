"""Offline Nemotron runners. Model paths and shared lock are explicitly configured by the operator."""
import atexit
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import threading
import time
import wave

MODEL_ID = 'nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B'
REVISION = '9d8cd6b79bfdd427b6558a5fdbf76fcaf8c75341'
STOCK_PROMPT = 'Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only.'
JAPANESE_PROMPT = STOCK_PROMPT.replace('as spoken', 'Japanese')
SHORT_PROMPT = 'Audio transcription. Language: Japanese. Include only words actually spoken, including fillers and repetitions. Do not translate. Output nothing if no speech is audible. Output: transcript only.'
ALIASES = ('nemotron-stock', 'nemotron-mlx8')
ASSETS = Path(__file__).parent / 'nemotron'


def prompt_for(alias, samples):
    if alias == 'nemotron-stock': return STOCK_PROMPT
    return SHORT_PROMPT if samples <= 32000 else JAPANESE_PROMPT


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()


def verify_model(path, alias):
    path = Path(path)
    if path.is_symlink() or not path.is_dir(): raise ValueError('Use a regular local model directory')
    if alias == 'nemotron-mlx8':
        manifest = json.loads((ASSETS / 'mlx8-manifest.json').read_text())
        if any(p.is_symlink() or not p.is_file() for p in path.iterdir()): raise ValueError('Model inventory must be regular files')
        if {p.name for p in path.iterdir()} != set(manifest['files']): raise ValueError('Model inventory mismatch')
        for name, expected in manifest['files'].items():
            if Path(name).name != name or digest(path / name) != expected: raise ValueError('Model checksum mismatch')
        if json.loads((path / 'config.json').read_text())['quantization'] != {'group_size':64,'bits':8,'mode':'affine'}: raise ValueError('Quantization mismatch')
    elif alias == 'nemotron-stock':
        manifest = json.loads((ASSETS / 'source-manifest.json').read_text())
        if digest(path / 'config.json') != manifest['config_sha256']: raise ValueError('Source config mismatch')
        shards = {p.name:p for p in path.glob('*.safetensors')}
        if set(shards) != set(manifest['model_headers']): raise ValueError('Source shard inventory mismatch')
        for name, p in shards.items():
            with p.open('rb') as f:
                length = struct.unpack('<Q', f.read(8))[0]
                if length > 100*1024**2: raise ValueError('Invalid safetensors header')
                header = f.read(length)
            expected = manifest['model_headers'][name]
            if p.stat().st_size != expected['size'] or hashlib.sha256(header).hexdigest() != expected['header_sha256']: raise ValueError('Source header mismatch')
    else: raise ValueError('Unsupported Nemotron alias')


def preflight():
    import psutil
    workers = {'nemotron_mlx_worker.py','nemotron_worker.py','batch_benchmark.py','nemotron_benchmark.py'}
    own = psutil.Process()
    ignored = {own.pid, *(p.pid for p in own.parents())}
    for p in psutil.process_iter(['pid','cmdline']):
        if p.pid in ignored: continue
        args = p.info['cmdline'] or []
        if any(Path(a).name in workers or a.startswith('voicetype.') or Path(a).name == 'voicetype' for a in args):
            raise RuntimeError('Another model app/evaluation is active; retry after its normal owner releases it')
    if psutil.virtual_memory().available < 16*2**30: raise RuntimeError('At least16GiB available memory required')


def acquire_lease(path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink(): raise ValueError('Unsafe lock directory')
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid(): raise ValueError('Unsafe lock file')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return os.fdopen(fd, 'r+')
    except BaseException:
        os.close(fd)
        raise


def read_audio(path):
    import numpy as np
    with wave.open(str(path)) as w:
        if (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype()) != (16000,1,2,'NONE'): raise ValueError('Expected16kHz mono PCM16 WAV')
        audio = np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)/32768
    if len(audio) < 320: raise ValueError('Audio must contain at least320 samples')
    return audio


def resource_watch(stop, initial_swap):
    import psutil
    while not stop.wait(.1):
        try:
            if psutil.virtual_memory().available < 16*2**30 or psutil.swap_memory().used > initial_swap+512*2**20:
                os.write(2,b'Nemotron resource guard stopped this evaluation\n');os._exit(75)
        except BaseException: os._exit(75)


class Runner:
    def __init__(self, alias):
        if alias not in ALIASES: raise ValueError('Unknown alias')
        env = 'ASRLANCE_NEMOTRON_MLX8_MODEL' if alias == 'nemotron-mlx8' else 'ASRLANCE_NEMOTRON_STOCK_MODEL'
        if not os.environ.get(env): raise ValueError(f'Set {env} to the verified local model directory')
        if not os.environ.get('ASRLANCE_MODEL_LOCK'): raise ValueError('Set ASRLANCE_MODEL_LOCK to the shared model lock used by your apps')
        self.alias=alias; self.closed=False; self.stop=threading.Event(); self.lease=None
        self.model_path=Path(os.environ[env]).expanduser()
        preflight()
        self.lease=acquire_lease(os.environ['ASRLANCE_MODEL_LOCK'])
        try:
            preflight(); verify_model(self.model_path,alias)
            os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',PYTORCH_MPS_LOW_WATERMARK_RATIO='.65',PYTORCH_MPS_HIGH_WATERMARK_RATIO='.8')
            import psutil
            initial_swap=psutil.swap_memory().used
            self.guard_thread=threading.Thread(target=resource_watch,args=(self.stop,initial_swap),daemon=True);self.guard_thread.start()
            if alias == 'nemotron-mlx8':
                import mlx.core as mx
                from mlx_audio.stt import load
                mx.set_cache_limit(2**30);mx.set_memory_limit(50*2**30)
                self.mx=mx;self.model=load(str(self.model_path),strict=True)
            else:
                import torch
                from transformers import AutoModelForMultimodalLM,AutoProcessor,core_model_loading
                if not torch.backends.mps.is_available(): raise RuntimeError('Stock runner requires Apple MPS')
                torch.manual_seed(0);torch.set_num_threads(8);self.torch=torch
                self.processor=AutoProcessor.from_pretrained(self.model_path,local_files_only=True,trust_remote_code=False)
                original=core_model_loading.set_param_for_module
                def assign(model,name,param,*a,**kw):
                    result=original(model,name,param,*a,**kw)
                    if param.device.type=='mps' and param.numel()>10_000_000: torch.mps.synchronize();torch.mps.empty_cache()
                    return result
                core_model_loading.set_param_for_module=assign
                try: self.model=AutoModelForMultimodalLM.from_pretrained(self.model_path,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map='mps',attn_implementation='eager').eval()
                finally: core_model_loading.set_param_for_module=original
            self.sync();atexit.register(self.close)
        except BaseException:
            self.close();raise

    def sync(self):
        if self.alias=='nemotron-mlx8': self.mx.synchronize()
        else: self.torch.mps.synchronize()

    def __call__(self,path):
        if self.closed: raise RuntimeError('Runner closed')
        audio=read_audio(path);prompt=prompt_for(self.alias,len(audio))
        with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
            if self.alias=='nemotron-mlx8':
                result=self.model.generate(audio,prompt=prompt,temperature=0,max_tokens=256,enable_thinking=False)
                capped=result.generation_tokens>=256;text=result.text
            else:
                inputs=self.processor.apply_transcription_request(audio,prompt=prompt,return_tensors='pt').to('mps')
                with self.torch.inference_mode(): result=self.model.generate(**inputs,max_new_tokens=256,do_sample=False)
                capped=result.shape[1]-inputs.input_ids.shape[1]>=256
                text=self.processor.decode(result[:,inputs.input_ids.shape[1]:].cpu(),skip_special_tokens=True)[0].strip()
        self.sync()
        if capped: raise RuntimeError('Generation token cap reached; run is not score eligible')
        return text

    def close(self):
        if self.closed: return
        self.closed=True; self.stop.set()
        if hasattr(self,'guard_thread'): self.guard_thread.join(timeout=1)
        try:
            if hasattr(self,'model'): del self.model
            if hasattr(self,'processor'): del self.processor
            if hasattr(self,'mx'): self.mx.synchronize();self.mx.clear_cache()
            if hasattr(self,'torch'): self.torch.mps.synchronize();self.torch.mps.empty_cache()
        finally:
            if self.lease is not None: self.lease.close();self.lease=None


def recognize_file(path,alias):
    import psutil
    runner=Runner(alias)
    try:
        process=psutil.Process();process.cpu_percent(None);start=time.perf_counter()
        text=runner(path);elapsed=time.perf_counter()-start;cpu=process.cpu_percent(None)
        return text,elapsed,cpu,cpu
    finally: runner.close()


def run_batch(alias,audio_dir,output_csv,builder=Runner):
    """Numeric-only output; failed/capped inference never becomes a scored empty string."""
    import csv
    import statistics
    import tempfile
    from benchmark import calculate_cer
    paths=sorted(Path(audio_dir).glob('*.wav'));output=Path(output_csv)
    if output.exists(): raise ValueError('Refusing to overwrite existing results')
    if not output.parent.is_dir(): raise ValueError('Create the result directory before running')
    if not paths or any(not p.with_suffix('.txt').is_file() for p in paths): raise ValueError('Expected WAV/reference pairs')
    # Validate format and paired references before loading a large model.
    for p in paths: read_audio(p)
    started=time.perf_counter();runner=builder(alias)
    try:
        load_s=time.perf_counter()-started
        with tempfile.TemporaryDirectory() as d:
            silence=Path(d)/'silence.wav'
            with wave.open(str(silence),'wb') as w:w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(bytes(32000))
            start=time.perf_counter()
            for _ in range(2): runner(silence)
            warmup_s=time.perf_counter()-start
        rows=[]
        for i,p in enumerate(paths,1):
            ref=p.with_suffix('.txt').read_text().strip();start=time.perf_counter();hyp=runner(p);elapsed=time.perf_counter()-start
            with wave.open(str(p)) as w:duration=w.getnframes()/16000
            score=calculate_cer(ref,hyp)
            rows.append({'case':i,'cer_percent':score,'processing_time_s':elapsed,'audio_s':duration})
        # x mode prevents overwriting a file created concurrently after the initial check.
        with output.open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print(json.dumps({'alias':alias,'status':'COMPLETE','cases':len(rows),'macro_cer':statistics.mean(r['cer_percent'] for r in rows),'mean_inference_s':statistics.mean(r['processing_time_s'] for r in rows),'rtf':sum(r['processing_time_s'] for r in rows)/sum(r['audio_s'] for r in rows),'load_s':load_s,'warmup_s':warmup_s,'postprocessing':'none','model_revision':REVISION}))
    finally: runner.close()
