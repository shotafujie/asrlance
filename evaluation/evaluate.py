"""Frozen100 local-only benchmark. Never logs private recognition/reference text."""
import argparse,ast,contextlib,csv,hashlib,importlib.metadata,json,os,statistics,sys,tempfile,threading,time,wave
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from manifest import digest,validate_manifest
import nemotron_runtime as n
MODEL_IDS={'mlx':'mlx-community/whisper-large-v3-turbo','qwen':'Qwen/Qwen3-ASR-0.6B','qwen17b':'Qwen/Qwen3-ASR-1.7B','kotoba':'kotoba-tech/kotoba-whisper-v2.0','nue':'yky-h/nue-asr','granite':'ibm-granite/granite-4.0-1b-speech','cohere':'CohereLabs/cohere-transcribe-03-2026','gemma':'unsloth/gemma-4-E2B-it-qat-q4_0-unquantized'}
PRECISION={'mlx':'MLX cached model','qwen':'FP16/MPS','qwen17b':'FP16/MPS','kotoba':'FP32/MPS','nue':'FP32/MPS','granite':'FP16/MPS','cohere':'FP16/MPS','gemma':'FP16/MPS','nemotron-stock':'BF16/MPS eager','nemotron-mlx8':'MLX8 affine/group64'}

def aggregate(rows):
 return {'n':len(rows),'macro_cer':statistics.mean(r['cer'] for r in rows),'micro_cer':100*sum(r['errors'] for r in rows)/sum(r['reference_chars'] for r in rows),'mean_s':statistics.mean(r['seconds'] for r in rows),'total_s':sum(r['seconds'] for r in rows),'audio_s':sum(r['samples'] for r in rows)/16000,'rtf':sum(r['seconds'] for r in rows)/(sum(r['samples'] for r in rows)/16000),'empty_outputs':sum(r['empty_output'] for r in rows)}

def complete(rows,manifest):
 return len(rows)==len(manifest) and {r['case'] for r in rows}=={r['case'] for r in manifest} and not any(r['token_cap_hit'] is True for r in rows)

def run(args):
 if args.out.exists():raise ValueError('refuse overwrite')
 os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_XET='1',ASRLANCE_MODEL_LOCK=str(args.lock))
 manifest=validate_manifest(args.frozen);items=manifest['rows']
 tree=ast.parse((REPO/'benchmark.py').read_text());fn=next(x for x in tree.body if getattr(x,'name','')=='calculate_cer');scope={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'scorer','exec'),scope);score=scope['calculate_cer']
 meta={'status':'STARTED','score_eligible':False,'alias':args.alias,'harness_sha256':digest(Path(__file__)),'manifest_sha256':digest(args.frozen/'manifest.json'),'scorer_sha256':digest(REPO/'benchmark.py'),'runner_sha256':digest(REPO/'batch_benchmark.py'),'runtime_sha256':digest(REPO/'nemotron_runtime.py'),'precision':PRECISION[args.alias],'postprocessing':'existing clean_gemma_transcription' if args.alias=='gemma' else 'none','timing':'batch1 read+preprocess+generate+decode+GPU sync; warmup and load separate','token_cap_observable':args.alias.startswith('nemotron'),'versions':{}}
 for pkg in ['torch','transformers','mlx','mlx-audio','mlx-whisper','qwen-asr','nue-asr','numpy','psutil']:
  try:meta['versions'][pkg]=importlib.metadata.version(pkg)
  except importlib.metadata.PackageNotFoundError:pass
 if args.alias in MODEL_IDS:
  model_id=MODEL_IDS[args.alias];cache=Path.home()/'.cache/huggingface/hub'/('models--'+model_id.replace('/','--'));revision=(cache/'refs/main').read_text().strip();snapshot=cache/'snapshots'/revision
  meta.update(model_id=model_id,revision=revision,config_sha256=digest(snapshot/'config.json'),weight_inventory={p.name:p.stat().st_size for p in snapshot.iterdir() if p.suffix in ['.safetensors','.bin','.npz']})
 else:meta.update(model_id=n.MODEL_ID,revision=n.REVISION,model_manifest_sha256=digest(n.ASSETS/('mlx8-manifest.json' if args.alias=='nemotron-mlx8' else 'source-manifest.json')),prompt_policy='stock as-spoken' if args.alias=='nemotron-stock' else 'Japanese; short<=32000 samples')
 n.preflight();args.out.mkdir(parents=True,mode=0o700);(args.out/'summary.json').write_text(json.dumps(meta,indent=2)+'\n');rows=[];stop=threading.Event();samples=[];runner=None;stage='preflight';lease=None;mx=None;torch=None
 import psutil
 initial_swap=psutil.swap_memory().used;probe=None
 def observe():
  while not stop.wait(.1):
   try:
    sample={'rss':psutil.Process().memory_info().rss,'swap':psutil.swap_memory().used,'available':psutil.virtual_memory().available}
    if probe:sample['mps_driver']=probe()
    samples.append(sample)
   except BaseException:pass
 try:
  if not args.alias.startswith('nemotron'):lease=n.acquire_lease(args.lock)
  with contextlib.nullcontext():
   n.preflight();threading.Thread(target=n.resource_watch,args=(stop,initial_swap),daemon=True).start();threading.Thread(target=observe,daemon=True).start()
   stage='load';start=time.perf_counter();mx=None;torch=None
   if args.alias in ('mlx','nemotron-mlx8'):
    import mlx.core as mx
    mx.reset_peak_memory();mx.set_cache_limit(2**30)
   else:
    import torch
    torch.set_num_threads(8);torch.manual_seed(0)
    if torch.backends.mps.is_available():probe=torch.mps.driver_allocated_memory
   def sync():
    if mx is not None:mx.synchronize()
    if torch is not None and torch.backends.mps.is_available():torch.mps.synchronize()
   from batch_benchmark import RUNNERS
   with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):runner=RUNNERS[args.alias]()
   sync();meta['load_s']=time.perf_counter()-start
   stage='warmup';warm=[]
   def invoke(path):
    with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):return runner(str(path))
   with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'zero.wav'
    with wave.open(str(p),'wb') as w:w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(bytes(32000))
    for _ in range(2):
     start=time.perf_counter();invoke(p);sync();warm.append(time.perf_counter()-start)
   meta['warmup_s']=warm
   for item in items:
    stage='case_'+str(item['case']);p=args.frozen/'data'/item['file'];ref=p.with_suffix('.txt').read_text().strip();sync();start=time.perf_counter();hyp=invoke(p);sync();elapsed=time.perf_counter()-start;c=score(ref,hyp);chars=len(ref.replace(' ','').replace('　',''))
    rows.append({'case':item['case'],'cohort':item['cohort'],'samples':item['samples'],'cer':c,'errors':round(c*chars/100),'reference_chars':chars,'seconds':elapsed,'empty_output':not bool(hyp.strip()),'token_cap_hit':False if args.alias.startswith('nemotron') else None})
    (args.out/'numeric.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({'alias':args.alias,'case':item['case'],'cer':round(c,3)}),flush=True)
   for cohort in ['all','self_voice50','koepus_test','koepus_added28']:
    part=rows if cohort=='all' else [r for r in rows if r['cohort']==cohort];meta[cohort]=aggregate(part)
   meta.update(status='COMPLETE',score_eligible=complete(rows,items),observed_rss_peak=max((x['rss'] for x in samples),default=None),swap_delta=max((x['swap'] for x in samples),default=initial_swap)-initial_swap,available_min=min((x['available'] for x in samples),default=None))
   if mx is not None:meta['mlx_allocator_peak_bytes']=mx.get_peak_memory()
   if probe:meta['mps_driver_observed_peak_bytes']=max([x.get('mps_driver',0) for x in samples]+[probe()])
   with (args.out/'numeric.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 except BaseException as e:
  meta.update(status='FAILED',error_type=type(e).__name__,stage=stage,completed_cases=len(rows),score_eligible=False)
  print(json.dumps({'alias':args.alias,'status':'FAILED','stage':stage,'error_type':type(e).__name__}),flush=True)
 finally:
  try:
   if hasattr(runner,'close'):runner.close()
  except BaseException as e:meta.update(status='FAILED',score_eligible=False,cleanup_error_type=type(e).__name__)
  finally:
   runner=None
   try:
    import gc
    gc.collect()
    if mx is not None:mx.synchronize();mx.clear_cache()
    if torch is not None and torch.backends.mps.is_available():torch.mps.synchronize();torch.mps.empty_cache()
   except BaseException as e:meta.update(status='FAILED',score_eligible=False,cleanup_error_type=type(e).__name__)
   finally:
    stop.set()
    try:
     if lease is not None:lease.close()
    finally:(args.out/'summary.json').write_text(json.dumps(meta,indent=2)+'\n')
 return 0 if meta['score_eligible'] else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('alias',choices=list(PRECISION));p.add_argument('--frozen',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);sys.exit(run(p.parse_args()))
