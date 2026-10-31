"""Validate a private, fixed local evaluation manifest; never upload data."""
import hashlib,json,wave
from pathlib import Path
from collections import Counter
COUNTS={'all':100,'self_voice50':50,'koepus_test':22,'koepus_added28':28}
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def validate_manifest(root):
 root=Path(root);m=json.loads((root/'manifest.json').read_text());rows=m['rows']
 if m.get('status')!='FROZEN' or m.get('counts')!=COUNTS or len(rows)!=100:raise ValueError('manifest status/counts')
 if {r['case'] for r in rows}!=set(range(1,101)) or Counter(r['cohort'] for r in rows)!=Counter({k:v for k,v in COUNTS.items() if k!='all'}):raise ValueError('case/cohort membership')
 if len({r['audio_sha256'] for r in rows})!=100:raise ValueError('duplicate audio hash')
 for r in rows:
  name=r['file']
  if Path(name).name!=name or not name.endswith('.wav'):raise ValueError('unsafe filename')
  p=root/'data'/name;ref=p.with_suffix('.txt')
  if p.is_symlink() or ref.is_symlink() or digest(p)!=r['audio_sha256'] or digest(ref)!=r['reference_sha256']:raise ValueError('input identity')
  if not ref.read_text().strip():raise ValueError('empty reference')
  with wave.open(str(p)) as w:
   if (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype(),w.getnframes())!=(16000,1,2,'NONE',r['samples']) or r['samples']<320:raise ValueError('audio format/length')
 return m
