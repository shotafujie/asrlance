"""Verify private numeric runs; export aggregate comparison without utterance text."""
import argparse,csv,json,math
from pathlib import Path
from evaluate import aggregate,complete
ALIASES=['mlx','qwen17b','qwen','kotoba','nue','granite','cohere','gemma','nemotron-mlx8','nemotron-stock']
def require(condition,reason):
 if not condition:raise ValueError(reason)
def summarize(root):
 summaries=[];baseline=None;identity=None;table=[]
 for alias in ALIASES:
  directory=root/'runs'/alias
  s=json.loads((directory/'summary.json').read_text());rows=json.loads((directory/'numeric.json').read_text())
  require(s['status']=='COMPLETE' and s['score_eligible'],'incomplete configuration')
  signature=(s['harness_sha256'],s['manifest_sha256'],s['scorer_sha256'],s['runner_sha256'],s['runtime_sha256'])
  require(baseline is None or baseline==signature,'configuration hashes differ');baseline=signature
  require(complete(rows,[{'case':i} for i in range(1,101)]),'missing/duplicate/capped case')
  for r in rows:
   require(r['reference_chars']>0 and r['samples']>=320 and r['errors']>=0 and math.isfinite(r['cer']) and r['cer']>=0 and math.isfinite(r['seconds']) and r['seconds']>=0,'invalid numeric row')
   require(math.isclose(r['cer'],100*r['errors']/r['reference_chars'],rel_tol=1e-10,abs_tol=1e-10),'CER/edit mismatch')
  row_identity=sorted((r['case'],r['cohort'],r['samples'],r['reference_chars']) for r in rows)
  require(identity is None or identity==row_identity,'case identity differs between models');identity=row_identity
  for cohort,count in [('all',100),('self_voice50',50),('koepus_test',22),('koepus_added28',28)]:
   subset=rows if cohort=='all' else [r for r in rows if r['cohort']==cohort]
   require(len(subset)==count,'cohort count mismatch');a=aggregate(subset)
   for k,v in a.items():require(math.isclose(v,s[cohort][k],rel_tol=1e-10,abs_tol=1e-10),'aggregate mismatch')
   table.append(dict(alias=alias,cohort=cohort,**a,load_s=s['load_s'],rss_peak_bytes=s['observed_rss_peak'],mlx_peak_bytes=s.get('mlx_allocator_peak_bytes'),mps_driver_peak_bytes=s.get('mps_driver_observed_peak_bytes')))
  summaries.append(s)
 with (root/'comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
 # Private provenance: do not publish this file; only selected aggregates are public.
 (root/'verified-summary.json').write_text(json.dumps({'status':'VERIFIED','configurations':len(summaries),'cases_per_configuration':100,'total_inferences':1000,'common_hashes':baseline,'summaries':summaries},indent=2)+'\n')
 return summaries
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();summarize(args.root)
