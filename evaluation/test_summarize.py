import json,tempfile,unittest
from pathlib import Path
from evaluate import aggregate
from summarize import ALIASES,summarize
class SummaryContracts(unittest.TestCase):
 def fixture(self,root):
  rows=[dict(case=i,cohort='self_voice50' if i<=50 else 'koepus_test' if i<=72 else 'koepus_added28',samples=320,reference_chars=2,errors=i%2,cer=50.*(i%2),seconds=.01,empty_output=False,token_cap_hit=None) for i in range(1,101)]
  for alias in ALIASES:
   d=root/'runs'/alias;d.mkdir(parents=True);s=dict(status='COMPLETE',score_eligible=True,harness_sha256='harness',manifest_sha256='private',scorer_sha256='scorer',runner_sha256='runner',runtime_sha256='runtime',load_s=1,observed_rss_peak=100)
   for cohort in ['all','self_voice50','koepus_test','koepus_added28']:s[cohort]=aggregate(rows if cohort=='all' else [r for r in rows if r['cohort']==cohort])
   (d/'summary.json').write_text(json.dumps(s));(d/'numeric.json').write_text(json.dumps(rows))
 def test_complete_aggregate(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.fixture(root);s=summarize(root);self.assertEqual(len(s),10);self.assertEqual(s[0]['all']['macro_cer'],25.);self.assertTrue((root/'comparison.csv').exists())
 def test_modified_numeric_rejected(self):
  for key,value in [('cer',float('nan')),('errors',2),('reference_chars',3),('samples',321),('case',2)]:
   with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
    root=Path(d);self.fixture(root);p=root/'runs/qwen/numeric.json';rows=json.loads(p.read_text());rows[0][key]=value;p.write_text(json.dumps(rows))
    with self.assertRaises(ValueError):summarize(root)
 def test_changed_conditions_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.fixture(root);p=root/'runs/qwen/summary.json';s=json.loads(p.read_text());s['manifest_sha256']='different';p.write_text(json.dumps(s))
   with self.assertRaises(ValueError):summarize(root)
if __name__=='__main__':unittest.main()
