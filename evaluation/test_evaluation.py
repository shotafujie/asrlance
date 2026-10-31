import json,tempfile,unittest,wave
from pathlib import Path
from manifest import COUNTS,digest,validate_manifest
from evaluate import aggregate,complete
class EvaluationContracts(unittest.TestCase):
 def fixture(self,root):
  data=root/'data';data.mkdir();rows=[]
  for case in range(1,101):
   p=data/f'case_{case:03d}.wav'
   with wave.open(str(p),'wb') as w:w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(case.to_bytes(2,'little')*320)
   p.with_suffix('.txt').write_text('fixture\n');rows.append(dict(case=case,cohort='self_voice50' if case<=50 else 'koepus_test' if case<=72 else 'koepus_added28',file=p.name,samples=320,audio_sha256=digest(p),reference_sha256=digest(p.with_suffix('.txt'))))
  m={'status':'FROZEN','counts':COUNTS,'rows':rows};(root/'manifest.json').write_text(json.dumps(m));return m
 def test_good_fixed_set_and_content_integrity(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.fixture(root);self.assertEqual(len(validate_manifest(root)['rows']),100);(root/'data/case_001.txt').write_text('changed')
   with self.assertRaises(ValueError):validate_manifest(root)
 def test_invalid_membership_duplicate_and_length(self):
  for field,value in [('case',2),('cohort','koepus_added28'),('samples',321),('file','../escape.wav'),('audio_sha256','duplicate')]:
   with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
    root=Path(d);m=self.fixture(root)
    if field=='audio_sha256':value=m['rows'][1]['audio_sha256']
    m['rows'][0][field]=value;(root/'manifest.json').write_text(json.dumps(m))
    with self.assertRaises(ValueError):validate_manifest(root)
 def test_macro_micro_and_completeness(self):
  rows=[dict(case=1,cer=50.,errors=1,reference_chars=2,seconds=1,samples=16000,empty_output=False,token_cap_hit=None),dict(case=2,cer=10.,errors=1,reference_chars=10,seconds=3,samples=32000,empty_output=False,token_cap_hit=None)]
  a=aggregate(rows);self.assertEqual(a['macro_cer'],30);self.assertAlmostEqual(a['micro_cer'],100/6);self.assertAlmostEqual(a['rtf'],4/3);self.assertTrue(complete(rows,rows));self.assertFalse(complete(rows[:1],rows));self.assertFalse(complete([rows[0],rows[0]],rows));rows[1]['token_cap_hit']=True;self.assertFalse(complete(rows,rows))
if __name__=='__main__':unittest.main()
