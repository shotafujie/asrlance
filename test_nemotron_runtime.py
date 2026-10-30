import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
import nemotron_runtime as n


def fixtures(root):
    p=root/'test.wav'
    with wave.open(str(p),'wb') as w:w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(bytes(640))
    p.with_suffix('.txt').write_text('はい。')
    return p


class Contract(unittest.TestCase):
    def test_model_list_and_normal_batch_dispatch(self):
        import batch_benchmark as b
        import fileRecognizer as f
        for alias in n.ALIASES:
            self.assertIn(alias,f.get_available_models());self.assertIn(alias,b.RUNNERS)
            with patch('nemotron_runtime.Runner',return_value='fake') as load:self.assertEqual(b.RUNNERS[alias](),'fake');load.assert_called_once_with(alias)
            with patch('nemotron_runtime.recognize_file',return_value=('ok',1,0,0)) as run:
                self.assertEqual(f.recognize('synthetic.wav',alias)[0],'ok');run.assert_called_once_with('synthetic.wav',alias)
    def test_boundary_and_stock_prompt(self):
        self.assertEqual(n.prompt_for('nemotron-mlx8',32000),n.SHORT_PROMPT)
        self.assertEqual(n.prompt_for('nemotron-mlx8',32001),n.JAPANESE_PROMPT)
        self.assertEqual(n.prompt_for('nemotron-stock',1000),n.STOCK_PROMPT)
    def test_configuration_required_before_load(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'MODEL'):n.Runner('nemotron-mlx8')
        with patch.dict(os.environ,{'ASRLANCE_NEMOTRON_MLX8_MODEL':'not-a-model'},clear=True):
            with self.assertRaisesRegex(ValueError,'LOCK'):n.Runner('nemotron-mlx8')
    def test_lock_exclusion_and_release(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.lock'
            with n.acquire_lease(p):
                with self.assertRaises(BlockingIOError):n.acquire_lease(p)
            n.acquire_lease(p).close()
    def test_model_hash_and_inventory_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);assets=root/'assets';assets.mkdir();model=root/'model';model.mkdir()
            config={'quantization':{'group_size':64,'bits':8,'mode':'affine'}}
            (model/'config.json').write_text(json.dumps(config));(assets/'mlx8-manifest.json').write_text(json.dumps({'files':{'config.json':n.digest(model/'config.json')}}))
            with patch.object(n,'ASSETS',assets):
                n.verify_model(model,'nemotron-mlx8')
                (model/'extra').write_text('x')
                with self.assertRaises(ValueError):n.verify_model(model,'nemotron-mlx8')
                (model/'extra').unlink();(model/'config.json').write_text('{}')
                with self.assertRaises(ValueError):n.verify_model(model,'nemotron-mlx8')
    def test_batch_numeric_output_and_failure_not_scored(self):
        class Fake:
            closed=False
            def __call__(self,path):return 'はい'
            def close(self):self.closed=True
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixtures(root);out=root/'result.csv';runner=Fake()
            with contextlib.redirect_stdout(io.StringIO()) as output:n.run_batch('nemotron-mlx8',root,out,builder=lambda _:runner)
            self.assertTrue(runner.closed);self.assertNotIn('はい',out.read_text());self.assertAlmostEqual(json.loads(output.getvalue())['macro_cer'],100/3)
            with self.assertRaises(ValueError):n.run_batch('nemotron-mlx8',root,out,builder=lambda _:Fake())
            class Failure(Fake):
                def __call__(self,path):raise RuntimeError('token cap')
            bad=Failure()
            with self.assertRaises(RuntimeError):n.run_batch('nemotron-mlx8',root,root/'bad.csv',builder=lambda _:bad)
            self.assertTrue(bad.closed);self.assertFalse((root/'bad.csv').exists())
    def test_cli_routes_nemotron_numeric_batch(self):
        import batch_benchmark as b
        import sys
        with tempfile.TemporaryDirectory() as d:
            fixtures(Path(d))
            with patch.object(sys,'argv',['batch_benchmark.py','nemotron-mlx8',d,str(Path(d)/'result.csv')]),patch('nemotron_runtime.run_batch') as run:b.main();self.assertEqual(run.call_args.args[:3],('nemotron-mlx8',d,str(Path(d)/'result.csv')))
    def test_wrong_audio_format_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'wrong.wav'
            with wave.open(str(p),'wb') as w:w.setparams((1,2,8000,0,'NONE','not compressed'));w.writeframes(bytes(640))
            with self.assertRaises(ValueError):n.read_audio(p)

class Lifecycle(unittest.TestCase):
    def test_token_cap_rejected_by_actual_adapter(self):
        from types import SimpleNamespace
        r=n.Runner.__new__(n.Runner);r.closed=False;r.alias='nemotron-mlx8';r.mx=SimpleNamespace(synchronize=lambda:None)
        with tempfile.TemporaryDirectory() as d:
            p=fixtures(Path(d))
            r.model=SimpleNamespace(generate=lambda *a,**k:SimpleNamespace(text='はい',generation_tokens=256))
            with self.assertRaisesRegex(RuntimeError,'cap'):r(p)
            r.model=SimpleNamespace(generate=lambda *a,**k:SimpleNamespace(text='',generation_tokens=1))
            self.assertEqual(r(p),'')
    def test_guard_observation_error_fails_closed(self):
        from unittest.mock import Mock
        stop=Mock();stop.wait.return_value=False
        with patch('psutil.virtual_memory',side_effect=OSError('unavailable')),patch('nemotron_runtime.os._exit',side_effect=SystemExit(75)) as end:
            with self.assertRaises(SystemExit):n.resource_watch(stop,0)
            end.assert_called_once_with(75)
    def test_gpu_cleanup_failure_still_releases_lease(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import threading
        r=n.Runner.__new__(n.Runner);r.closed=False;r.stop=threading.Event();lease=Mock();r.lease=lease
        r.mx=SimpleNamespace(synchronize=Mock(side_effect=RuntimeError('device failure')))
        with self.assertRaises(RuntimeError):r.close()
        lease.close.assert_called_once();self.assertIsNone(r.lease)

if __name__=='__main__':unittest.main()
