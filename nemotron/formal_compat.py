"""Exact unused CTC key exclusion for recorded formal NVIDIA checkpoint only."""
import json
from pathlib import Path

def install(require_source_ctc=True):
    from mlx_audio.stt.models.nemotron_transcribe.nemotron_transcribe import Model
    expected=json.loads(Path(__file__).with_name('ctc_allowlist.json').read_text())
    original=Model.sanitize
    def sanitize(weights):
        actual={k for k in weights if k.startswith('ctc_timestamp.')}
        if require_source_ctc:
            if actual!=set(expected):raise ValueError('Formal CTC allowlist mismatch')
            for k in actual:
                if list(weights[k].shape)!=expected[k]['shape']:raise ValueError('Formal CTC shape mismatch')
        elif actual:
            raise ValueError('Converted model unexpectedly contains CTC weights')
        return original({k:v for k,v in weights.items() if k not in actual})
    Model.sanitize=staticmethod(sanitize)
