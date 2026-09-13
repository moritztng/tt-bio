"""How many TriangleMultiplication calls a 512 aa fold actually makes.

The D1 prediction priced the deleted unsqueeze at the census's 280 PairformerLayer calls, i.e.
560 trimul calls. The measured saving is 1.69x that, so either the fold makes more trimul calls
than the pairformer trunk accounts for, or the op costs more than its kernel row. This counts.
"""
import os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perf" / "b2x-baseline-attrib"))
import torch
torch.set_grad_enabled(False)
import tt_bio.tenstorrent as T

CALLS = {}
for name in ("TriangleMultiplication", "PairformerLayer", "TriangleAttention"):
    cls = getattr(T, name)
    orig = cls.__dict__["__call__"]
    CALLS[name] = [0]
    def make(n, o):
        def w(self, *a, **k):
            CALLS[n][0] += 1
            return o(self, *a, **k)
        return w
    cls.__call__ = make(name, orig)

sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
import tt_baseline as B
from tt_bio.main import _resolve_recycling_steps
B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
B.SAMPLING_STEPS = 200
sys.path.insert(0, str(ROOT / "perf" / "other512"))
from fold_ab_multi import patch_boltz2_cfg
patch_boltz2_cfg()
T.get_device(trace_region_size=1 << 30)
fix = ROOT / "perf" / "size512" / "fixtures"
one_fold, meta, state = B.build_fold(
    "boltz2", Path("/home/ttuser/scratch/uod/.msa_512"),
    fix / "cdk2x2_512.yaml", fix / "cdk2x2_512.a3m")
t0 = time.perf_counter()
one_fold()
print("fold %.3f s" % (time.perf_counter() - t0))
for k, v in CALLS.items():
    print("%-26s %d calls" % (k, v[0]))
