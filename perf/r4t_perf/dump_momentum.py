"""r4t-perf (D34): dump the exact matrices upstream's GroupedMuon feeds to Newton-Schulz on a FRESH step.

Runs upstream's own step through r4t-accuracy's curve_upstream.py (model, data, loss, clip, optimizers untouched),
starting a fresh model at plan step K (a different example) with seed offset K (different t and noise), and
stops right after the first Muon step. Saves every NS input as bf16 (upstream's first NS op is .bfloat16(), so
bf16 is lossless for both host and device paths), grouped by shape, plus names/order.
usage: dump_momentum.py K OUT.pt -- <curve_upstream args>
"""
import sys, os, json, torch
K, OUT = int(sys.argv[1]), sys.argv[2]
rest = sys.argv[sys.argv.index("--") + 1:]
sys.path.insert(0, os.environ["CURVE_DIR"])
import curve_upstream as cu
from tt_bio.train.curve import CurvePlan
import rfproteina.utils.grouped_muon as gm

_step = CurvePlan.step
CurvePlan.step = lambda self, i: _step(self, i + K)
got = []
_orth = gm._orthogonalize
def orth(updates, *a, **k):
    got.append(updates.detach().to(torch.bfloat16).clone())
    return _orth(updates, *a, **k)
gm._orthogonalize = orth
_gstep = gm.GroupedMuon.step
def gstep(self, *a, **k):
    r = _gstep(self, *a, **k)
    torch.save(dict(K=K, groups=got, shapes=[list(g.shape) for g in got]), OUT)
    print(json.dumps(dict(K=K, n_groups=len(got), n_matrices=sum(g.shape[0] for g in got),
                          elems=sum(g.numel() for g in got), out=OUT)), flush=True)
    os._exit(0)
gm.GroupedMuon.step = gstep
sys.argv = ["curve_upstream.py"] + rest + ["--seed-offset", str(K)]
cu.main()
