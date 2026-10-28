"""OpenFold3 / OpenBind-0 stage split around perf/spd/bench.py: where a fold's time goes, by stage.

    python perf/spd/of3_split.py <bench.py args...>

Runs bench.py unchanged in this process after wrapping the trunk's stages with a device sync on
each side. Per `OpenFold3.fold` call it appends one line to <out>/split.jsonl, in the order bench
writes bench.jsonl: fold_s, trunk_s, and inside the trunk msa_embed_s (the pristine `m`, host or
device), msa_module_s (4 MSA blocks x every cycle), template_s, pairformer_s (48 blocks x every
cycle); diffusion + confidence is fold_s - trunk_s. host_io_s is every to_torch / from_torch /
from_device / to_device call made during the fold, synced, wherever it sits (so it overlaps the
stages above), with its count. The syncs cost a little: quote fold time from a plain bench arm and
use this one only for the split.
"""
import json, runpy, sys, time
from pathlib import Path

import ttnn

import tt_bio
from tt_bio import openfold3_fold, openfold3_trunk
from tt_bio.tenstorrent import get_device

OUT = Path(sys.argv[sys.argv.index("--out") + 1])
acc, io = {}, [0.0, 0]
depth = [0]


def sync():
    ttnn.synchronize_device(get_device())


def timed(name, fn):
    def run(*a, **k):
        sync()
        t = time.perf_counter()
        r = fn(*a, **k)
        sync()
        acc[name] = acc.get(name, 0.0) + time.perf_counter() - t
        return r
    return run


def timed_io(fn):
    def run(*a, **k):
        if depth[0] == 0:
            return fn(*a, **k)
        t = time.perf_counter()
        r = fn(*a, **k)
        sync()
        io[0] += time.perf_counter() - t
        io[1] += 1
        return r
    return run


for n in ("to_torch", "from_torch", "from_device", "to_device"):
    setattr(ttnn, n, timed_io(getattr(ttnn, n)))
# msa_embed and the tensors' own uploads are module globals the trunk resolves at call time.
openfold3_trunk.msa_embed = timed("msa_embed_s", openfold3_trunk.msa_embed)

_init = openfold3_trunk.OF3Trunk.__init__


def init(self, *a, **k):
    _init(self, *a, **k)
    for attr in ("msa_module", "template", "pairformer"):
        setattr(self, attr, timed(f"{attr}_s", getattr(self, attr)))


openfold3_trunk.OF3Trunk.__init__ = init
_trunk_call = openfold3_trunk.OF3Trunk.__call__
openfold3_trunk.OF3Trunk.__call__ = lambda self, *a, **k: timed("trunk_s", _trunk_call)(self, *a, **k)
_fold = openfold3_fold.OpenFold3.fold


def fold(self, *a, **k):
    acc.clear()
    io[:] = [0.0, 0]
    depth[0] += 1
    try:
        r = timed("fold_s", _fold)(self, *a, **k)
    finally:
        depth[0] -= 1
    msa = k.get("msa_feat")
    rec = dict(acc, host_io_s=io[0], host_io_n=io[1],
               msa_shape=list(msa.shape) if msa is not None else None)
    with open(OUT / "split.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return r


openfold3_fold.OpenFold3.fold = fold
sys.argv = [str(Path(tt_bio.__file__).parents[1] / "perf/spd/bench.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
