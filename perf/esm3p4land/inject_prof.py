"""Where the per-loop ESMFold2 injection spends its time (spd-esmregress).

Runs fold_ab.py unchanged with the injection's pieces wrapped in device-synchronised timers:
the LM dropout draw (host bernoulli + upload + device multiply), the LM encoder, and the folding
trunk. The syncs remove host/device overlap, so the sum is an upper bound on what each piece costs
in an unsynced fold; fold_ab's own fold_s is printed beside it. Same arguments as fold_ab.py.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ttnn  # noqa: E402

import tt_bio.esmfold2_runtime as RT  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402

TIMES = defaultdict(list)


def timed(name, fn):
    def wrap(*a, **kw):
        dev = T.get_device()
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        out = fn(*a, **kw)
        ttnn.synchronize_device(dev)
        TIMES[name].append(time.perf_counter() - t)
        return out
    return wrap


_bern = RT.torch.Tensor.bernoulli_


def bern(self, *a, **kw):
    t = time.perf_counter()
    out = _bern(self, *a, **kw)
    TIMES["host_bernoulli"].append(time.perf_counter() - t)
    return out


RT.torch.Tensor.bernoulli_ = bern
RT._lm_dropout = timed("lm_dropout", RT._lm_dropout)
RT._trunk_on_device = timed("lm_encoder", RT._trunk_on_device)

import fold_ab  # noqa: E402

if __name__ == "__main__":
    try:
        fold_ab.main()
    finally:
        rep = {k: {"n": len(v), "sum_s": round(sum(v), 3), "mean_s": round(sum(v) / len(v), 4)}
               for k, v in TIMES.items()}
        print("INJECT_PROF " + json.dumps(rep), flush=True)
