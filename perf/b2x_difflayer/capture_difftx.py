#!/usr/bin/env python3
"""Dump the raw ttnn.graph capture of one settled DiffusionTransformerLayer call.

`perf/bioir_roofline/fold_bytes_512.py` reports one number per phase (214.45 MB for the
token DiT). This writes the whole captured node list out so the number can be itemised
per op and per tensor offline, which is what the BioIR comparison needs: their 54.8 MB is
a line-by-line count off `DiffusionTransformerLayer.forward`, so ours has to be counted
the same way before the 3.91x ratio means anything.

No timing here. The capture perturbs what it measures and the fold runs with a short
rollout, so nothing in this file is a performance number.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402

SEEN = defaultdict(int)
DUMP = {}
BUSY = {"on": False}
DEV = {"d": None}


def sig(prefix, args):
    parts = []
    for x in args:
        sh = getattr(x, "shape", None)
        if sh is not None:
            parts.append("x".join(str(int(d)) for d in sh))
    return prefix + "|" + ",".join(parts[:2])


def wrap(cls, prefix, on_call):
    orig = cls.__call__

    def call(self, *a, **k):
        s = sig(prefix, a)
        n = SEEN[s]
        SEEN[s] = n + 1
        want = (n == on_call) and s not in DUMP and not BUSY["on"]
        if want:
            BUSY["on"] = True
            ttnn.synchronize_device(DEV["d"])
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        r = orig(self, *a, **k)
        if want:
            ttnn.synchronize_device(DEV["d"])
            g = ttnn.graph.end_graph_capture()
            DUMP[s] = {
                "sig": s,
                "inputs": [
                    {"shape": [int(d) for d in x.shape], "dtype": str(x.dtype),
                     "mem": str(x.memory_config().buffer_type)}
                    for x in a if getattr(x, "shape", None) is not None
                ],
                "nodes": g,
            }
            BUSY["on"] = False
        return r

    cls.__call__ = call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import tt_baseline as B
    B.SAMPLING_STEPS = a.steps
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    import fold_ab_multi as FAM
    from tt_bio.main import _resolve_recycling_steps
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    FAM.patch_boltz2_cfg()

    fix = ROOT / "perf" / "size512" / "fixtures"
    tgt, a3m = fix / f"cdk2x2_{a.size}.yaml", fix / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold(
        "boltz2", ROOT / f".msa_bytes_{a.size}", tgt, a3m)
    DEV["d"] = T.get_device()

    wrap(T.DiffusionTransformerLayer, "difftx", 3)
    wrap(T.PairformerLayer, "pairformer", 3)

    fold_s, m = one_fold()
    print("FOLD %.2f s steps=%d shapes=%s" % (fold_s, a.steps, sorted(DUMP)), flush=True)

    Path(a.out).write_text(json.dumps(
        {"size": a.size, "steps": a.steps, "fold_s": round(fold_s, 3),
         "calls": list(DUMP.values())}))
    print("WROTE " + a.out)
    T.cleanup()


if __name__ == "__main__":
    main()
