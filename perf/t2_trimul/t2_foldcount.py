#!/usr/bin/env python3
"""Count, in a real 298 aa protenix-v2 fold, how many times each op I own actually runs.

The ledger converts a block-level ms to ms/fold by multiplying by 480 (48 Pairformer blocks x 10
recycles). `perfwar-pairformer-matmul-dataflow` counted 524 block executions instead. Neither is
assumed here: every call site is counted directly, per fold, warm.

    python3 perf/t2_trimul/t2_foldcount.py --target examples/prot300.yaml \
        --msa-a3m scripts/gpu_vs_tt/fixtures/prot300.a3m --out perf/t2_trimul/foldcount.json
"""
import argparse
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

COUNTS = OrderedDict()
FOLDS = []
SHAPES = {}


def counted(name, fn, shape_of=None):
    def wrapper(*a, **k):
        COUNTS[name] = COUNTS.get(name, 0) + 1
        if shape_of:
            try:
                key = f"{name}|{shape_of(*a, **k)}"
                SHAPES[key] = SHAPES.get(key, 0) + 1
            except Exception:
                pass
        return fn(*a, **k)
    return wrapper


def install():
    import tt_bio.tenstorrent as T
    import tt_bio.protenix as P

    T.TriangleMultiplication.__call__ = counted(
        "TriangleMultiplication", T.TriangleMultiplication.__call__,
        shape_of=lambda self, x, *a, **k: list(x.shape))
    T.TriangleAttention.__call__ = counted(
        "TriangleAttention", T.TriangleAttention.__call__,
        shape_of=lambda self, x, *a, **k: list(x.shape))
    T.PairformerLayer.__call__ = counted(
        "PairformerLayer", T.PairformerLayer.__call__,
        shape_of=lambda self, s_, z_, *a, **k: list(z_.shape))
    T._trimul_out_proj = counted("trimul_out_proj", T._trimul_out_proj,
                                 shape_of=lambda x, *a, **k: list(x.shape))
    T._pair_proj_linear = counted("pair_proj_linear", T._pair_proj_linear,
                                  shape_of=lambda x, w, *a, **k: [list(x.shape), list(w.shape)])
    P.Trunk.__call__ = counted("Trunk", P.Trunk.__call__)

    orig = P.Protenix.fold

    def fold(*a, **k):
        COUNTS.clear()
        SHAPES.clear()
        try:
            return orig(*a, **k)
        finally:
            FOLDS.append({"counts": dict(COUNTS), "shapes": dict(SHAPES)})
    P.Protenix.fold = fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--msa-a3m", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    install()
    spec = importlib.util.spec_from_file_location(
        "tt_baseline", REPO_ROOT / "scripts" / "gpu_vs_tt" / "tt_baseline.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    msa_dir = Path("~/.cache/tt-bio-gpu-vs-tt/msa").expanduser()
    res = tb.measure("protenix-v2", args.repeat, msa_dir, Path("/tmp/t2_foldcount_base.json"),
                     args.target, args.msa_a3m, "298 aa")
    out = {"measure": res, "folds": FOLDS}
    args.out.write_text(json.dumps(out, indent=2, default=str))
    for i, f in enumerate(FOLDS):
        print(f"--- fold {i} ({'cold' if i == 0 else 'warm'})", flush=True)
        for k, v in f["counts"].items():
            print(f"   {k:26s} {v}", flush=True)
        for k, v in sorted(f["shapes"].items(), key=lambda kv: -kv[1]):
            print(f"   {k}  x{v}", flush=True)
    print("WROTE " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
