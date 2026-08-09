#!/usr/bin/env python3
"""Count Pairformer-block executions per trunk stage in a LIVE 298 aa protenix-v2 fold.

Settles the org's Q1: the ledger converts a block-level ms to ms/fold by multiplying by 480
(48 blocks x 10 recycles), while a live-fold `trimul.out_proj` call count implied 524. Neither
number was ever counted per stage. This counts, per stage (pf_stack / trunk_msa / trunk_template):

  * PairformerLayer.__call__ executions, with the padded z shape and whether the single track runs
  * TriangleMultiplication / TriangleAttention / Transition / OuterProductMean / PairWeightedAveraging
    executions
  * every ttnn.linear and ttnn.matmul call, bucketed by its immediate tt_bio call site

Counting only: no op is re-run, nothing is timed inside the wrappers beyond stage_split's existing
synced stage boundaries, so the stage denominators this prints are directly comparable to
perf/trunk_dispatch/trunk_detail.py's.

    PYTHONPATH=$PWD TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:... \
      python3 perf/msa_template/stage_counts.py --out perf/msa_template/counts_p298.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "stage_split", REPO_ROOT / "perf" / "stage_split_298" / "stage_split.py")
SS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SS)

STACK = ["other"]                    # innermost trunk stage
BLOCKS = []                          # one record per PairformerLayer call
MODCOUNT = Counter()                 # (stage, class) -> calls
OPCOUNT = Counter()                  # (stage, op, site) -> calls
FOLDS = [0]


def stage():
    return STACK[-1]


def stage_wrap(name, fn):
    def inner(*a, **k):
        STACK.append(name)
        try:
            return fn(*a, **k)
        finally:
            STACK.pop()
    return inner


def shp(t):
    try:
        return "x".join(str(d) for d in t.padded_shape) + "/" + "x".join(str(d) for d in t.shape)
    except Exception:                                            # noqa: BLE001
        return "?"


def install():
    import ttnn
    import tt_bio.protenix as P
    import tt_bio.tenstorrent as T

    T.Pairformer.__call__ = stage_wrap("pf_stack", T.Pairformer.__call__)
    P.Trunk._msa = stage_wrap("trunk_msa", P.Trunk._msa)
    P.Trunk._template = stage_wrap("trunk_template", P.Trunk._template)

    pl_orig = T.PairformerLayer.__call__

    def pl_call(self, s, z, *a, **k):
        BLOCKS.append({"fold": FOLDS[0], "stage": stage(), "z": shp(z),
                       "s": shp(s) if s is not None else None,
                       "transform_s": bool(self.transform_s),
                       "n_tri_heads": getattr(self.triangle_attention_start, "n_heads", None)})
        MODCOUNT[(stage(), "PairformerLayer")] += 1
        return pl_orig(self, s, z, *a, **k)

    T.PairformerLayer.__call__ = pl_call

    for cls in ("TriangleMultiplication", "TriangleAttention", "Transition",
                "OuterProductMean", "PairWeightedAveraging", "AttentionPairBias"):
        c = getattr(T, cls)
        orig = c.__call__

        def mk(orig=orig, nm=cls):
            def inner(self, *a, **k):
                MODCOUNT[(stage(), nm)] += 1
                return orig(self, *a, **k)
            return inner
        c.__call__ = mk()

    for nm in ("linear", "matmul"):
        fn = getattr(ttnn, nm)

        def mk(fn=fn, nm=nm):
            def inner(*a, **k):
                f = sys._getframe(1)
                site = f.f_code.co_filename.split("/")[-1] + ":" + str(f.f_lineno)
                OPCOUNT[(stage(), nm, site)] += 1
                return fn(*a, **k)
            return inner
        setattr(ttnn, nm, mk())

    # Fold boundary marker so the per-fold counts can be separated (cold fold vs warm).
    orig_fold = P.Protenix.fold

    def fold(*a, **k):
        try:
            return orig_fold(*a, **k)
        finally:
            FOLDS[0] += 1
    P.Protenix.fold = fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, default=REPO_ROOT / "examples" / "prot300.yaml")
    ap.add_argument("--msa-a3m", type=Path,
                    default=REPO_ROOT / "scripts" / "gpu_vs_tt" / "fixtures" / "prot300.a3m")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    SS.install_patches()
    import tt_bio.protenix as P
    import tt_bio.tenstorrent as T
    T.Pairformer.__call__ = SS.timed("pf_stack", T.Pairformer.__call__)
    P.Trunk._msa = SS.timed("trunk_msa", P.Trunk._msa)
    P.Trunk._template = SS.timed("trunk_template", P.Trunk._template)
    install()

    spec = importlib.util.spec_from_file_location(
        "tt_baseline", REPO_ROOT / "scripts" / "gpu_vs_tt" / "tt_baseline.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)

    msa_dir = Path("~/.cache/tt-bio-gpu-vs-tt/msa").expanduser()
    res = tb.measure("protenix-v2", args.repeat, msa_dir, args.out,
                     args.target, args.msa_a3m, "298 aa")

    last = FOLDS[0] - 1                       # the final (warm) fold
    warm_blocks = [b for b in BLOCKS if b["fold"] == last]
    per_stage = Counter(b["stage"] for b in warm_blocks)
    shapes = OrderedDict()
    for b in warm_blocks:
        key = (b["stage"], b["z"], b["transform_s"], b["n_tri_heads"])
        shapes[key] = shapes.get(key, 0) + 1

    warm = SS.FOLD_MARKS[-1]
    out = {
        "fold_total_s": warm["total_s"],
        "stages": {n: [c, t] for n, (c, t) in warm["stages"].items()},
        "gross": {n: [c, t] for n, (c, t) in warm["gross"].items()},
        "n_folds": FOLDS[0],
        "blocks_per_fold_by_stage": dict(per_stage),
        "blocks_total_per_fold": sum(per_stage.values()),
        "block_variants": [{"stage": k[0], "z_padded/logical": k[1], "transform_s": k[2],
                            "n_tri_heads": k[3], "calls": v} for k, v in shapes.items()],
        "module_calls_all_folds": {f"{s}|{c}": n for (s, c), n in sorted(MODCOUNT.items())},
        "op_calls_all_folds": {f"{s}|{o}|{t}": n for (s, o, t), n in sorted(OPCOUNT.items())},
        "latency": res,
    }
    args.out.write_text(json.dumps(out, indent=2, default=str))

    print(f"\n=== warm fold {warm['total_s']}s ===", flush=True)
    for n, (c, t) in sorted(warm["gross"].items(), key=lambda kv: -kv[1][1]):
        print(f"  {n:18s} n={c:<5d} {t:8.3f}s {100*t/warm['total_s']:5.1f}%", flush=True)
    print("\n=== PairformerLayer executions in ONE fold ===", flush=True)
    for s, n in per_stage.most_common():
        print(f"  {s:16s} {n}", flush=True)
    print(f"  TOTAL            {sum(per_stage.values())}", flush=True)
    print("\n=== block variants (padded/logical z, single track) ===", flush=True)
    for v in out["block_variants"]:
        print(f"  {v['stage']:16s} z={v['z_padded/logical']:28s} transform_s={v['transform_s']!s:5s} "
              f"heads={v['n_tri_heads']} calls={v['calls']}", flush=True)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
