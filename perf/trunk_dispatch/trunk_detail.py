#!/usr/bin/env python3
"""In-model trunk sub-stage split: template embedder vs MSA module vs 48-block Pairformer.

Extends perf/stage_split_298/stage_split.py (same synced-both-sides protocol) with three
more boundaries inside Trunk.__call__, so the trunk's 25-46 s at 298 aa can be attributed
to the Pairformer stack (the unit dispatch_probe.py benches standalone) rather than assumed.
Adds ~30 syncs per fold on top of stage_split's ~8; the perturbation stays negligible
because every added boundary is called 10x per fold, not per block.

    PYTHONPATH=$PWD TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_HOLDER=worker:... \
      python3 perf/trunk_dispatch/trunk_detail.py --model protenix-v2 \
        --target examples/prot300.yaml --msa-a3m scripts/gpu_vs_tt/fixtures/prot300.a3m \
        --label "298 aa" --repeat 2 --out /tmp/trunk_detail_p298.json
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "stage_split", REPO_ROOT / "perf" / "stage_split_298" / "stage_split.py")
SS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SS)


def install_trunk_patches():
    import tt_bio.protenix as P
    import tt_bio.tenstorrent as T
    T.Pairformer.__call__ = SS.timed("pf_stack", T.Pairformer.__call__)
    P.Trunk._msa = SS.timed("trunk_msa", P.Trunk._msa)
    P.Trunk._template = SS.timed("trunk_template", P.Trunk._template)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["protenix-v2", "opendde"])
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--msa-a3m", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    SS.install_patches()
    install_trunk_patches()

    spec = importlib.util.spec_from_file_location(
        "tt_baseline", REPO_ROOT / "scripts" / "gpu_vs_tt" / "tt_baseline.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)

    msa_dir = Path("~/.cache/tt-bio-gpu-vs-tt/msa").expanduser()
    res = tb.measure(args.model, args.repeat, msa_dir, args.out,
                     args.target, args.msa_a3m, args.label)
    out = dict(res)
    out["folds"] = SS.FOLD_MARKS
    args.out.write_text(json.dumps(out, indent=2, default=str))

    warm = SS.FOLD_MARKS[-1]
    print(f"\n=== {args.model} {args.label} WARM (total {warm['total_s']}s) ===", flush=True)
    print("  -- top-level stages --", flush=True)
    for n, (c, t) in sorted(warm["stages"].items(), key=lambda kv: -kv[1][1]):
        print(f"  {n:18s} n={c:<5d} {t:8.3f}s {100*t/warm['total_s']:5.1f}%", flush=True)
    print("  -- gross (any depth) --", flush=True)
    for n, (c, t) in sorted(warm["gross"].items(), key=lambda kv: -kv[1][1]):
        print(f"  {n:18s} n={c:<5d} {t:8.3f}s {100*t/warm['total_s']:5.1f}%", flush=True)
    g = warm["gross"]
    if "pf_stack" in g:
        c, t = g["pf_stack"]
        print(f"\n  pf_stack: {c} calls, {t:.3f}s -> {1e3*t/(c*48):.2f} ms per 48-block-stack block",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
