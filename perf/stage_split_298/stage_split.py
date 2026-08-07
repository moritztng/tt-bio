#!/usr/bin/env python3
"""Synced stage split for the 298-aa TT-vs-GPU gap (planning-pass instrument).

Wraps the named stages of Protenix-v2 / OpenDDE `fold()` with
`ttnn.synchronize_device()` on BOTH sides of every timed region, so queued device
work is charged to the stage that issued it rather than to whichever later call
happens to block (the RFD3 inversion). Only ~8 syncs per fold, so the
perturbation is negligible and the numbers are directly comparable to the bare
`tt_baseline.py` wall-clock.

Stages are recorded only at nesting depth 0 (a `_to_host` inside `edm_sample`
is charged to the diffusion stage, not double-counted), plus a gross tally per
name for reference.

Usage (card pinned, lease held):

    TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_HOLDER=worker:... \
      python3 perf/stage_split_298/stage_split.py --model protenix-v2 \
        --target examples/prot300.yaml \
        --msa-a3m scripts/gpu_vs_tt/fixtures/prot300.a3m --label "298 aa" \
        --out /tmp/split_protenix_298.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEPTH = [0]
TOP = OrderedDict()      # depth-0 only
GROSS = OrderedDict()    # every call, any depth
FOLD_MARKS = []          # one dict per fold


def _sync():
    import ttnn
    import tt_bio.tenstorrent as T
    dev = T._device or T.get_device()
    ttnn.synchronize_device(dev)


def _record(store, name, dt):
    e = store.setdefault(name, [0, 0.0])
    e[0] += 1
    e[1] += dt


def timed(name, fn):
    def wrapper(*a, **k):
        _sync()
        t0 = time.perf_counter()
        DEPTH[0] += 1
        try:
            return fn(*a, **k)
        finally:
            DEPTH[0] -= 1
            _sync()
            dt = time.perf_counter() - t0
            _record(GROSS, name, dt)
            if DEPTH[0] == 0:
                _record(TOP, name, dt)
    return wrapper


def install_patches():
    """Patch the stage boundaries. Import order matters: patch module attributes
    before the model is constructed, and patch `edm_sample` on the module object
    (OpenDDE imports it inside `fold`, so the lookup happens at call time)."""
    import tt_bio.protenix as P
    import tt_bio.opendde as O

    P.edm_sample = timed("diffusion", P.edm_sample)
    P.Trunk.__call__ = timed("trunk", P.Trunk.__call__)
    P.Protenix._generate_relp = staticmethod(timed("relp_host", P.Protenix.__dict__["_generate_relp"].__func__))
    P.Protenix._to_host = staticmethod(timed("to_host", P.Protenix.__dict__["_to_host"].__func__))
    P.Protenix._diffusion_pair_cond = timed("diff_pair_cond", P.Protenix._diffusion_pair_cond)
    P.Protenix._plm_z_term = timed("plm_z_term", P.Protenix._plm_z_term)
    P.Protenix._atom_feat_inputs = timed("atom_feat_host", P.Protenix._atom_feat_inputs)
    P.ConfidenceHead.confidence = timed("confidence", P.ConfidenceHead.confidence)
    if hasattr(P.ConfidenceHead, "confidence_device"):
        P.ConfidenceHead.confidence_device = timed("confidence", P.ConfidenceHead.confidence_device)
    O.OpenDDE.expand_and_refine = timed("expand_refine", O.OpenDDE.expand_and_refine)
    O.StructuralTokenExpander.__call__ = timed("expander", O.StructuralTokenExpander.__call__)

    # Fold boundary: snapshot + reset the per-fold tallies.
    for cls, meth in ((P.Protenix, "fold"), (O.OpenDDE, "fold")):
        orig = getattr(cls, meth)

        def make(orig=orig):
            def fold(*a, **k):
                TOP.clear()
                GROSS.clear()
                _sync()
                t0 = time.perf_counter()
                try:
                    return orig(*a, **k)
                finally:
                    _sync()
                    total = time.perf_counter() - t0
                    top = {n: [c, round(t, 3)] for n, (c, t) in TOP.items()}
                    acc = sum(t for _, t in top.values())
                    FOLD_MARKS.append(dict(
                        total_s=round(total, 3),
                        stages=top,
                        gross={n: [c, round(t, 3)] for n, (c, t) in GROSS.items()},
                        unattributed_s=round(total - acc, 3),
                    ))
            return fold
        setattr(cls, meth, make())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["protenix-v2", "opendde"])
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--msa-a3m", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    install_patches()

    spec = importlib.util.spec_from_file_location(
        "tt_baseline", REPO_ROOT / "scripts" / "gpu_vs_tt" / "tt_baseline.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)

    msa_dir = Path("~/.cache/tt-bio-gpu-vs-tt/msa").expanduser()
    res = tb.measure(args.model, args.repeat, msa_dir, args.out,
                     args.target, args.msa_a3m, args.label)
    # measure() already wrote its own JSON; append the split alongside it.
    out = dict(res)
    out["folds"] = FOLD_MARKS          # [0] = cold, [1:] = warm
    args.out.write_text(json.dumps(out, indent=2, default=str))
    warm = FOLD_MARKS[1] if len(FOLD_MARKS) > 1 else FOLD_MARKS[-1]
    print(f"\n=== {args.model} {args.label} WARM stage split "
          f"(total {warm['total_s']}s) ===", flush=True)
    for n, (c, t) in sorted(warm["stages"].items(), key=lambda kv: -kv[1][1]):
        print(f"  {n:18s} n={c:<4d} {t:8.3f}s  {100*t/warm['total_s']:5.1f}%", flush=True)
    print(f"  {'unattributed':18s}          {warm['unattributed_s']:8.3f}s  "
          f"{100*warm['unattributed_s']/warm['total_s']:5.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
