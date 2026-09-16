#!/usr/bin/env python3
"""The bar a confidence-score change is scored against: the model's own seed-to-seed scatter.

`score_scalars.py` measures what TT_BIO_DEVICE_CONF_HEADS does to pTM/pAE/pDE. That number is
only readable against something. For a coordinate the campaign has one -- 0.60 A kill bar at
512 aa, against a 1.84 A seed floor -- but pTM is not an Angstrom and this lever moves no atom,
so the coordinate bar does not apply. The honest comparison is the same one the coordinate bar
is built from: re-run the model with a different diffusion seed, on the SAME fixture, same arm,
same process, and see how far its own confidence scores move. A lever whose shift is a small
fraction of that scatter is inside variation the model already has.

One arm throughout (the lever pinned OFF, which is the base of score_scalars' comparison), one
process, one device open, N seeds. `build_fold` bakes `seed` into the cfg dict it returns as
`meta["job_cfg"]`, and `one_fold` reads that same dict every fold, so the seed is changed by
writing it there -- no second fold driver.

Capture is imported from `score_scalars` rather than re-implemented: one definition of what a
confidence reading is, or the two numbers are not comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(HERE))
FIX = REPO / "perf" / "size512" / "fixtures"

import score_scalars as S  # noqa: E402  (capture/diff/SCALARS live there)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--flag", default="TT_BIO_DEVICE_CONF_HEADS")
    ap.add_argument("--flag-value", default="0", help="the arm the scatter is measured on")
    ap.add_argument("--target", type=Path, default=None)
    ap.add_argument("--a3m", type=Path, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    import torch
    torch.set_grad_enabled(False)
    import tt_baseline as B
    import tt_bio as _TB
    import tt_bio.boltz2 as boltz2
    from tt_bio.main import _resolve_recycling_steps

    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(REPO / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    assert a.flag not in os.environ, f"{a.flag} is set in the environment; this script owns it"
    os.environ[a.flag] = a.flag_value

    grabbed: list = []
    orig = boltz2.ConfidenceHeads.forward

    def wrapped(self, *args, **kw):
        out = orig(self, *args, **kw)
        grabbed.append(S.capture(out))
        return out

    boltz2.ConfidenceHeads.forward = wrapped

    tgt = a.target or FIX / f"cdk2x2_{a.size}.yaml"
    a3m = a.a3m or FIX / f"cdk2x2_{a.size}.a3m"
    msa_dir = HERE / f".msa_{a.tag or a.size}"
    one_fold, meta, _state = B.build_fold("boltz2", msa_dir, tgt, a3m)
    job_cfg = meta["job_cfg"]
    assert "seed" in job_cfg, "build_fold no longer exposes the fold's seed; this driver is stale"

    out = {"doc": __doc__,
           "env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "host": os.uname().nodename, "size": a.size, "target": str(tgt),
                   "arm": f"{a.flag}={a.flag_value}", "seeds": seeds,
                   "commit": os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip(),
                   "tt_bio_file": _TB.__file__,
                   "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS},
           "folds": []}

    print("=== cold fold (discarded) ===", flush=True)
    one_fold()
    grabbed.clear()

    keep = {}
    for s in seeds:
        job_cfg["seed"] = s
        t0 = time.perf_counter()
        _t, m = one_fold()
        assert grabbed, "the confidence head did not run"
        keep[s] = grabbed[-1]
        out["folds"].append({"seed": s, "wall_s": round(time.perf_counter() - t0, 4),
                             "plddt": m.get("plddt"),
                             **{k: v for k, v in keep[s].items() if not k.endswith("_")}})
        print(f"  seed={s}  plddt={m.get('plddt')}  "
              + " ".join(f"{k}={keep[s][k]}" for k in S.SCALARS if k in keep[s]), flush=True)

    # The scatter, in the two shapes the lever's reading comes in: a spread over the scalars,
    # and a mean-abs matrix distance against the first seed (same quantity score_scalars reports
    # for the arm, so the two are directly divisible).
    ref = keep[seeds[0]]
    spread = {}
    for k in S.SCALARS:
        vals = [keep[s][k] for s in seeds if k in keep[s]]
        if len(vals) < 2:
            continue
        spread[k] = {"values": vals, "min": min(vals), "max": max(vals),
                     "range": round(max(vals) - min(vals), 6),
                     "stdev": round(st.stdev(vals), 6)}
    pairs = {str(s): S.diff(ref, keep[s]) for s in seeds[1:]}
    out["scatter"] = spread
    out["vs_seed0"] = pairs
    # Negative control on the instrument itself: if the seeds did not actually change the fold,
    # a zero scatter would read as "the model is stable" when it means "the seed never took".
    moved = any(v["range"] > 0 for v in spread.values())
    out["seed_took"] = bool(moved)
    print("  SCATTER:", json.dumps({k: v["range"] for k, v in spread.items()}), flush=True)
    for s, d in pairs.items():
        print(f"  seed0 vs seed{s}:", json.dumps(d), flush=True)
    assert moved, "every scalar identical across seeds -- the seed did not reach the sampler"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
