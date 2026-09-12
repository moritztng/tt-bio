#!/usr/bin/env python3
"""The 512 aa Boltz-2 fold A/B for the atom key window, on whglx Wormhole.

A step ratio is not a fold ratio. `b2z2-sampler-ceiling-map` puts the whole sampler at 5.28 s of
the fold, so a 1.07x on the step is worth ~1.9 % of the fold and nothing else moves -- this
measures that rather than multiplying it out.

Both arms set the flag explicitly so no arm can inherit the previous one's state, `base` runs at
the first AND the last position of every rep so drift across the rep is visible instead of folded
into the ratio, and the cold fold is discarded. The structure is hashed on every run: the window
is NOT bit-exact against the one-hot matmul (it is exact against the definition the matmul
approximates), so the CIF hash is expected to move, and what it must not do is move between two
runs of the SAME arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def sha_dir(d: Path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(d.glob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--arms", default="base,window,base")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn                                                           # noqa: F401
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    import fold_ab_multi as FAM

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, "boltz2")
    FAM.patch_boltz2_cfg()

    fix = ROOT / "perf" / "size512" / "fixtures"
    T._ATOM_KEY_WINDOW = False
    one_fold, meta, state = B.build_fold(
        "boltz2", ROOT / f".msa_b2z2sf_{a.size}", fix / f"cdk2x2_{a.size}.yaml",
        fix / f"cdk2x2_{a.size}.a3m")
    struct_dir = Path(meta["struct_dir"])

    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "size": a.size, "recycling_steps": B.RECYCLING_STEPS,
           "sampling_steps": B.SAMPLING_STEPS,
           "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        a.out.write_text(json.dumps(res, indent=1))

    print(f"=== boltz2 {a.size} aa rec={B.RECYCLING_STEPS} steps={B.SAMPLING_STEPS}: cold ===",
          flush=True)
    cold_s, _ = one_fold()
    res["cold_s"] = round(cold_s, 3)
    print(f"  cold {cold_s:.3f} s (discarded)", flush=True)
    dump()

    for rep in range(a.reps):
        for arm in a.arms.split(","):
            T._ATOM_KEY_WINDOW = (arm == "window")
            s, _m = one_fold()
            row = {"rep": rep, "arm": arm, "flag": T._ATOM_KEY_WINDOW, "s": round(s, 3),
                   "cif": sha_dir(struct_dir)}
            res["runs"].append(row)
            print(f"  rep{rep} {arm:7s} {s:8.3f} s  {list(row['cif'].values())}", flush=True)
            dump()

    by = {}
    for r in res["runs"]:
        by.setdefault(r["arm"], []).append(r["s"])
    res["median_s"] = {k: round(st.median(v), 3) for k, v in by.items()}
    if "base" in by and "window" in by:
        res["fold_ratio"] = round(st.median(by["base"]) / st.median(by["window"]), 5)
    dump()
    print(json.dumps({"median_s": res["median_s"], "fold_ratio": res.get("fold_ratio")}),
          flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
