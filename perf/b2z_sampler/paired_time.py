#!/usr/bin/env python3
"""What a step/recycle setting is worth in seconds: arms interleaved in one process, one card.

The swarm shares whglx, so an absolute fold time taken here is worth nothing. This measures the
RATIO instead: every arm folds, in turn, in the same process, on the same card, with the same
weights and the same warm kernel caches, `--reps` times round. Contention that moves one arm moves
its neighbour a few seconds later, so it cancels out of the ratio; what does not cancel shows up
in the incumbent arm's own spread, which is reported as the A/A floor. A ratio smaller than the
floor has not been measured.

The first arm is the incumbent and every ratio is against it.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep import PANEL  # noqa: E402

OUT: dict = {"runs": []}
OUT_PATH: Path | None = None


def dump() -> None:
    if OUT_PATH is not None:
        tmp = OUT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(OUT, indent=1))
        tmp.replace(OUT_PATH)


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="cdk2x2_512", choices=sorted(PANEL))
    ap.add_argument("--arms", default="200:3,100:3,50:3,50:2,200:2",
                    help="steps:recycles, incumbent first")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    OUT_PATH = a.out
    a.out.parent.mkdir(parents=True, exist_ok=True)

    arms = [tuple(int(v) for v in x.split(":")) for x in a.arms.split(",")]
    spec = PANEL[a.target]
    tgt = ROOT / spec["yaml"]

    import torch
    torch.set_grad_enabled(False)
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"

    B.RECYCLING_STEPS, B.SAMPLING_STEPS = arms[0][1], arms[0][0]
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    import importlib.metadata as im
    OUT["env"] = {
        "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "ttnn": im.version("ttnn"), "target": a.target, "arms": arms, "reps": a.reps,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "timed_region": "predict_one (featurise + fold + CIF write)",
    }
    dump()

    msa_dir = Path(__file__).resolve().parent / f".msa_{a.target}"
    seed_tgt = tgt if "a3m" in spec else ROOT / "perf/size512/fixtures/cdk2x2_128.yaml"
    seed_a3m = (ROOT / spec["a3m"] if "a3m" in spec
                else ROOT / "perf/size512/fixtures/cdk2x2_128.a3m")
    _of, meta, state = B.build_fold("boltz2", msa_dir, seed_tgt, seed_a3m)
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "card_type", "aiclk_mhz")
                       if k in meta})
    base_cfg = dict(meta["job_cfg"])
    if spec.get("single_sequence"):
        base_cfg.update(single_sequence=True, use_msa_server=False)
    state.bind_run("b2z", base_cfg)
    state.pfn = lambda *a_, **k: None
    struct_dir = Path(meta["struct_dir"])
    dump()

    def fold(steps: int, recycles: int) -> float:
        pa = state.model.predict_args
        pa["sampling_steps"], pa["recycling_steps"] = steps, recycles
        cfg = dict(base_cfg, sampling_steps=steps, recycling_steps=recycles, seed=0,
                   struct_dir=str(struct_dir))
        for p in struct_dir.glob("*"):
            if p.is_file():
                p.unlink()
        t0 = time.perf_counter()
        state.predict_one(tgt, cfg)
        return time.perf_counter() - t0

    print(f"=== cold fold (discarded) ===", flush=True)
    OUT["cold_s"] = round(fold(*arms[0]), 3)
    dump()

    for rep in range(a.reps):
        for steps, rec in arms:
            dt = fold(steps, rec)
            OUT["runs"].append({"rep": rep, "steps": steps, "recycles": rec,
                                "fold_s": round(dt, 3),
                                "loadavg": open("/proc/loadavg").read().split()[:3]})
            dump()
            print(f"  rep{rep} {steps:>3}/{rec} {dt:8.3f}s", flush=True)

    inc = arms[0]
    by = {}
    for r in OUT["runs"]:
        by.setdefault((r["steps"], r["recycles"]), []).append(r["fold_s"])
    inc_s = by[inc]
    OUT["summary"] = {
        "incumbent": {"steps": inc[0], "recycles": inc[1],
                      "median_s": round(st.median(inc_s), 3), "s": inc_s,
                      "aa_floor_pct": round(100 * (max(inc_s) - min(inc_s)) / st.median(inc_s), 3)},
        "arms": [{"steps": k[0], "recycles": k[1], "n": len(v),
                  "median_s": round(st.median(v), 3),
                  "speedup_x": round(st.median(inc_s) / st.median(v), 4),
                  "seconds_removed": round(st.median(inc_s) - st.median(v), 3)}
                 for k, v in by.items() if k != inc],
    }
    dump()
    print(json.dumps(OUT["summary"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
