"""L6a at the fold: RFD3_SPARSE_BIAS off vs on, ms/step by the 20-vs-200-timestep slope.

Never divide one run. The first step of an RFD3 process costs 3.24 s registering ~3400
programs and a cold ``~/.cache/ttnn`` makes the whole run 3.5x slower
(``rfd3-baseline-seed-cold-cache-trap``), so both of those land in the intercept and only the
slope is the step. Arms alternate off/on inside one benchlock hold, and rep 0 vs rep 1 of the
OFF arm is the A/A floor -- measured in the same session as the effect, which is the only
floor worth quoting.

    /home/ttuser/.coworker/scripts/benchlock.sh rfd3-host-half -- \\
      /home/ttuser/tt-bio-dev/env/bin/python3 scripts/rfd3_port/p36_fold_ab.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

PY = "/home/ttuser/tt-bio-dev/env/bin/python3"
SPEC = "perf/p34/specs_iai.json"


def run(arm, steps, tag, card):
    out = Path("perf/p36/ab") / tag
    if out.exists():
        shutil.rmtree(out)
    env = dict(os.environ)
    env.update(
        TT_VISIBLE_DEVICES=str(card),
        TT_BIO_LEASE_HOLDER="worker:rfd3-host-half",
        PYTHONPATH=os.getcwd(),
        RFD3_SPARSE_BIAS="1" if arm == "on" else "0",
    )
    cmd = [PY, "-m", "tt_bio.main", "design", SPEC, "--model", "rfd3", "--from_pdb",
           "--out_dir", str(out), "--num_timesteps", str(steps),
           "--num_designs", "1", "--seed", "0"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    cif = out / "iai.cif"
    sha = hashlib.sha256(cif.read_bytes()).hexdigest() if cif.exists() else None
    (out.parent / f"{tag}.log").write_text(p.stdout + p.stderr)
    print(f"{tag:16s} arm={arm:3s} t={steps:3d} wall={wall:7.2f}s rc={p.returncode} "
          f"sha={sha[:16] if sha else 'NONE'}", flush=True)
    return {"arm": arm, "steps": steps, "wall": wall, "rc": p.returncode, "sha": sha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--card", type=int, default=1)
    ap.add_argument("--short", type=int, default=20)
    ap.add_argument("--long", type=int, default=200)
    ap.add_argument("--out", default="perf/p36/fold_ab.json")
    args = ap.parse_args()

    runs = []
    # One warm-up of the cheap arm, discarded: the first process of a session re-registers
    # every program and would be charged to whichever arm happened to go first.
    run("off", args.short, "warmup", args.card)
    for rep in range(args.reps):
        for arm in ("off", "on"):
            for steps in (args.short, args.long):
                r = run(arm, steps, f"{arm}_t{steps}_r{rep}", args.card)
                r["rep"] = rep
                runs.append(r)

    per_step = {}
    for arm in ("off", "on"):
        for rep in range(args.reps):
            s = next((r["wall"] for r in runs
                      if r["arm"] == arm and r["rep"] == rep and r["steps"] == args.short), None)
            l = next((r["wall"] for r in runs
                      if r["arm"] == arm and r["rep"] == rep and r["steps"] == args.long), None)
            if s and l:
                per_step[f"{arm}_r{rep}"] = (l - s) / (args.long - args.short) * 1e3

    res = {"runs": runs, "ms_per_step": per_step}
    off = [v for k, v in per_step.items() if k.startswith("off")]
    on = [v for k, v in per_step.items() if k.startswith("on")]
    if off and on:
        mo, mn = sum(off) / len(off), sum(on) / len(on)
        res["off_ms_step"], res["on_ms_step"] = mo, mn
        res["gain_pct"] = (mo - mn) / mo * 100
        res["aa_floor_pct"] = (max(off) - min(off)) / mo * 100 if len(off) > 1 else None
        shas = {r["sha"] for r in runs if r["steps"] == args.long and r["sha"]}
        res["long_sha_identical"] = len(shas) == 1
        res["long_sha"] = sorted(shas)
        print(f"\noff {mo:.1f} ms/step   on {mn:.1f} ms/step   "
              f"gain {res['gain_pct']:+.2f} %   A/A floor {res['aa_floor_pct']:.2f} %")
        print(f"200-step CIF identical across all arms: {res['long_sha_identical']} {res['long_sha']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
