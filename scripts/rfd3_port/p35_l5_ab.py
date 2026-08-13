"""The fold-level A/B for L5 (scale folded into the bias add), and for anything after it.

Two arms, two TREES, not one tree with a flag: the change is unconditional model code, and the
alternative -- rewriting `model.py` between runs -- races a live design against a half-written file.
`git worktree add` of the parent commit gives the off arm at zero risk, and both arms import their
own tree via PYTHONPATH.

Arms are interleaved A B A B (this lineage measured +13.3 % all-A-then-B where interleaved gave
+5.2 %), one fresh process each, and the two A runs are the session's A/A floor. Nothing is claimed
below that floor.

Every wall is a whole-invocation wall, which is what a user waits and the only figure that charges
an arm for its one-time costs. ms/step comes from the difference of two timestep counts so the fixed
setup drops out; never from dividing one run (the first step of a process costs 3.24 s at this
fixture, and a cold ~/.cache/ttnn costs 1114 ms/step).

Parity is the sha256 of each arm's output CIF. L5 is bit-exact by construction and was measured
`torch.equal` at the op, so the shas must be identical; if they are not, stop and do not report a
speedup.

    /home/ttuser/.coworker/scripts/benchlock.sh rfd3-host-half -- \
      env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:rfd3-host-half \
      /home/ttuser/tt-bio-dev/env/bin/python3 scripts/rfd3_port/p35_l5_ab.py \
        --off-rev HEAD~1 --reps 3 --out perf/p35/l5_ab.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

PY = "/home/ttuser/tt-bio-dev/env/bin/python3"
WT = Path(__file__).resolve().parents[2]


def sha_dir(d: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*.cif"))}


def run_one(tree: Path, steps: int, designs: int, seed: int, tag: str, ckpt: str,
            extra_env: dict | None = None) -> dict:
    out = WT / "perf/p35/ab" / tag
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(PYTHONPATH=str(tree))
    env.pop("RFD3_TUNE_MATMUL", None)
    env.update(extra_env or {})
    cmd = [PY, "-m", "tt_bio.main", "design", str(WT / "perf/p34/specs_iai.json"), "--model", "rfd3",
           "--from_pdb", "--out_dir", str(out), "--num_timesteps", str(steps),
           "--num_designs", str(designs), "--seed", str(seed)]
    if ckpt:
        cmd += ["--checkpoint", ckpt]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, cwd=str(tree), capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-2000:], flush=True)
        raise SystemExit(f"{tag} failed rc={p.returncode}")
    return {"wall_s": wall, "sha": sha_dir(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off-rev", default="HEAD~1", help="revision holding the OFF arm's model.py")
    ap.add_argument("--flag", default="", help="A/B one env flag in ONE tree (NAME), off=0 on=1, "
                                              "instead of two worktrees")
    ap.add_argument("--also-env", default="", help="NAME=VAL,... set in both arms")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--short", type=int, default=20)
    ap.add_argument("--long", type=int, default=200)
    ap.add_argument("--designs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    both = dict(kv.split("=", 1) for kv in a.also_env.split(",") if kv)
    if a.flag:
        arm_env = {"off": {**both, a.flag: "0"}, "on": {**both, a.flag: "1"}}
        off = None
        trees = {"off": WT, "on": WT}
    else:
        arm_env = {"off": dict(both), "on": dict(both)}
        off = Path("/tmp/rfd3_l5_off")
        if off.exists():
            subprocess.run(["git", "worktree", "remove", "--force", str(off)], cwd=str(WT))
        subprocess.run(["git", "worktree", "add", "--detach", str(off), a.off_rev], cwd=str(WT),
                       check=True)
        trees = {"off": off, "on": WT}
    rec: dict = {"off_rev": a.flag or a.off_rev, "flag": a.flag, "also_env": both,
                 "steps": [a.short, a.long], "designs": a.designs, "runs": []}
    try:
        for rep in range(a.reps):
            for arm in ("off", "on"):
                for steps in (a.short, a.long):
                    r = run_one(trees[arm], steps, a.designs, a.seed,
                                f"{arm}_t{steps}_r{rep}", a.ckpt, arm_env[arm])
                    r.update(arm=arm, steps=steps, rep=rep)
                    rec["runs"].append(r)
                    print(f"  {arm:3s} t={steps:3d} rep{rep}  {r['wall_s']:7.2f} s  "
                          f"{list(r['sha'].values())[0][:16]}", flush=True)
    finally:
        if off is not None:
            subprocess.run(["git", "worktree", "remove", "--force", str(off)], cwd=str(WT))

    for arm in ("off", "on"):
        w = {s: [r["wall_s"] for r in rec["runs"] if r["arm"] == arm and r["steps"] == s]
             for s in (a.short, a.long)}
        ms = (statistics.median(w[a.long]) - statistics.median(w[a.short])) / (a.long - a.short) * 1e3
        sdes = ms * (a.long - 1) / 1e3
        rec[arm] = {"walls": w, "ms_per_step": ms, "s_per_design_199": sdes,
                    "designs_per_hour_wall": 3600 / statistics.median(w[a.long])}
        print(f"{arm}: {ms:.1f} ms/step, {sdes:.1f} s/design of stepping, "
              f"{rec[arm]['designs_per_hour_wall']:.1f} designs/hour at the {a.long}-step wall",
              flush=True)
    aa = [r["wall_s"] for r in rec["runs"] if r["arm"] == "off" and r["steps"] == a.long]
    rec["aa_floor_pct"] = (max(aa) - min(aa)) / statistics.median(aa) * 100 if len(aa) > 1 else None
    rec["delta_pct"] = (rec["off"]["ms_per_step"] - rec["on"]["ms_per_step"]) / \
        rec["off"]["ms_per_step"] * 100
    shas = {json.dumps(r["sha"], sort_keys=True) for r in rec["runs"] if r["steps"] == a.long}
    rec["cif_identical_across_arms"] = len(shas) == 1
    print(f"delta {rec['delta_pct']:+.2f} % on ms/step, A/A floor "
          f"{rec['aa_floor_pct'] if rec['aa_floor_pct'] is None else round(rec['aa_floor_pct'], 2)} %, "
          f"CIFs identical across arms: {rec['cif_identical_across_arms']}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
