"""The fold-level A/B for L2: does admitting D=1 to the calibrated fuse_batch path move
designs/hour at the production config, and does it keep the design bit-for-bit?

Two arms, one fresh process each, alternating A B A B so thermal drift cancels (this lineage
measured +13.3% all-A-then-B where interleaved gave +5.2%):

  off -- shipped default, `RFD3_TUNE_MATMUL` unset.
  on  -- `RFD3_TUNE_MATMUL=1` with the size gate, i.e. the calibrated bit-exact program
         config now reaches D=1 as well as D=8.

The metric is per DESIGN, not per step: whole-invocation wall clock is what a user waits,
and it is the only number that charges the arm for its one-time calibration. ms/step comes
from the slope of two timestep counts so the fixed setup drops out, never from dividing one
run. Parity is the sha256 of each arm's output CIF -- the calibrator discards any program
config that is not bitwise equal to ttnn's default, so identical arms must produce identical
files, and that is checkable rather than argued.

Run under benchlock: a co-tenanted A/B on this box is not a slower measurement, it is a wrong
one, and this lineage once saw a -5 s "win" evaporate on a clean re-run.
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
ARMS = {"off": {}, "on": {"RFD3_TUNE_MATMUL": "1"}}


def sha_dir(d: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*.cif"))}


def run_one(arm: str, wt: Path, specs: Path, ckpt: str, steps: int, designs: int,
            batch: int, seed: int, tag: str) -> dict:
    out = wt / "perf/p34/ab" / tag
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(TT_VISIBLE_DEVICES="3", TT_BIO_LEASE_HOLDER="worker:rfd3-deep-perf",
               PYTHONPATH=str(wt))
    env.pop("RFD3_TUNE_MATMUL", None)      # never inherit the flag from the parent
    env.update(ARMS[arm])
    cmd = [PY, "-m", "tt_bio.main", "design", str(specs), "--model", "rfd3", "--from_pdb",
           "--checkpoint", ckpt, "--out_dir", str(out), "--num_timesteps", str(steps),
           "--num_designs", str(designs), "--batch_size", str(batch), "--seed", str(seed)]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, cwd=str(wt), capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        print(f"  !! {tag} FAILED rc={p.returncode}\n{p.stdout[-2500:]}\n{p.stderr[-2500:]}")
        return {"arm": arm, "tag": tag, "ok": False, "wall_s": wall, "steps": steps,
                "designs": designs, "batch": batch}
    shas = sha_dir(out)
    n_tuned = p.stdout.count("[tune]")
    print(f"  {tag:22s} arm={arm:3s} steps={steps:3d} D={designs} wall={wall:7.2f} s "
          f"cifs={len(shas)}", flush=True)
    return {"arm": arm, "tag": tag, "ok": True, "wall_s": wall, "steps": steps,
            "designs": designs, "batch": batch, "shas": shas, "n_tune_lines": n_tuned}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt", default="/home/ttuser/.coworker/wt/rfd3-deep-perf")
    ap.add_argument("--ckpt", default="/home/ttuser/.boltz/rfd3/weights")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--short_steps", type=int, default=20)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--designs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", action="store_true", default=True)
    ap.add_argument("--no_warmup", action="store_true")
    ap.add_argument("--no_slope", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    wt = Path(a.wt)
    specs = wt / "perf/p34/specs_iai.json"
    specs.parent.mkdir(parents=True, exist_ok=True)
    specs.write_text(json.dumps({"iai": {
        "input": "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb",
        "contig": "A1-10,230,A31-40"}}))

    rows = []
    if a.warmup and not a.no_warmup:
        print("[ab] warmup (discarded: first run on a fresh host is not comparable)", flush=True)
        run_one("off", wt, specs, a.ckpt, 5, 1, a.batch, a.seed, "warmup")

    # The slope pair, per arm, so ms/step never comes from dividing one run.
    print(f"[ab] slope pair at {a.short_steps} and {a.steps} timesteps", flush=True)
    slope = {}
    for arm in (() if a.no_slope else ("off", "on")):
        r = run_one(arm, wt, specs, a.ckpt, a.short_steps, a.designs, a.batch, a.seed,
                    f"slope_{arm}_t{a.short_steps}")
        rows.append(r); slope[arm] = r

    # The interleaved main A/B at the production timestep count.
    print(f"[ab] interleaved A/B, {a.reps} reps per arm at {a.steps} timesteps, "
          f"D={a.designs} batch={a.batch}", flush=True)
    for rep in range(a.reps):
        for arm in ("off", "on"):
            rows.append(run_one(arm, wt, specs, a.ckpt, a.steps, a.designs, a.batch,
                                a.seed, f"main_{arm}_r{rep}"))

    # --- report -----------------------------------------------------------------------
    def wl(arm):
        return [r["wall_s"] for r in rows
                if r["arm"] == arm and r["ok"] and r["steps"] == a.steps
                and r["tag"].startswith("main_")]

    summary = {"steps": a.steps, "short_steps": a.short_steps, "designs": a.designs,
               "batch": a.batch, "seed": a.seed, "arms": {}}
    print()
    for arm in ("off", "on"):
        w = wl(arm)
        if not w:
            continue
        med = statistics.median(w)
        s_design = med / a.designs
        # slope: (t_long - t_short) / (steps-1 - short_steps-1) per design
        sl = slope.get(arm)
        ms_step = None
        if sl and sl["ok"]:
            ms_step = (med - sl["wall_s"]) / ((a.steps - 1) - (a.short_steps - 1)) * 1e3 / a.designs
        summary["arms"][arm] = {
            "walls_s": w, "median_wall_s": med, "spread_pct": (max(w) - min(w)) / med * 100,
            "s_per_design": s_design, "designs_per_hour": 3600 / s_design,
            "ms_per_step_slope": ms_step,
            "slope_short_wall_s": sl["wall_s"] if sl else None,
        }
        print(f"[ab] {arm:3s}  walls={[f'{x:.2f}' for x in w]}  median={med:.2f} s  "
              f"spread={summary['arms'][arm]['spread_pct']:.1f} %")
        print(f"[ab]      s/design={s_design:.2f}  designs/hour={3600 / s_design:.1f}  "
              f"ms/step(slope)={ms_step if ms_step is None else f'{ms_step:.1f}'}")

    if "off" in summary["arms"] and "on" in summary["arms"]:
        o, n = summary["arms"]["off"], summary["arms"]["on"]
        summary["speedup_wall"] = o["median_wall_s"] / n["median_wall_s"]
        summary["delta_pct"] = (o["median_wall_s"] - n["median_wall_s"]) / o["median_wall_s"] * 100
        # A/A floor: the off arm's own spread is the noise floor the delta must beat.
        summary["aa_floor_pct"] = o["spread_pct"]
        print(f"\n[ab] on vs off: {summary['speedup_wall']:.4f}x "
              f"({summary['delta_pct']:+.2f} %)   A/A floor from the off arm's own "
              f"spread: +-{o['spread_pct']:.2f} %")
        verdict = ("INSIDE NOISE" if abs(summary["delta_pct"]) <= o["spread_pct"]
                   else "WIN" if summary["delta_pct"] > 0 else "REGRESSION")
        summary["verdict"] = verdict
        print(f"[ab] verdict: {verdict}")

    # --- parity: every arm must produce byte-identical CIFs ----------------------------
    ref = next((r["shas"] for r in rows if r["ok"] and r["tag"].startswith("main_")), None)
    mismatches = []
    for r in rows:
        if r["ok"] and r["tag"].startswith("main_") and r["shas"] != ref:
            mismatches.append(r["tag"])
    summary["parity_ref_shas"] = ref
    summary["parity_mismatches"] = mismatches
    print(f"\n[ab] parity: {'ALL ARMS BYTE-IDENTICAL' if not mismatches else 'MISMATCH ' + str(mismatches)}")
    print(f"[ab] ref CIF sha256: {ref}")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(f"[ab] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
