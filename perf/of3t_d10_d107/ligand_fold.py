#!/usr/bin/env python3
"""One real ligand fold, both ranking arms off its own samples.

What this closes: the branch changes what OpenFold3 serves for a target carrying a ligand or
a modified residue, and until now that was measured only on constructed logits. This runs the
production CLI on a protein+ligand target and reads, per diffusion sample, the frame mask the
fix builds, what it costs, and the pTM/ipTM the shipped rule and the fixed rule give on
IDENTICAL samples -- so the ranking difference carries no sampling noise at all.

AICLK is sampled DURING the fold. There is no speed claim here, but the frame-mask cost is a
time and a time on Blackhole without a clock is not a measurement.

  ligand_fold.py --card 1 --out <dir> [--samples 5] [--sampling-steps 200]
"""
import argparse
import json
import os
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--card", default="1")
ap.add_argument("--out", required=True)
ap.add_argument("--target", default=os.path.join(HERE, "targets", "fkbp_sb3.yaml"))
ap.add_argument("--samples", type=int, default=5)
ap.add_argument("--sampling-steps", type=int, default=200)
ap.add_argument("--seed", type=int, default=1)
# openbind, not openfold3, is the OF3-family model that HONOURS a ligand
# (`capabilities.py:91-94`): preview2 was released polymer-only and refuses one
# at the front door. Both run the same `openfold3_fold._confidence`, so the
# ligand path this fix is about is reachable on openbind and not on openfold3.
ap.add_argument("--model", default="openbind")
ap.add_argument("--python", default=os.path.join(os.path.expanduser("~"),
                                                 "tt-bio-dev", "env", "bin", "python"))
ap.add_argument("--tt-smi", default=os.path.join(os.path.expanduser("~"),
                                                 ".local", "bin", "tt-smi"))
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
rec = os.path.join(a.out, "capture.jsonl")
if os.path.exists(rec):
    os.remove(rec)

env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join(
    [ROOT, os.path.join(HERE, "_hook")]
    + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
env.update(OF3T_RECORD=rec, TT_VISIBLE_DEVICES=a.card, TT_BIO_LEASE_CARDS=a.card,
           TT_BIO_LEASE_HOLDER="worker:of3t-d10-d107")

clk, stop = [], threading.Event()


def _sample_clock():
    while not stop.wait(10.0):
        try:
            t = json.loads(subprocess.run([a.tt_smi, "-s"], capture_output=True, text=True,
                                          timeout=30).stdout)
            clk.append(int(t["device_info"][int(a.card)]["telemetry"]["aiclk"]))
        except Exception:
            pass


if os.path.exists(a.tt_smi):
    threading.Thread(target=_sample_clock, daemon=True).start()

cmd = [a.python, "-m", "tt_bio.main", "predict", a.target, "--model", a.model,
       "--out_dir", a.out, "--seed", str(a.seed),
       "--diffusion_samples", str(a.samples),
       "--sampling_steps", str(a.sampling_steps)]
t0 = time.time()
p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
wall = time.time() - t0
stop.set()

lines = [json.loads(x) for x in open(rec)] if os.path.exists(rec) else []
samples = [x for x in lines if x["kind"] == "sample"]

out = {"target": a.target, "model": a.model, "card": a.card, "seed": a.seed, "samples": a.samples,
       "sampling_steps": a.sampling_steps, "wall_s": round(wall, 2),
       "returncode": p.returncode,
       "aiclk_during": {"n": len(clk), "min": min(clk) if clk else None,
                        "max": max(clk) if clk else None,
                        "median": sorted(clk)[len(clk) // 2] if clk else None},
       "hook_installed": any(x["kind"] == "installed" for x in lines),
       "per_sample": samples}

if samples:
    ship = [0.8 * s["iptm_shipped"] + 0.2 * s["ptm_shipped"] for s in samples]
    fix = [0.8 * s["iptm_masked"] + 0.2 * s["ptm_masked"] for s in samples]
    order = lambda v: sorted(range(len(v)), key=lambda i: -v[i])
    out["ranking"] = {
        "shipped_scores": [round(x, 6) for x in ship],
        "fixed_scores": [round(x, 6) for x in fix],
        "shipped_order": order(ship), "fixed_order": order(fix),
        "orders_agree": order(ship) == order(fix),
        "served_same": order(ship)[0] == order(fix)[0],
        "max_abs_ptm_delta": max(abs(s["ptm_masked"] - s["ptm_shipped"]) for s in samples),
        "max_abs_iptm_delta": max(abs(s["iptm_masked"] - s["iptm_shipped"])
                                  for s in samples),
        "frameless_tokens": sorted({s.get("n_frameless") for s in samples}),
        "frameless_standard_residues": sorted({s.get("n_frameless_standard")
                                               for s in samples}),
        "atomized_tokens": sorted({s.get("n_atomized") for s in samples}),
        "n_atom": sorted({s.get("n_atom") for s in samples}),
        "frame_ms": {"min": round(min(s["frame_ms"] for s in samples), 3),
                     "max": round(max(s["frame_ms"] for s in samples), 3),
                     "total_s": round(sum(s["frame_ms"] for s in samples) / 1e3, 3)},
    }
    out["ranking"]["frame_share_of_fold_pct"] = round(
        100.0 * out["ranking"]["frame_ms"]["total_s"] / wall, 4)

if p.returncode != 0:
    out["stderr_tail"] = p.stderr[-4000:]
    out["stdout_tail"] = p.stdout[-2000:]

with open(os.path.join(a.out, "LIGAND_FOLD.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_sample"}, indent=1))
