"""One OpenFold3 fold, with every confidence signal recorded PER SAMPLE beside the true RMSD.

`of3t-pairbias` stored only the winner's aggregate confidence, which is why D10 -- "the
confidence head mis-ranks its own diffusion samples" -- could be seen and not diagnosed. This
driver runs the production CLI unchanged and lets `_hook/sitecustomize.py` record, for every
diffusion sample of every run:

  rmsd_ca      Ca-RMSD against the experimental structure, the truth the head is judged on;
  plddt        mean predicted lDDT over atoms, and over the Ca atoms alone;
  ptm / iptm   the two pTM reductions the ranking score reads. `iptm` is 0.0 by construction
               on a single chain, which is the point of recording it;
  disorder     the AF3 RASA term, and `has_clash`, the other two ranking inputs;
  rank_score   0.8*iptm + 0.2*ptm + 0.5*disorder - 100*has_clash, what actually selects;
  pae / pde    mean expected error in Angstrom, plus gPDE (AF3 SI 5.7 Eq 16);
  resolved     mean P(experimentally resolved), the fourth head output.

The fold runs in a multiprocessing SPAWN child, so the capture and the arm lever live in a
`sitecustomize` on the child's PYTHONPATH rather than in this process. The same PYTHONPATH
puts THIS CHECKOUT ahead of the environment's editable `tt_bio`, so the harness scores the
code it ships with and not whatever is installed (memory
`parity-gate-scores-installed-package-not-checkout`).

Two arms, one lever, exactly D1: the trunk Pairformer's `scale_pair_bias`. The resolved flag
is read back out of the recorded stream and asserted before the report is written.

    python3 perf/of3t_confhead/rank_fold.py --arm fix --seed 1 --card 0 \
        --msa-dir ~/of3t_confhead_msa --out-root ~/of3t_confhead_out/fold
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=("ship", "fix"), required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--samples", type=int, default=5)
ap.add_argument("--sampling-steps", type=int, default=200)
ap.add_argument("--target", default=os.path.join(ROOT, "examples/ubq.yaml"))
ap.add_argument("--gt", default=os.path.join(
    ROOT, "examples/ground_truth_structures/ubiquitin.pdb"))
ap.add_argument("--msa-dir", default=os.path.expanduser("~/of3t_confhead_msa"))
ap.add_argument("--out-root", default=os.path.expanduser("~/of3t_confhead_out/fold"))
ap.add_argument("--card", type=int, default=0)
ap.add_argument("--python", default=sys.executable)
ap.add_argument("--tt-smi", default=os.path.expanduser("~/.local/bin/tt-smi"))
a = ap.parse_args()

out_dir = os.path.join(a.out_root, f"{a.arm}_s{a.seed}")
os.makedirs(out_dir, exist_ok=True)
rec = os.path.join(out_dir, "capture.jsonl")
if os.path.exists(rec):
    os.remove(rec)

env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join(
    [ROOT, os.path.join(HERE, "_hook")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
env.update(OF3T_ARM=a.arm, OF3T_RECORD=rec, OF3T_GT=a.gt,
           TT_VISIBLE_DEVICES=str(a.card), TT_BIO_LEASE_CARDS=str(a.card),
           TT_BIO_LEASE_HOLDER="worker:of3t-confhead")

# AICLK, sampled DURING the fold. A fold time without one is not a measurement on Blackhole,
# and a chip held below ~1200 MHz by a co-tenant makes the timing an artifact to be declared
# rather than reported.
clk, stop = [], threading.Event()


def _sample_clock():
    while not stop.wait(10.0):
        try:
            t = json.loads(subprocess.run([a.tt_smi, "-s"], capture_output=True, text=True,
                                          timeout=30).stdout)
            clk.append(int(t["device_info"][a.card]["telemetry"]["aiclk"]))
        except Exception:
            pass


if os.path.exists(a.tt_smi):
    threading.Thread(target=_sample_clock, daemon=True).start()

cmd = [a.python, "-m", "tt_bio.main", "predict", a.target, "--model", "openfold3",
       "--out_dir", out_dir, "--seed", str(a.seed),
       "--diffusion_samples", str(a.samples), "--sampling_steps", str(a.sampling_steps),
       "--use_msa_server", "--msa_dir", a.msa_dir]
t0 = time.time()
p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
wall = time.time() - t0
stop.set()

lines = [json.loads(x) for x in open(rec)] if os.path.exists(rec) else []
samples = [x for x in lines if x["kind"] == "sample"]
trunk = [x for x in lines if x["kind"] == "trunk_pairformer"]

if p.returncode != 0 or len(samples) != a.samples:
    sys.stderr.write(p.stdout[-4000:] + "\n" + p.stderr[-4000:] + "\n")
    raise SystemExit(f"fold failed: rc={p.returncode} samples={len(samples)}/{a.samples}")

flags = {(t["scale_pair_bias"], t["tri_att_scale_pair_bias"]) for t in trunk}
assert len(flags) == 1, f"trunk built with mixed flags: {flags}"
scale, tri = next(iter(flags))
assert scale is (a.arm == "fix"), f"arm {a.arm} resolved scale_pair_bias={scale}"

report = {"arm": a.arm, "seed": a.seed, "samples": a.samples,
          "trunk_scale_pair_bias": scale, "trunk_tri_att_scale_pair_bias": tri,
          "n_trunk_pairformers": len(trunk), "wall_s": round(wall, 2),
          "aiclk_during": {"n": len(clk), "min": min(clk) if clk else None,
                           "max": max(clk) if clk else None,
                           "median": sorted(clk)[len(clk) // 2] if clk else None},
          "ground_truth": a.gt, "per_sample": samples}
path = os.path.join(out_dir, "samples.json")
json.dump(report, open(path, "w"), indent=1)
print(f"[{a.arm} s{a.seed}] trunk scale_pair_bias={scale} tri={tri}  wall {wall:.1f}s  "
      f"aiclk {report['aiclk_during']}")
for r in samples:
    print(f"  sample {r['sample']}: rmsd {r['rmsd_ca']:6.3f} A  plddt {r['plddt']:.4f}  "
          f"ptm {r['ptm']:.4f}  iptm {r['iptm']:.4f}  disorder {r['disorder']:.4f}  "
          f"score {r['rank_score']:.4f}  pae {r['pae_mean']:5.2f}  gpde {r['gpde']:5.2f}")
print("wrote", path)
