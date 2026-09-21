"""One fold of one model on one target, with the ranking rule's inputs recorded PER SAMPLE.

A ranking rule is post-forward: it cannot move a diffusion sample, only decide which of them
is written as `<stem>.cif`. So one fold per (model, target, seed) prices EVERY candidate rule,
by re-ranking the recorded scalars offline. That is the whole reason this campaign can cover
four models and several targets at all.

`tt_bio.ranking.ranking_score` appends one line per call to $TT_BIO_RANK_RECORD. The first
`--samples` lines are samples 0..n-1 in order, at all three sites: worker._protenix_emit sorts
before it does anything else, rf3 builds one summary per sample in index order, and
openfold3_fold computes the score inside _confidence as each sample comes off the diffusion
head. The later lines are the same scalars re-read for the metrics rows. `_check` asserts the
model's own reported best score equals the max over the first n, which is what makes the
"first n in order" reading a checked one rather than an assumption.

    python3 perf/of3t_rankunify/fold_one.py --model protenix-v2 --target examples/ubq.yaml \
        --gt examples/ground_truth_structures/ubiquitin.pdb --seed 1 --card 2
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--gt", default=None)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--card", type=int, default=2)
ap.add_argument("--samples", type=int, default=5)
ap.add_argument("--sampling-steps", type=int, default=None)
ap.add_argument("--out-root", default=os.path.expanduser("~/of3t_rankunify_out"))
ap.add_argument("--msa-dir", default=os.path.expanduser("~/of3t_rankunify_msa"))
ap.add_argument("--python", default=os.path.expanduser("~/tt-bio-dev/env/bin/python"))
ap.add_argument("--tt-smi", default=os.path.expanduser("~/.local/bin/tt-smi"))
a = ap.parse_args()

stem = pathlib.Path(a.target).stem
tag = f"{a.model}__{stem}__s{a.seed}"
out = pathlib.Path(a.out_root) / tag
out.mkdir(parents=True, exist_ok=True)
if (out / "samples.json").exists():
    print(f"SKIP {tag}")
    sys.exit(0)
# tt_bio skips a target whose results.json already exists, so a run that died in
# post-processing would otherwise "resume" into a fold that never happened and record nothing.
import shutil
for child in out.iterdir():
    shutil.rmtree(child) if child.is_dir() else child.unlink()
rec = out / "rank_record.jsonl"

env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
env.update(TT_VISIBLE_DEVICES=str(a.card), TT_BIO_LEASE_CARDS=str(a.card),
           TT_BIO_LEASE_HOLDER="worker:of3t-rankunify", TT_BIO_RANK_RECORD=str(rec))

# AICLK sampled DURING the fold. A Blackhole number without one is not a measurement, and a
# chip a co-tenant is holding at 800 makes the timing an artifact to declare, not report.
clk, stop = [], threading.Event()


def _clock():
    while not stop.wait(5.0):
        try:
            t = json.loads(subprocess.run([a.tt_smi, "-s"], capture_output=True, text=True,
                                          timeout=30).stdout)
            clk.append(int(t["device_info"][a.card]["telemetry"]["aiclk"]))
        except Exception:
            pass


if os.path.exists(a.tt_smi):
    threading.Thread(target=_clock, daemon=True).start()

cmd = [a.python, "-m", "tt_bio.main", "predict", a.target, "--model", a.model,
       "--out_dir", str(out), "--seed", str(a.seed),
       "--diffusion_samples", str(a.samples), "--use_msa_server", "--msa_dir", a.msa_dir]
if a.sampling_steps:
    cmd += ["--sampling_steps", str(a.sampling_steps)]
t0 = time.time()
proc = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
wall = time.time() - t0
stop.set()
(out / "fold.log").write_text(proc.stdout + proc.stderr)
if proc.returncode != 0:
    print(f"FAIL {tag} rc={proc.returncode}")
    print(proc.stdout[-2000:], proc.stderr[-2000:])
    sys.exit(proc.returncode)

rows = [json.loads(l) for l in rec.read_text().splitlines()] if rec.exists() else []
assert len(rows) >= a.samples, f"{tag}: {len(rows)} ranking calls for {a.samples} samples"
per = rows[:a.samples]

res = json.loads(next(out.rglob("results.json")).read_text())
struct = next(out.rglob("structures"))
order = sorted(range(a.samples), key=lambda k: per[k]["score"], reverse=True)
rank_of = {k: r for r, k in enumerate(order)}


def _f(k):
    r = rank_of[k]
    hits = [p for p in struct.iterdir()
            if p.stem == (stem if r == 0 else f"{stem}_model_{r}")]
    assert len(hits) == 1, f"{tag}: rank {r} -> {hits}"
    return hits[0]


def _rmsd(path):
    if not a.gt:
        return None
    sys.path.insert(0, str(HERE)); from ca_rmsd import ca_rmsd
    return ca_rmsd(str(path), a.gt)


samples = []
for k in range(a.samples):
    f = _f(k)
    samples.append(dict(per[k], sample=k, rank=rank_of[k], file=f.name,
                        sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                        rmsd_ca=_rmsd(f)))

# The model reports one row per RANK in results.json. Every one of them must match the
# sample the record says landed at that rank, on the score and on pLDDT/pTM/ipTM. That checks
# the whole rank-to-sample mapping, which is what the offline re-ranking rests on -- not just
# that the winner looks right.
served = [s for s in samples if s["rank"] == 0][0]


def _rows(obj):
    if isinstance(obj, list):
        for x in obj:
            yield from _rows(x)
    elif isinstance(obj, dict):
        if "all_runs" in obj and isinstance(obj["all_runs"], list):
            yield obj["all_runs"]
        for v in obj.values():
            yield from _rows(v)


runs = next(iter(_rows(res)), None)
check, mismatch = None, []
if runs and len(runs) == a.samples:
    check = True
    by_rank = {s["rank"]: s for s in samples}
    for r, row in enumerate(runs):
        got = by_rank[row.get("rank", r)]
        for key, mine in (("ptm", got["ptm"]), ("iptm", got["iptm"] or 0.0),
                          ("plddt", got["plddt"]),
                          ("confidence_score", got["score"]),
                          ("ranking_score", got["score"])):
            if key not in row or row[key] is None:
                continue
            # 6e-5, not 1e-3: the published values are rounded to 4 decimals at
            # worst, so anything above 5e-5 is a real disagreement. At 1e-3 this
            # check passed four rf3 cells whose per-sample record was doubled.
            if abs(float(row[key]) - float(mine)) > 6e-5:
                check = False
                mismatch.append((r, key, row[key], mine))

payload = dict(tag=tag, model=a.model, target=a.target, gt=a.gt, seed=a.seed, card=a.card,
               n_samples=a.samples, wall_s=round(wall, 1),
               aiclk=dict(n=len(clk), median=sorted(clk)[len(clk) // 2] if clk else None,
                          min=min(clk) if clk else None, max=max(clk) if clk else None),
               n_rank_calls=len(rows), rank_map_verified=check, rank_map_mismatch=mismatch, samples=samples)
(out / "samples.json").write_text(json.dumps(payload, indent=2) + "\n")
print(f"OK {tag} {wall:.1f}s aiclk={payload['aiclk']['median']} "
      f"served=rank0 sample={served['sample']} rmsd={served['rmsd_ca']} check={check}")
