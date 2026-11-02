#!/usr/bin/env python3
"""Index the B200 reference predictions for mol1-quality.

    python3 mol1_refindex.py --qb2 ttuser@tt-quietbox2:/home/ttuser/mol1-fair-gpu-ref --out FILE LABEL [LABEL ...]
    python3 mol1_refindex.py --qb2 ttuser@tt-quietbox2:/home/ttuser/mol1-gpu-fp32 --arm gpu_fp32 --sets paired \
        --seeds 42,66 --out FILE LABEL [LABEL ...]

Several sessions ran the same job list (forward and reverse boxes, relaunches seeded with earlier outputs), so a job
can appear in more than one ledger. The structures live on qb2 under <dir>/<label>/ref/<set>/<case>/seed_<s>/; the
ledgers were pulled to perf/kitcmp/results/<label>/mol1/ref/jobs.jsonl. One record per expected job: the session
that produced its structures, the qb2 directory, sample count, recipe, wall time. Jobs with no structures anywhere
are listed under "missing" so a grader sees the gap instead of a silently shorter set. --arm names the arm in the
index (gpu = the bf16 reference, gpu_fp32 = upstream's fp32 mode, mol1_ref.py --dtype fp32); --sets/--seeds narrow the
expected job list to what that arm ran.
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
SEEDS, SELFTEST = [42, 66, 101, 2024, 8888], [7, 13, 77, 1234, 9999]
ARMS = {"gpu": "Mol-1 upstream reference predictions on NVIDIA B200, chosen arm cueq-triatt+chunk5+tf32 "
               "(state/mol1/gpu/B200.md), protocol state/mol1/PROTOCOL.md",
        "gpu_fp32": "Mol-1 upstream in its fp32 mode (autocast target float32, TF32 matmuls off) on NVIDIA B200, "
                    "kernels and settings otherwise as arm gpu (mol1_ref.py --dtype fp32), decision D-orch-56"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qb2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=list(ARMS), default="gpu")
    ap.add_argument("--sets", default="paired,selftest,headline")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)), help="GPU seeds (paired and headline)")
    ap.add_argument("labels", nargs="+")
    a = ap.parse_args()
    host, root = a.qb2.split(":", 1)
    sh = lambda c: subprocess.run(["ssh", "-o", "BatchMode=yes", host, c], capture_output=True, text=True,
                                  timeout=900, check=True).stdout
    idx = json.loads(sh(f"cat {root}/inputs/INDEX.json"))
    stem = lambda c: Path(idx["cases"][c]).stem  # mol1_ref.py's directory name for a case
    head = set(idx["headline_only"])
    paired = [c for c in idx["cases"] if c not in head]
    seeds = [int(x) for x in a.seeds.split(",")]
    jobs = ([(c, "paired", s) for c in paired for s in seeds] + [(c, "selftest", s) for c in paired for s in SELFTEST]
            + [(c, "headline", s) for c in sorted(head) for s in seeds])
    jobs = [j for j in jobs if j[1] in a.sets.split(",")]

    # (label, set, stem, seed) -> number of CIFs; summaries mark a finished job (mol1_ref.py's skip rule)
    cifs, done = defaultdict(int), set()
    listing = sh(f"cd {root} && find {' '.join(l + '/ref' for l in a.labels)} -path '*/predictions/*' "
                 r"\( -name '*.cif' -o -name '*summary_confidence*' \)")
    for p in listing.split():
        lab, _, st, cs, sd, _, f = p.split("/")[:7]
        k = (lab, st, cs, int(sd.removeprefix("seed_")))
        if f.endswith(".cif"):
            cifs[k] += 1
        else:
            done.add(k)

    led = {}
    for lab in a.labels:
        f = R / lab / "mol1/ref/jobs.jsonl"
        for line in f.read_text().splitlines() if f.exists() else []:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("status") == "ok":
                led[(lab, d["set"], stem(d["case"]), d["seed"])] = d

    records, missing = [], []
    for c, st, s in jobs:
        hit = next((lab for lab in a.labels if (lab, st, stem(c), s) in done), None)
        if hit is None:
            missing.append({"case": c, "set": st, "seed": s})
            continue
        k = (hit, st, stem(c), s)
        d = led.get(k, {})
        records.append({"case": c, "set": st, "seed": s, "recipe": idx["recipes"].get(c, "default"),
                        "session": hit, "samples": cifs[k], "wall_s": d.get("wall_s"),
                        "dir": f"{host}:{root}/{hit}/ref/{st}/{stem(c)}/seed_{s}/predictions"})
    sets = a.sets.split(",")
    out = {"what": ARMS[a.arm], "arm": a.arm,
           "inputs": f"{host}:{root}/inputs (INDEX.json, sha256 per file)",
           "seeds": {k: v for k, v in {"paired": seeds, "selftest": SELFTEST, "headline": seeds}.items() if k in sets},
           "expected": len(jobs), "present": len(records), "missing_n": len(missing),
           "sessions": a.labels, "jobs": records, "missing": missing}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(records)}/{len(jobs)} jobs present, {len(missing)} missing -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
