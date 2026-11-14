#!/usr/bin/env python3
"""OpenDDE at the size-ladder's own fold, one tree per arm: lever census, CIF digest, runtime, AICLK.

Each arm is a tt_bio tree run through the SAME gate venv (one ttnn), the same fixture and the same
command the ladder runs (scripts/release_gate.py::_size_ladder_fold), so arms differ only in tt_bio.
Arms are interleaved per rep, so a box-wide clock or load drift lands on every arm alike.

    ab.py --card 0 --rungs 256,1024 --reps 2 --out DIR A=/tree/a B=/tree/b ...
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "perf"))
sys.path.insert(0, str(REPO))
from clocksample import during  # noqa: E402

GATE_TREE = Path("/home/ttuser/scratch/shipflow/trees/f9b8ca9e9f9d")
FIXTURES = GATE_TREE / "perf" / "size512" / "fixtures"


def cif_digest(out_dir: Path):
    cifs = sorted(out_dir.rglob("*.cif"))
    if not cifs:
        return None, None
    lines = [ln for ln in cifs[0].read_text().splitlines() if ln.startswith(("ATOM", "HETATM"))]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest(), str(cifs[0])


def runtime_s(out_dir: Path):
    for p in out_dir.rglob("results.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        for k in ("runtime_s", "fold_s"):
            if isinstance(d, dict) and k in d:
                return d[k]
        if isinstance(d, list) and d and isinstance(d[0], dict) and "runtime_s" in d[0]:
            return d[0]["runtime_s"]
        return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--rungs", default="256,1024")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--python", default="/home/ttuser/scratch/shipflow/venv-f9b8ca9e9f9d/bin/python")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("arms", nargs="+", metavar="NAME=TREE")
    a = ap.parse_args()
    a.out = a.out.resolve()
    arms = [x.split("=", 1) for x in a.arms]
    a.out.mkdir(parents=True, exist_ok=True)
    rows = a.out / "rows.jsonl"
    os.environ["TT_VISIBLE_DEVICES"] = str(a.card)
    for rung in [int(r) for r in a.rungs.split(",")]:
        for rep in range(a.reps):
            for name, tree in arms:
                label = f"{name}-{rung}-r{rep}"
                od = a.out / f"out_{label}"
                cj = a.out / f"census_{label}.json"
                cmd = [a.python, str(GATE_TREE / "scripts" / "lever_census.py"), "--tt-bio", a.python,
                       "--label", label, "--out", str(cj), "--pythonpath", tree, "--",
                       "-m", "tt_bio.main", "predict", str(FIXTURES / f"cdk2x2_{rung}.yaml"),
                       "--model", "opendde", "--single_sequence", "--sampling_steps", "6",
                       "--diffusion_samples", "1", "--seed", "0", "--out_dir", str(od)]
                t0 = time.time()
                with open(a.out / f"{label}.log", "w") as fp, during() as clk:
                    rc = subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT, cwd=tree).returncode
                dig, cif = cif_digest(od)
                row = {"arm": name, "tree": tree, "rung": rung, "rep": rep, "rc": rc,
                       "wall_s": round(time.time() - t0, 1), "runtime_s": runtime_s(od),
                       "digest": dig, "cif": cif, "clock": clk.summary().get(0), "clock_line": clk.line(0),
                       "load_max": max(clk.load) if clk.load else None, "card": a.card}
                with open(rows, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
