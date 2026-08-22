#!/usr/bin/env python3
"""Fold in and out of the issue-#9 token window inside ONE process.

The grid workaround is the only place tt-bio moves the main compute grid after the
device is open, and it moves it back when the fold returns. A/B'ing whole processes
(perf/protenix_grid/ab.py) never exercises that restore: every arm there folds once
and exits. A server folds many sizes in one process, so a restore that leaves a stale
program config behind would corrupt the NEXT fold, not the windowed one.

So: fold 512, then 506, then 512 again, all in one `tt-bio predict` process, and
require the two 512 folds to agree atom for atom. On a 13x10 card the middle fold
switches the grid and switches it back; on an 11x10 card nothing switches and this is
an A/A control on the same code.

  TT_VISIBLE_DEVICES=0 python3 perf/protenix_grid/restore.py --out perf/protenix_grid/restore.json
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "perf" / "size512" / "fixtures"
SWITCH_LINE = "[tt-bio] protenix-v2:"


def atoms_md5(cif: Path):
    """Coordinates only: the file header carries the target name, which differs
    between s1 and s3 by construction and says nothing about the structure."""
    rows = [l for l in cif.read_text().splitlines()
            if l.startswith(("ATOM", "HETATM"))]
    return hashlib.md5("\n".join(rows).encode()).hexdigest(), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="512,506,512",
                    help="fold order in one process; the middle one is the window")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="perf/protenix_grid/restore.json")
    ap.add_argument("--workdir", default="perf/protenix_grid/restore_work")
    a = ap.parse_args()

    rungs = [int(r) for r in a.rungs.split(",")]
    work = ROOT / a.workdir
    shutil.rmtree(work, ignore_errors=True)
    (work / "in").mkdir(parents=True)
    stems = []
    for i, rung in enumerate(rungs, 1):
        stem = f"s{i}_{rung}"
        shutil.copy(FIXTURES / f"cdk2x2_{rung}.yaml", work / "in" / f"{stem}.yaml")
        stems.append((stem, rung))

    env = dict(os.environ)
    env.pop("TT_BIO_FORCE_GRID", None)      # an explicit pin disables the workaround
    env["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(work / "in"),
           "--model", "protenix-v2", "--single_sequence", "--sampling_steps", "6",
           "--diffusion_samples", "1", "--seed", "0", "--out_dir", str(work / "out")]
    log = work / "run.log"
    t0 = time.monotonic()
    with open(log, "wb") as fh:
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=fh,
                             stderr=subprocess.STDOUT, timeout=a.timeout)
    wall = round(time.monotonic() - t0, 1)
    text = log.read_text(errors="replace")

    folds = []
    for stem, rung in stems:
        cif = next((work / "out").rglob(f"{stem}.cif"), None)
        row = {"stem": stem, "rung": rung}
        if cif:
            row["atoms_md5"], row["n_atom_rows"] = atoms_md5(cif)
        # One results.json for the whole directory, one row per target.
        res = next((work / "out").rglob("results.json"), None)
        if res:
            rows = [r for r in json.loads(res.read_text())
                    if r.get("id") == stem and r.get("status") == "ok"]
            if rows:
                row["plddt"] = rows[0].get("plddt")
                row["runtime_s"] = rows[0].get("runtime_s")
        folds.append(row)

    switches = [l.strip() for l in text.splitlines() if SWITCH_LINE in l]
    first, last = folds[0], folds[-1]
    same = (first.get("atoms_md5") is not None
            and first.get("atoms_md5") == last.get("atoms_md5")
            and first.get("plddt") == last.get("plddt"))
    out = {"rc": rc, "wall_s": wall, "one_process": True, "folds": folds,
           "switch_lines": switches, "switched": len(switches),
           "restore_bit_exact": bool(same),
           "pass": bool(rc == 0 and same and all("atoms_md5" in f for f in folds))}
    (ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
