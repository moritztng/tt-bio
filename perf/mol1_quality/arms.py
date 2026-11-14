"""Assemble the GPU arms grade.py reads from mol1-fair-gpu's reference sessions, by hardlink, complete folds only.

    python3 arms.py [--fp32] ROOT OUT [LABEL ...]

ROOT holds one dir per session (mol1-b200-ref2, ...), each with ref/{paired,selftest,headline}/<case>/seed_<s>/
predictions/. A seed dir counts once all five samples have their CIF and summary JSON (a session still running leaves
half-written ones; files younger than 2 min wait for the next run). It lands in
OUT/{gpu,gpu_self,headline}/<case>/seed_<s>/predictions/ unless that seed is already there. Without LABELs every ROOT/mol1-b200-* session is read. Re-run whenever a session adds folds.
--fp32: ROOT holds mol1-gpu-fp32's sessions (ROOT/mol1-*, upstream fp32 mode); their paired folds land in OUT/gpu_fp32.
"""
import os
import shutil
import sys
import time
from pathlib import Path

ARMS = {"paired": "gpu", "selftest": "gpu_self", "headline": "headline"}
N_SAMPLE = 5


def complete(seed_dir):
    p, name = seed_dir / "predictions", seed_dir.parent.name
    return all((p / f"{name}_sample_{k}.cif").exists() and (p / f"{name}_summary_confidence_sample_{k}.json").exists()
               for k in range(N_SAMPLE)) and time.time() - max(f.stat().st_mtime for f in p.iterdir()) > 120  # not mid-write


def main(root, out, labels, arms=ARMS, glob="mol1-b200-*"):
    root, out = Path(root), Path(out)
    sessions = [root / x for x in labels] or sorted(root.glob(glob))
    added = dict.fromkeys(arms.values(), 0)
    for s in sessions:
        for part, arm in arms.items():
            for seed_dir in sorted((s / "ref" / part).glob("*/seed_*")):
                dst = out / arm / seed_dir.parent.name / seed_dir.name
                if dst.exists() or not complete(seed_dir):
                    continue
                tmp = dst.with_name(f"{dst.name}.tmp{os.getpid()}")  # renamed whole: a crash or a racing run leaves no half fold
                (tmp / "predictions").mkdir(parents=True)
                for f in (seed_dir / "predictions").iterdir():
                    os.link(f, tmp / "predictions" / f.name)
                try:
                    tmp.rename(dst)
                    added[arm] += 1
                except OSError:
                    shutil.rmtree(tmp)
    n = {a: len(list((out / a).glob("*/seed_*"))) for a in arms.values()}
    print(f"added {added}; now {n} seed folds")


if __name__ == "__main__":
    if sys.argv[1] == "--fp32":
        main(sys.argv[2], sys.argv[3], sys.argv[4:], {"paired": "gpu_fp32"}, "mol1-*")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3:])
