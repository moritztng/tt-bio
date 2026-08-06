#!/usr/bin/env python3
"""Pull the peer host's saturation labels into this host's tree, so the analysis can run.

The campaign folds on two hosts that share no filesystem, but every §7 number is a join of
labels.json (per-sample DockQ) with results.json (rank -> confidence_score). Both are small;
the structures they were computed from are 200 MB+ per fold and are NOT copied.

Idempotent, additive (no --delete), and safe to run while folds are still writing: an
incomplete fold simply has no labels.json yet.
"""
import socket, subprocess, sys
from pathlib import Path

BASE = Path.home() / "abag_xm" / "saturation"
PEER = {"tt-quietbox": "tt-quietbox2", "tt-quietbox2": "tt-quietbox"}
MODEL_DIRS = ("opendde", "protenix", "boltz2")


def main():
    peer = PEER[socket.gethostname()]
    rc = 0
    for model in MODEL_DIRS:
        (BASE / model).mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["rsync", "-a", "--prune-empty-dirs", "--timeout=120",
             "--include=*/", "--include=labels.json", "--include=results.json",
             "--exclude=*",
             f"{peer}:abag_xm/saturation/{model}/", str(BASE / model) + "/"],
            capture_output=True, text=True)
        print(f"{model}: rc={r.returncode} {r.stderr.strip()[:200]}")
        rc |= r.returncode
    for name in ("progress.jsonl",):
        tag = "qb2" if peer.endswith("2") else "qb1"
        r = subprocess.run(["scp", "-q", f"{peer}:abag_xm/saturation/{name}",
                            str(BASE / f"progress_{tag}.jsonl")], capture_output=True, text=True)
        print(f"{name} -> progress_{tag}.jsonl: rc={r.returncode}")
        rc |= r.returncode
    n = sum(1 for _ in BASE.glob("*/*/labels.json"))
    print(f"labels present locally: {n}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
