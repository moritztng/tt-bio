#!/usr/bin/env python3
"""Warm the MSA cache for the chains no fold can supply on its own host.

Five Tier-A targets have chains with no cached a3m. On qb1 a fold would simply search
inline; on qb2 there is neither `colabfold_search` nor a real `msa_db`, so those folds
fail in ~9 s with "colabfold_search not found" and can never complete there. This script
generates the missing alignments on qb1 so they can be copied to qb2.

It calls the SAME function the fold calls (`compute_msa_offline`), with the same
defaults (`use_env=False`, `pairing_strategy="greedy"`, `pair=True`) and the same
per-hash flock, grouping each target's missing chains exactly as a fold's `to_gen`
would. Anything else risks producing alignments unlike the other 346 in the cache,
which is precisely the input homogeneity the fairness contract depends on.

    OMP_NUM_THREADS=8 PYTHONPATH=<wt> python3 scripts/abag_xm_warm_missing_msa.py
"""
import fcntl
import hashlib
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tt_bio.main import compute_msa_offline  # noqa: E402

MSA_DIR = Path.home() / "abag_xm" / "msa_cache"
DB_PATH = str(Path.home() / ".boltz" / "msa_db")
YAML_DIR = ROOT / "examples" / "abag_xm"


def missing_chains(target):
    d = yaml.safe_load((YAML_DIR / f"{target}.yaml").open())
    out = {}
    for ent in d.get("sequences", []):
        for kind, v in ent.items():
            seq = v.get("sequence")
            if not seq or kind != "protein":
                continue
            h = hashlib.sha256(seq.encode()).hexdigest()[:16]
            if not (MSA_DIR / f"{h}.a3m").exists() and not (MSA_DIR / f"{h}.csv").exists():
                out[h] = seq
    return out


def main():
    targets = sys.argv[1:] or ["9l9y", "9ly2", "9msc", "9mz8", "9q1l"]
    for target in targets:
        to_gen = missing_chains(target)
        if not to_gen:
            print(f"[warm] {target}: nothing missing", flush=True)
            continue
        # Same lock discipline as prepare_features: sorted-hash order, so a live fold
        # searching the same sequence cannot race this or deadlock against it.
        locks = []
        try:
            for h in sorted(to_gen):
                lf = open(MSA_DIR / f".{h}.lock", "w")
                fcntl.flock(lf, fcntl.LOCK_EX)
                locks.append(lf)
            to_gen = {h: s for h, s in to_gen.items()
                      if not (MSA_DIR / f"{h}.a3m").exists()
                      and not (MSA_DIR / f"{h}.csv").exists()}
            if not to_gen:
                print(f"[warm] {target}: produced by a concurrent fold while waiting", flush=True)
                continue
            t0 = time.time()
            print(f"[warm] {target}: searching {len(to_gen)} chain(s) "
                  f"{sorted(to_gen)} lens={[len(s) for s in to_gen.values()]}", flush=True)
            compute_msa_offline(to_gen, target, MSA_DIR, DB_PATH,
                                use_env=False, pairing_strategy="greedy")
            made = [h for h in to_gen if (MSA_DIR / f"{h}.a3m").exists()]
            print(f"[warm] {target}: {len(made)}/{len(to_gen)} a3m written in "
                  f"{time.time() - t0:.0f}s", flush=True)
            if len(made) != len(to_gen):
                print(f"[warm] {target}: STILL MISSING "
                      f"{sorted(set(to_gen) - set(made))}", flush=True)
        finally:
            for lf in locks:
                fcntl.flock(lf, fcntl.LOCK_UN)
                lf.close()
    print("[warm] done", flush=True)


if __name__ == "__main__":
    main()
