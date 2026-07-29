#!/usr/bin/env python3
"""Audit the MSA cache against what the Tier-A targets actually demand, and optionally
fetch what is missing from another host.

Turn 37 compared the two hosts' caches to each other, found them byte-identical, and
concluded no MSA search could occur. Both were missing the same 7 required alignments:
comparing two caches proves consistency, not completeness. This checks the cache against
demand -- every protein chain of every target, keyed the way tt_bio keys it
(``sha256(seq)[:16].a3m``) -- which is the only check that can catch that.

It matters because the hosts are not equally capable: qb1 has colabfold_search and the
real msa_db and will search inline, while qb2 has neither and a missing alignment there
is a fold that can never succeed.

    python3 scripts/abag_xm_msa_audit.py                    # audit this host
    python3 scripts/abag_xm_msa_audit.py --fetch-from qb1   # and rsync what is missing

Exit status is 1 while anything is missing, so it can gate a launch.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
YAML_DIR = ROOT / "examples" / "abag_xm"
MSA_DIR = Path.home() / "abag_xm" / "msa_cache"
SLICES = ROOT / "docs" / "implementation-parity-data" / "abag-xm-tier-a-slices.json"
# slices 0-3 are tt-quietbox2's share, 4-7 tt-quietbox's (see abag_xm_tiera_launch.sh)
HOST_OF_SLICE = {i: ("tt-quietbox2" if i < 4 else "tt-quietbox") for i in range(8)}


def demand():
    """{seq_hash: (target, chain_id, length)} for every protein chain of every target."""
    need = {}
    for f in sorted(YAML_DIR.glob("*.yaml")):
        d = yaml.safe_load(f.open())
        for ent in d.get("sequences", []):
            for kind, v in ent.items():
                seq = v.get("sequence")
                if seq and kind == "protein":
                    h = hashlib.sha256(seq.encode()).hexdigest()[:16]
                    need.setdefault(h, (f.stem, v.get("id"), len(seq)))
    return need


def owning_host():
    if not SLICES.exists():
        return {}
    sl = json.load(SLICES.open())["slices_8"]
    return {t: HOST_OF_SLICE[int(k)] for k, v in sl.items() for t in v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-from", metavar="HOST",
                    help="rsync missing a3m from HOST's cache (read-only on the source)")
    a = ap.parse_args()

    need = demand()
    owner = owning_host()
    have = {f.stem for f in MSA_DIR.glob("*.a3m")} | {f.stem for f in MSA_DIR.glob("*.csv")}
    missing = sorted(set(need) - have)
    print(f"targets {len(list(YAML_DIR.glob('*.yaml')))} | chains needed {len(need)} | "
          f"cached {len(have)} | present {len(need) - len(missing)} | MISSING {len(missing)}")

    if missing and a.fetch_from:
        MSA_DIR.mkdir(parents=True, exist_ok=True)
        src = [f"{a.fetch_from}:{MSA_DIR}/{h}.a3m" for h in missing]
        print(f"[fetch] rsync {len(src)} a3m from {a.fetch_from}")
        # Not --ignore-missing-args in one shot: a per-file call reports which are absent
        # on the source too, which is the information the operator actually needs.
        for h, s in zip(missing, src):
            r = subprocess.run(["rsync", "-a", s, str(MSA_DIR) + "/"],
                               capture_output=True, text=True)
            print(f"  {h}: {'ok' if r.returncode == 0 else 'NOT ON SOURCE'}")
        have = {f.stem for f in MSA_DIR.glob("*.a3m")} | {f.stem for f in MSA_DIR.glob("*.csv")}
        missing = sorted(set(need) - have)
        print(f"[fetch] MISSING now {len(missing)}")

    for h in missing:
        target, chain, length = need[h]
        print(f"  MISSING {h}  {target} chain {chain} len {length}  "
              f"-> owned by {owner.get(target, '?')}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
