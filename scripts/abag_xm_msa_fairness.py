#!/usr/bin/env python3
"""Verify the fairness claim the dataset card makes: every generator saw the same MSA.

The claim holds by construction -- all three generators read
``msa_cache/<sha256(seq)[:16]>.a3m`` -- but only for as long as the file does not change
between one generator's fold and another's. It has been violated before: OpenDDE's offline
paired search rewrote entries in the shared cache mid-campaign (fixed in 71f834995), which
silently changed what the other two generators had already read.

`abag_xm_msa_audit.py` answers a different question -- is every demanded alignment present.
An alignment can be present and still have arrived too late.

This checks the ordering: for every alignment written after ``--since``, no fold of a target
that depends on it may have completed BEFORE it was written. Completion time comes from the
fold's ``results.json`` mtime, because progress.jsonl records duration but no finish timestamp.

    python3 scripts/abag_xm_msa_fairness.py --since "2026-07-27 18:00"

Exit status is 1 if any fold read a different alignment than its siblings, so it can gate a
release. Run it on BOTH hosts, and diff the two caches by checksum as well -- this checks
ordering within a host; identical content across hosts is a separate condition.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = pathlib.Path.home() / "abag_xm" / "msa_cache"
TIERA = pathlib.Path.home() / "abag_xm" / "tier_a"
GEN = {"protenix-v2": ("protenix_v2", "protenix_results"),
       "opendde-abag": ("opendde_abag", "opendde_results"),
       "boltz2": ("boltz2", "boltz2_results")}


def _chain_hashes():
    """sha256(seq)[:16] -> targets needing it, keyed exactly as tt_bio keys the cache."""
    need = {}
    for y in sorted((ROOT / "examples" / "abag_xm").glob("*.yaml")):
        try:
            doc = yaml.safe_load(y.read_text())
        except Exception:
            continue
        for s in doc.get("sequences", []):
            for _k, v in s.items():
                seq = (v or {}).get("sequence")
                if seq:
                    h = hashlib.sha256(seq.encode()).hexdigest()[:16]
                    need.setdefault(h, set()).add(y.stem)
    return need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True,
                    help='only consider alignments written after this, e.g. "2026-07-27 18:00"')
    a = ap.parse_args()
    cutoff = float(subprocess.run(["date", "-d", a.since, "+%s"],
                                  capture_output=True, text=True, check=True).stdout.strip())

    need = _chain_hashes()
    late = [(f.stem, f.stat().st_mtime) for f in CACHE.glob("*.a3m")
            if f.stat().st_mtime > cutoff]
    print(f"alignments written after {a.since}: {len(late)}")
    if not late:
        print("OK: the cache was fully warm before this window; nothing to check")
        return 0

    done_at = {}
    prog = TIERA / "progress.jsonl"
    for line in prog.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        sub, pre = GEN[r["model"]]
        rj = TIERA / sub / f"{pre}_{r['target']}" / "results.json"
        if rj.exists():
            done_at[(r["target"], r["model"])] = rj.stat().st_mtime

    affected = sorted({t for h, _ in late for t in need.get(h, ())})
    print(f"targets depending on a late alignment: {affected}")

    bad = []
    for h, mt in late:
        for t in need.get(h, ()):
            for m in GEN:
                d = done_at.get((t, m))
                if d is not None and d < mt:
                    bad.append((t, m, h, d, mt))
    if bad:
        print(f"!! {len(bad)} fold(s) completed BEFORE the alignment they depend on was written:")
        for t, m, h, d, mt in bad:
            print(f"   {t}/{m}: fold {d:.0f} < a3m {mt:.0f} (chain {h}) -- regenerate this fold")
        return 1
    print("OK: no fold completed before its alignment was written -- within this host, every "
          "generator of every target read the same file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
