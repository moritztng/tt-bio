#!/usr/bin/env python3
"""Audit how the label scripts identify chains, and how many targets that costs.

`abag_xm_interface_lddt.py` and `abag_xm_cdr_rmsd.py` each carry their own copy of a
`_find_chain` that requires an EXACT sequence match between a YAML chain and a chain of
the ground-truth structure. Natives routinely carry a few extra residues past the end of
the construct, so the match fails and both label families are lost for that target --
silently: interface_lddt records an `_error` string and cdr_rmsd records `cdrs: {}`, and
nothing gates on either. It surfaced only when a ranker join showed 100 empty cells.

This reports the real cost and, importantly, whether recovery needs realignment: it prints
the start-offset distribution over every prefix match. An offset of 0 everywhere means a
prefix matcher plus truncation to the overlap is sufficient and no index shifting is
involved.

    python3 scripts/abag_xm_chain_match_audit.py

Exit 1 if any target would lose a label family, so it can gate a release.
"""
import collections
import importlib.util
import sys
from pathlib import Path

import yaml as _yaml

ROOT = Path(__file__).resolve().parent.parent
GT = ROOT / "examples" / "ground_truth_structures"
YAML_DIR = ROOT / "examples" / "abag_xm"
NEEDED = ("A", "H", "L")


def _lddt_module():
    spec = importlib.util.spec_from_file_location(
        "abag_xm_interface_lddt", ROOT / "scripts" / "abag_xm_interface_lddt.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    il = _lddt_module()
    stat, offsets, losers = collections.Counter(), collections.Counter(), []
    for yf in sorted(YAML_DIR.glob("*.yaml")):
        native = GT / f"{yf.stem}.cif"
        if not native.exists():
            stat["no native structure"] += 1
            continue
        doc = _yaml.safe_load(yf.read_text())
        wanted = {s["protein"]["id"]: s["protein"]["sequence"]
                  for s in doc["sequences"]
                  if s.get("protein", {}).get("id") in NEEDED}
        try:
            native_seqs = {k: il._seq_of(v) for k, v in il._load(str(native)).items()}
        except Exception:
            stat["native failed to load"] += 1
            continue
        kind = {}
        for cid, seq in wanted.items():
            if any(s == seq for s in native_seqs.values()):
                kind[cid] = "exact"
            elif any(seq and seq in s for s in native_seqs.values()):
                kind[cid] = "prefix"
                for s in native_seqs.values():
                    if seq and seq in s:
                        offsets[s.index(seq)] += 1
                        break
            else:
                kind[cid] = "none"
        if all(v == "exact" for v in kind.values()):
            stat["all chains exact -- labelled today"] += 1
        elif any(v == "none" for v in kind.values()):
            stat["a chain matches NEITHER -- needs alignment"] += 1
            losers.append((yf.stem, kind))
        else:
            stat["exact-or-prefix -- recoverable by prefix match"] += 1
            losers.append((yf.stem, kind))

    total = sum(stat.values())
    print(f"targets: {total}")
    for k, v in stat.most_common():
        print(f"  {k}: {v} ({100 * v / total:.0f}%)")
    print(f"\nstart-offset over every prefix match: {dict(offsets)}"
          f"{'  <- all zero: truncation suffices, no realignment' if set(offsets) <= {0} else ''}")
    print(f"\ntargets losing interface_lddt + cdr_rmsd today: {len(losers)}")
    for name, kind in losers[:10]:
        print(f"  {name}: {kind}")
    if len(losers) > 10:
        print(f"  ... and {len(losers) - 10} more")
    return 1 if losers else 0


if __name__ == "__main__":
    sys.exit(main())
