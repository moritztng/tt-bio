#!/usr/bin/env python3
"""Search the MSAs the booth folds with, once, while there is a network.

    python3 demo/sc26/engine/msa_search.py [attract.json ...]

Writes demo/sc26/engine/msa/<hash>.a3m for every sequence in the lists (default: attract.json),
through tt-bio's own search path with `tt-bio predict --use_msa_server`'s defaults (ColabFold
server, unpaired, no environmental database). The chip workers read these and never search, so the
booth needs no network. A sequence that already has its file is skipped.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))


def main():
    from tt_bio.cache import seq_hash
    from tt_bio.main import _generate_esmfold2_a3m
    out = HERE / "msa"
    lists = [Path(a) for a in sys.argv[1:]] or [HERE / "attract.json"]
    seqs = {seq_hash(p["sequence"]): p["sequence"] for f in lists for p in json.loads(f.read_text())}
    need = {h: s for h, s in seqs.items() if not (out / f"{h}.a3m").is_file()}
    if need:
        _generate_esmfold2_a3m(need, "sc26", out, None, False, "https://api.colabfold.com", "greedy",
                               None, None, None)
    for h, s in seqs.items():
        f = out / f"{h}.a3m"
        rows = f.read_text().count(">") if f.is_file() else 0
        print(f"{h} {len(s):4d} aa  {rows:6d} sequences  {f.name if rows else 'MISSING'}")
        if not rows:
            sys.exit(1)


if __name__ == "__main__":
    main()
