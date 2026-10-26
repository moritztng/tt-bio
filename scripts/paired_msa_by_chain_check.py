"""Grade the per-chain paired MSA search against stock colabfold_search on a box with the DB.

Every case is searched twice from empty caches: stock (``compute_msa_offline(pair=True)``,
one colabfold_search call) and by chain (``chain_cache=``, the cache shared across cases in
order, so a later case that repeats a chain reuses its search). The per-chain a3m files must be
byte-identical. Prints one line per case with both walls and writes the table as JSON.

    python scripts/paired_msa_by_chain_check.py --db ~/japanfold/msa/db --out /tmp/pmc \\
        examples/abag_xm/9wpm.yaml examples/abag_xm/9wpm.yaml:0+examples/abag_xm/9gei.yaml:1 ...

A case is a YAML (its protein chains) or ``file:i+file:j+...`` picking chains by index.
CPU only; needs colabfold_search on PATH.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import yaml

from tt_bio.cache import seq_hash
from tt_bio.main import compute_msa_offline


def _chains(path):
    d = yaml.safe_load(open(path))
    return [x["protein"]["sequence"] for x in d["sequences"] if "protein" in x]


def case_seqs(spec):
    if "+" not in spec and ":" not in spec:
        return list(dict.fromkeys(_chains(spec)))
    return [_chains(p)[int(i)] for p, i in (part.rsplit(":", 1) for part in spec.split("+"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("cases", nargs="+")
    a = ap.parse_args()
    shutil.rmtree(a.out, ignore_errors=True)
    cache = a.out / "chain-cache"
    rows = []
    for n, spec in enumerate(a.cases):
        seqs = {seq_hash(s): s for s in case_seqs(spec)}
        got = {}
        for mode in ("stock", "chain"):
            d = a.out / f"{n}_{mode}"
            d.mkdir(parents=True)
            t = time.time()
            compute_msa_offline(seqs, f"case{n}", d, a.db, pair=True,
                                chain_cache=cache if mode == "chain" else None)
            got[mode] = (time.time() - t, [(d / f"{h}.a3m").read_bytes() for h in seqs])
        same = got["stock"][1] == got["chain"][1]
        row = {"case": spec, "lengths": [len(s) for s in seqs.values()],
               "stock_s": round(got["stock"][0], 1), "chain_s": round(got["chain"][0], 1),
               "identical": same,
               "sha": hashlib.sha256(b"".join(got["stock"][1])).hexdigest()[:16]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (a.out / "result.json").write_text(json.dumps(rows, indent=1))
    raise SystemExit(0 if all(r["identical"] for r in rows) else 1)


if __name__ == "__main__":
    main()
