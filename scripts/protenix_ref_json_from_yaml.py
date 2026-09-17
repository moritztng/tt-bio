#!/usr/bin/env python3
"""Build an official-Protenix inference JSON from a tt-bio input yaml, wiring each
protein chain to the a3m the tt-bio device leg folds for that chain.

Why this exists: every other protenix reference script lets Protenix run its own MSA
search (``use_msa=True`` against protenix-server.com). A multimer parity leg cannot do
that -- the device folds per-chain alignments out of a cache, and a reference that
searches its own would compare two different inputs. Protenix takes a per-chain
``unpairedMsaPath``/``pairedMsaPath`` (runner/msa_search.py::need_msa_search), and with
those set it skips the search entirely, so pointing them at the leg's own a3m files
makes the reference read the SAME alignment bytes as the device.

The a3m for a chain is ``<msa_dir>/<seq_hash>.a3m``, the same sequence-hash cache
contract tt_bio.cache.seq_hash / full_parity_gate._stage_msa use, so the fixture's
``msa/`` directory drops straight in.

Usage:
  scripts/protenix_ref_json_from_yaml.py examples/abag_xm/9ncy.yaml 9ncy \
      --msa-dir docs/implementation-parity-data/ref-fixtures/protenix-v2/9ncy/<tag>/msa \
      --msa-dir-at /root/work/9ncy_msa \
      --out /tmp/prot_9ncy.json

``--msa-dir`` is where the a3ms are read from now (to check they exist and record their
depth); ``--msa-dir-at`` is the path they will have when Protenix runs, which is what
goes into the JSON. They are the same directory unless the JSON is built on one host and
folded on another, which is the vast.ai reference case.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def seq_hash(seq: str) -> str:
    """tt_bio.cache.seq_hash, inlined so this script runs on a box without tt-bio."""
    return hashlib.sha256(seq.encode()).hexdigest()[:16]


def chains_from_yaml(path: Path) -> list[tuple[str, str]]:
    doc = yaml.safe_load(path.read_text())
    out = []
    for entry in doc.get("sequences", []):
        if "protein" not in entry:
            raise SystemExit(f"{path}: only protein chains are supported, got {sorted(entry)}")
        c = entry["protein"]
        out.append((str(c["id"]), str(c["sequence"])))
    if not out:
        raise SystemExit(f"{path}: no protein chains")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml_path", type=Path)
    ap.add_argument("name", help="task name; Protenix names its output files after it")
    ap.add_argument("--msa-dir", type=Path, required=True,
                    help="directory holding <seq_hash>.a3m per protein chain, readable now")
    ap.add_argument("--msa-dir-at", default="",
                    help="the msa dir's path at fold time (default: --msa-dir)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    at = args.msa_dir_at or str(args.msa_dir)
    sequences = []
    for cid, seq in chains_from_yaml(args.yaml_path):
        h = seq_hash(seq)
        a3m = args.msa_dir / f"{h}.a3m"
        if not a3m.exists():
            raise SystemExit(f"chain {cid}: no a3m for seq_hash {h} under {args.msa_dir}")
        rows = a3m.read_text().count(">")
        query = a3m.read_text().split("\n")[1].strip()
        if query.replace("-", "").upper() != seq.upper():
            raise SystemExit(
                f"chain {cid}: {a3m.name} query row is not this chain's sequence "
                f"({len(query)} vs {len(seq)} residues)")
        print(f"chain {cid}: {len(seq)} res, {a3m.name}, {rows} rows", file=sys.stderr)
        sequences.append({"proteinChain": {
            "sequence": seq,
            "count": 1,
            # unpaired only: the device assembles these per-chain alignments
            # block-diagonally (tt_bio/protenix_data.py) with no paired block.
            "unpairedMsaPath": f"{at}/{h}.a3m",
        }})

    task = [{"name": args.name, "sequences": sequences, "modelSeeds": [], "assembly_id": None}]
    args.out.write_text(json.dumps(task, indent=4) + "\n")
    print(f"wrote {args.out} ({len(sequences)} chains, msa at {at})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
