#!/usr/bin/env python3
"""Emit matched benchmark inputs for the tt-bio and the upstream CUDA stacks.

One tt-bio YAML in, two things out:

  * a copy of the YAML (what ``tt-bio predict`` runs), repeated ``--repeats`` times so a
    single process folds the same target warm-up + N times, and
  * a Protenix/OpenDDE ``input.json`` whose every protein chain points at a
    ``precomputed_msa_dir`` holding the *same* a3m file tt-bio consumed.

The point is that the MSA on both sides is one file, not two files that ought to match.
tt-bio caches unpaired per-chain alignments as ``{sha256(sequence)[:16]}.a3m`` (see
``_resolve_a3m_text`` in tt_bio/main.py); Protenix reads ``<dir>/non_pairing.a3m`` and
``<dir>/pairing.a3m`` (protenix/data/msa/msa_featurizer.py). So we hardlink/copy the tt-bio
cache entry into place as ``non_pairing.a3m`` and write no ``pairing.a3m``.

Single-chain targets are byte-identical by construction and ``--verify`` proves it. Multi-chain
targets have no paired block under this layout, which is a legitimate identical config for both
sides but is NOT what either stack does by default -- run those with pairing disabled on both
sides, and say so in the report.

Usage:
    python3 make_inputs.py --yaml ../../examples/615.yaml --msa-cache ~/.boltz/bench_msa \
        --out-dir ./inputs/T615 --repeats 6 --verify
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml


def seq_hash(sequence: str) -> str:
    """tt-bio's a3m cache key: sha256 of the sequence, first 16 hex chars."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def read_chains(yaml_path: Path):
    doc = yaml.safe_load(yaml_path.read_text())
    chains = []
    for entry in doc["sequences"]:
        kind, body = next(iter(entry.items()))
        if kind != "protein":
            raise SystemExit(
                f"{yaml_path.name}: chain type {kind!r} is not handled by this converter. "
                "The benchmark targets are protein-only on purpose."
            )
        chains.append((body["id"], body["sequence"]))
    return chains


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, type=Path, help="tt-bio input YAML")
    ap.add_argument("--msa-cache", required=True, type=Path,
                    help="directory holding tt-bio's {seq_hash}.a3m files")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--repeats", type=int, default=6,
                    help="total folds per process = 1 warm-up + N timed")
    ap.add_argument("--name", default=None, help="target name (default: YAML stem)")
    ap.add_argument("--verify", action="store_true",
                    help="assert the emitted non_pairing.a3m is byte-identical to the cache entry")
    args = ap.parse_args()

    name = args.name or args.yaml.stem
    chains = read_chains(args.yaml)
    if len(chains) > 1:
        print(f"note: {name} has {len(chains)} chains -- emitted unpaired-only, which both stacks "
              f"must be told to honour. See the state doc.", file=sys.stderr)

    tt_dir = args.out_dir / "tt"
    msa_dir = args.out_dir / "msa"
    tt_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)

    # tt-bio side: one YAML per fold, so a single `tt-bio predict <dir>/` does warm-up + repeats
    # in one process (same warm state the GPU side measures).
    for i in range(args.repeats):
        shutil.copyfile(args.yaml, tt_dir / f"{name}_r{i:02d}.yaml")

    # CUDA side: one JSON, chains pointing at per-chain precomputed MSA dirs.
    json_chains = []
    for idx, (chain_id, sequence) in enumerate(chains, start=1):
        h = seq_hash(sequence)
        src = args.msa_cache.expanduser() / f"{h}.a3m"
        if not src.exists():
            raise SystemExit(
                f"missing MSA for chain {chain_id} ({h}.a3m) in {args.msa_cache}. "
                f"Populate the cache first with a tt-bio predict run using --msa_dir."
            )
        chain_msa = msa_dir / str(idx)
        chain_msa.mkdir(parents=True, exist_ok=True)
        dst = chain_msa / "non_pairing.a3m"
        shutil.copyfile(src, dst)
        if args.verify and dst.read_bytes() != src.read_bytes():
            raise SystemExit(f"MSA copy for chain {chain_id} is not byte-identical to {src}")
        json_chains.append({
            "proteinChain": {
                "sequence": sequence,
                "count": 1,
                "id": [chain_id],
                "msa": {"precomputed_msa_dir": str(chain_msa.resolve())},
            }
        })

    # Repeat the target as N entries so the CUDA runner also folds it warm-up + N times in one
    # process. Names must differ or the output dirs collide.
    entries = [{"sequences": json_chains, "name": f"{name}_r{i:02d}"} for i in range(args.repeats)]
    (args.out_dir / "input.json").write_text(json.dumps(entries, indent=2) + "\n")

    print(f"{name}: {len(chains)} chain(s), {args.repeats} folds/process")
    print(f"  tt-bio inputs : {tt_dir}")
    print(f"  cuda input    : {args.out_dir / 'input.json'}")
    print(f"  shared MSA    : {msa_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
