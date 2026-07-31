"""Build an MSA directory whose per-chain a3m files are truncated to a depth cap.

opendde-abag and protenix-v2 ignore ``--max_msa_seqs`` -- the worker resolves a
chain's a3m without a depth argument -- so the only way to bound MSA depth for
them is to physically truncate the a3m records and point ``--msa_dir`` at the
result. This builds that directory for one target, leaving the source cache
untouched.

The paired MSA directory is copied verbatim. Truncating it would change which
species pairings the model sees (a different, larger scientific perturbation
than trimming the tail of a single-chain search), and its block is min-truncated
across chains anyway, so it is already shallow.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil

import yaml


def chain_sequences(target_yaml: pathlib.Path) -> list[str]:
    """Protein chain sequences of a target, one entry per copy."""
    spec = yaml.safe_load(target_yaml.read_text())
    seqs: list[str] = []
    for entry in spec.get("sequences", []):
        protein = entry.get("protein")
        if protein and isinstance(protein.get("sequence"), str):
            ids = protein.get("id")
            copies = len(ids) if isinstance(ids, list) else 1
            seqs.extend([protein["sequence"].strip()] * copies)
    return seqs


def truncate_a3m(text: str, cap: int) -> tuple[str, int, int]:
    """Keep the first ``cap`` whole records. Returns (text, kept, original)."""
    # An a3m record is a '>' header plus its (possibly wrapped) sequence lines.
    records: list[list[str]] = []
    for line in text.splitlines(keepends=True):
        if line.startswith(">"):
            records.append([line])
        elif records:
            records[-1].append(line)
    kept = records[:cap]
    return "".join("".join(r) for r in kept), len(kept), len(records)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target_yaml", type=pathlib.Path)
    ap.add_argument("source_msa_dir", type=pathlib.Path)
    ap.add_argument("out_msa_dir", type=pathlib.Path)
    ap.add_argument("--cap", type=int, required=True, help="max a3m records per chain")
    args = ap.parse_args()

    target = args.target_yaml.stem
    args.out_msa_dir.mkdir(parents=True, exist_ok=True)

    for seq in dict.fromkeys(chain_sequences(args.target_yaml)):
        digest = hashlib.sha256(seq.encode()).hexdigest()[:16]
        src = args.source_msa_dir / f"{digest}.a3m"
        if not src.exists():
            raise SystemExit(f"missing a3m for chain len={len(seq)} ({digest}) in {args.source_msa_dir}")
        text, kept, total = truncate_a3m(src.read_text(), args.cap)
        (args.out_msa_dir / f"{digest}.a3m").write_text(text)
        print("  %s len=%-5d %6d -> %6d records" % (digest, len(seq), total, kept))

    # Paired MSA dirs are named "<target>_paired*" by tt-bio; copy them as-is so
    # no online search is attempted.
    for paired in args.source_msa_dir.glob(f"{target}_paired*"):
        dest = args.out_msa_dir / paired.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(paired, dest)
        print("  copied paired dir %s" % paired.name)

    print("cap=%d msa_dir ready: %s" % (args.cap, args.out_msa_dir))


if __name__ == "__main__":
    main()
