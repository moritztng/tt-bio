"""Build an MSA directory whose per-chain a3m files are truncated to a depth cap.

opendde-abag and protenix-v2 ignore ``--max_msa_seqs`` -- the worker resolves a
chain's a3m without a depth argument -- so the only way to bound MSA depth for
them is to physically truncate the a3m records and point ``--msa_dir`` at the
result. This builds that directory for one target, leaving the source cache
untouched.

The paired MSA is capped too, and this matters more than it looks. The effective
MSA depth the model sees is ``unpaired_block + paired_block``, so capping only the
unpaired side leaves the total unbounded: measured on the Galaxy, targets whose
paired directory was absent had it fetched mid-fold from api.colabfold.com and ran
at depth 5500-5902 under a nominal cap of 4096 (+1404 to +1806 rows). That is both
a bigger footprint than intended and a *network-dependent, irreproducible* one.

So a target with no pre-existing paired directory is an error here rather than
something to paper over -- fold it once to materialise the pairing, then cap.
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
    ap.add_argument("--paired-cap", type=int, default=0,
                    help="max paired-MSA rows to keep (0 = drop the paired block entirely)")
    ap.add_argument("--allow-missing-paired", action="store_true",
                    help="proceed when no paired dir exists; the fold will then search online, "
                         "giving an unbounded and irreproducible depth")
    args = ap.parse_args()

    target = args.target_yaml.stem
    args.out_msa_dir.mkdir(parents=True, exist_ok=True)
    n_chains = len(chain_sequences(args.target_yaml))

    for seq in dict.fromkeys(chain_sequences(args.target_yaml)):
        digest = hashlib.sha256(seq.encode()).hexdigest()[:16]
        src = args.source_msa_dir / f"{digest}.a3m"
        if not src.exists():
            raise SystemExit(f"missing a3m for chain len={len(seq)} ({digest}) in {args.source_msa_dir}")
        text, kept, total = truncate_a3m(src.read_text(), args.cap)
        (args.out_msa_dir / f"{digest}.a3m").write_text(text)
        print("  %s len=%-5d %6d -> %6d records" % (digest, len(seq), total, kept))

    # Paired MSA dirs are named "<target>_paired*" by tt-bio. Copy them so no online
    # search runs, then bound the paired block: pair.a3m holds one record per chain
    # per pairing, and the featurizer takes the min row count across chains and drops
    # the query (protenix_data.py:395-397), so keeping n_chains*(P+1) records leaves
    # P paired rows.
    paired_dirs = list(args.source_msa_dir.glob(f"{target}_paired*"))
    if not paired_dirs and not args.allow_missing_paired:
        raise SystemExit(
            f"no paired MSA dir for {target} in {args.source_msa_dir}. Folding would search "
            "api.colabfold.com mid-run, making the effective MSA depth unbounded and "
            "irreproducible (measured: +1404 to +1806 rows over the cap). Materialise the "
            "pairing first, or pass --allow-missing-paired to accept that."
        )
    for paired in paired_dirs:
        dest = args.out_msa_dir / paired.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(paired, dest)
        pair_a3m = dest / "pair.a3m"
        if pair_a3m.exists():
            text, kept, total = truncate_a3m(pair_a3m.read_text(), n_chains * (args.paired_cap + 1))
            pair_a3m.write_text(text)
            print("  paired dir %s: %d -> %d records (~%d paired rows)"
                  % (paired.name, total, kept, max(0, kept // max(1, n_chains) - 1)))
        else:
            print("  paired dir %s (no pair.a3m to cap)" % paired.name)

    print("cap=%d paired_cap=%d msa_dir ready: %s"
          % (args.cap, args.paired_cap, args.out_msa_dir))


if __name__ == "__main__":
    main()
