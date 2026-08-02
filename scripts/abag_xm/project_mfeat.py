"""Project each AbAg-XM target's opendde MSA depth and ``m_feat`` DRAM footprint.

Writes one JSON record per target for ``analyze_mfeat_panel.py`` to report on.

``m_feat`` is the c_m=128 MSA projection, and its size is
``depth * pad32(tokens) * 128 * 2`` (bf16). Validated against three measured
Wormhole allocator refusals to the byte (9q7y 1.133 GiB, 9ly6 1.140 GiB,
9j4c 1.094 GiB).

Getting ``depth`` right is the whole game, and it has two parts
(``protenix_data.build_complex_features``):

* the **unpaired** block, padded to ``max_d`` = the deepest single chain's a3m;
* the **paired** block, stacked on top, contributing ``min_pd - 1`` rows -- the
  minimum row count across chains, with the query dropped.

A ``pair.a3m`` holds one record *per chain* per pairing, so its paired-row count
is ``records / n_chains - 1``, NOT its record count. Counting records instead
overstates depth by ~3x on the paired block, which is what made an earlier pass
misread this projection as a 20-28% overestimate and chase a nonexistent dedup
effect. Chain a3m files are named ``sha256(seq)[:16]`` by tt-bio itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import yaml

C_M, DTYPE_BYTES = 128, 2


def pad32(n: int) -> int:
    return ((n + 31) // 32) * 32


def count_records(path: pathlib.Path) -> int:
    """Number of '>' records in an a3m."""
    n = 0
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                n += 1
    return n


def chain_sequences(target_yaml: pathlib.Path) -> list[str]:
    spec = yaml.safe_load(target_yaml.read_text())
    seqs: list[str] = []
    for entry in spec.get("sequences", []):
        protein = entry.get("protein")
        if protein and isinstance(protein.get("sequence"), str):
            ids = protein.get("id")
            copies = len(ids) if isinstance(ids, list) else 1
            seqs.extend([protein["sequence"].strip()] * copies)
    return seqs


def paired_rows(cache: pathlib.Path, target: str, n_chains: int) -> int:
    """Paired-block row count: min over chains, query dropped."""
    for d in sorted(cache.glob(f"{target}_paired*")):
        pair_a3m = d / "pair.a3m"
        if pair_a3m.exists():
            records = count_records(pair_a3m)
            return max(0, records // max(1, n_chains) - 1)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("panel_dir", type=pathlib.Path, help="dir of <target>.yaml files")
    ap.add_argument("msa_cache", type=pathlib.Path)
    ap.add_argument("out_json", type=pathlib.Path)
    args = ap.parse_args()

    rows = []
    for target_yaml in sorted(args.panel_dir.glob("*.yaml")):
        target = target_yaml.stem
        seqs = chain_sequences(target_yaml)
        tokens = sum(len(s) for s in seqs)

        depths, missing = [], 0
        for seq in dict.fromkeys(seqs):
            digest = hashlib.sha256(seq.encode()).hexdigest()[:16]
            a3m = args.msa_cache / f"{digest}.a3m"
            if a3m.exists():
                depths.append(count_records(a3m))
            else:
                missing += 1

        max_chain_depth = max(depths) if depths else 0
        paired = paired_rows(args.msa_cache, target, len(seqs))
        depth = max_chain_depth + paired
        rows.append({
            "target": target,
            "tokens": tokens,
            "pad": pad32(tokens),
            "n_chains": len(seqs),
            "max_chain_depth": max_chain_depth,
            "paired": paired,
            "depth": depth,
            "missing_a3m": missing,
            "mfeat_gib": depth * pad32(tokens) * C_M * DTYPE_BYTES / 2**30,
        })

    args.out_json.write_text(json.dumps(rows, indent=1))
    complete = [r for r in rows if r["missing_a3m"] == 0 and r["depth"] > 0]
    print("targets total      : %d" % len(rows))
    print("with complete MSAs : %d" % len(complete))
    print("incomplete/missing : %d" % (len(rows) - len(complete)))
    print("wrote %s" % args.out_json)


if __name__ == "__main__":
    main()
