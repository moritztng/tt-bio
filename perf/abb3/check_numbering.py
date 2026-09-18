"""Confirm the released region labels really are ANARCI IMGT, independently of their code.

Their data pipeline calls ``get_region(definition="imgt")`` on ``exs.sabdab``, which is
Exscientia-internal and not released, so the labels cannot be regenerated. They can still be
checked, and the IMGT scheme pins its own boundaries with conserved anchors: CDR3 is positions
105-117, so it is immediately preceded by the Cys at 104 and followed by the Trp at 118 (heavy)
or the Phe at 118 (light). Kabat and Chothia put their CDR3 boundaries elsewhere, so an anchor
that holds on every structure is evidence for IMGT specifically rather than for "some numbering".

Usage: check_numbering.py <output-dir-from-zenodo> [--variant base-loss]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from tt_bio.antibody_rmsd import residue_indices

EXPECTED = {("cdrh3", "before"): "CYS", ("cdrh3", "after"): "TRP",
            ("cdrl3", "before"): "CYS", ("cdrl3", "after"): "PHE"}


def residue_names(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith("ATOM"):
            out.setdefault((line[21], int(line[22:26]), line[26]), line[17:20].strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--variant", default="base-loss")
    args = ap.parse_args()

    root = args.root / args.variant
    seen: Counter = Counter()
    for pt in sorted((root / "plddt").glob("*.pt")):
        regions = list(torch.load(pt, weights_only=False)["region"])
        names = residue_names(root / "true" / f"{pt.stem}.pdb")
        by_index = {i: names[k] for k, i in residue_indices(list(names), regions).items()}
        for cdr in ("cdrh3", "cdrl3"):
            at = [i for i, r in enumerate(regions) if r == cdr]
            if not at:
                continue
            seen[(cdr, "before", by_index.get(at[0] - 1, "missing"))] += 1
            seen[(cdr, "after", by_index.get(at[-1] + 1, "missing"))] += 1

    ok = True
    for (cdr, side), want in EXPECTED.items():
        counts = {res: n for (c, s, res), n in seen.items() if c == cdr and s == side}
        total = sum(counts.values())
        hit = counts.get(want, 0)
        ok &= hit == total
        others = {k: v for k, v in counts.items() if k != want}
        print(f"{cdr} {side:6s} expect {want}: {hit}/{total}" + (f"  other={others}" if others else ""))
    print("\nIMGT anchors hold on every structure" if ok else "\nANCHOR MISMATCH -- regions are not IMGT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
