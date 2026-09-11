"""Turn a cdk2x2 protein rung into a Nesso-1 affinity input at the same residue count.

Nesso-1 scores a protein/ligand pair through `tt-bio affinity`, not `predict`, so it cannot
walk the structure rungs directly. The protein here is the SAME tiled CDK2 sequence every
other rung uses, so a Nesso-1 ladder and a folding ladder are the same residue axis; the
ligand is the one already committed under perf/nesso1/inputs, so the ladder varies exactly
one thing. Affinity reads no alignment, which is why these carry no `msa:` line -- and it is
why a Nesso-1 rung is not comparable to a folding rung at the same number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

LIGAND = "NCC(=O)NCC(=O)NCC(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)O"  # perf/nesso1/inputs/screen
TMPL = ("sequences:\n  - protein:\n      id: A\n      sequence: {seq}\n"
        "  - ligand:\n      id: B\n      smiles: '{lig}'\n"
        "properties:\n  - affinity:\n      binder: B\n")


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for src in sys.argv[2:]:
        spec = yaml.safe_load(Path(src).read_text())
        seq = spec["sequences"][0]["protein"]["sequence"]
        name = f"nesso1_{len(seq)}"
        (out / f"{name}.yaml").write_text(TMPL.format(seq=seq, lig=LIGAND))
        print(f"wrote {name}.yaml ({len(seq)} aa) in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
