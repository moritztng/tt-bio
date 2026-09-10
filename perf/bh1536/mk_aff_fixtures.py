#!/usr/bin/env python3
"""Affinity-model fixtures for the 1536 ladder: the same tandem-repeat CDK2 chain the
predict rungs use, with MTX bound so nesso1 has a binder to score.

The ligand is the point, not decoration. nesso1 tokenises ligand heavy atoms on top of the
polymer, so 1536 residues + MTX is 1569 tokens -- a token count that is NOT a multiple of 32,
which is exactly the arithmetic the campaign asks somebody to test (the token axis has to
bucket to 32 and an unmasked tail has cost 72x here before).
"""
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
FIX = WT / "perf" / "size512" / "fixtures"
OUT = Path(__file__).resolve().parent / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

TEMPLATE = """version: 1
sequences:
  - protein:
      id: A
      sequence: {seq}
      msa: empty
  - ligand:
      id: B
      ccd: MTX
properties:
  - affinity:
      binder: B
"""

for L in [int(x) for x in sys.argv[1:]]:
    src = (FIX / f"cdk2x2_{L}.yaml").read_text()
    seq = src.split("sequence: ", 1)[1].split("\n", 1)[0].strip()
    assert len(seq) == L, f"{L}: fixture sequence is {len(seq)} aa"
    (OUT / f"aff_{L}.yaml").write_text(TEMPLATE.format(seq=seq))
    print(f"aff_{L}.yaml  protein={L} aa  + MTX")
