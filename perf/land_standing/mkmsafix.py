#!/usr/bin/env python3
"""Tiled-CDK2 fixtures WITH their MSA, at any token count, seeded into an offline cache.

The dividing-k lever's last open question needs a fixture where OpenFold3 is confident at 832
tokens. Every fold this row has taken was `--single_sequence`, which is the obvious reason the
confidence heads read 0.37. There are cached a3m files on this box for this exact sequence family
at 13 lengths, none of them 832, and `build_sweep_fixtures.py` shows the rule: tile the query and
tile every aligned row, so depth stays constant and token count is the only variable.

This rebuilds that here rather than writing into another row's tree, and additionally names each
a3m by the cache key `tt_bio.cache.seq_hash` uses, so a fold can take it with
`--msa_dir <dir> --msa_cache_only` and never touch the network.

298 is the self-check and the precondition: it is one unmodified CDK2 monomer, so if OpenFold3
with this MSA is not confident THERE, no tiling of it will be confident either and the fixture
family is the wrong one.
"""
import hashlib
import pathlib
import sys

WT = pathlib.Path("/home/ttuser/.coworker/wt/land-standing")
sys.path.insert(0, str(WT / "scripts"))
from capacity_fixture import cut                                   # noqa: E402

SRC_A3M = WT / "scripts/gpu_vs_tt/fixtures/prot300.a3m"
OUT = pathlib.Path(sys.argv[1])
SIZES = [int(x) for x in sys.argv[2].split(",")]

lines = SRC_A3M.read_text().rstrip("\n").split("\n")
heads, rows = lines[0::2], lines[1::2]
assert all(h.startswith(">") for h in heads), "a3m is not strict header/row pairs"
query = rows[0]
assert query == query.upper(), "query row carries insertions"
UNIT = len(query)

msa_dir = OUT / "msa"
msa_dir.mkdir(parents=True, exist_ok=True)
for L in SIZES:
    reps = -(-L // UNIT)
    seq = (query * reps)[:L]
    assert len(seq) == L
    a3m = "\n".join("%s\n%s" % (h, cut(r * reps, L)) for h, r in zip(heads, rows)) + "\n"
    assert a3m.split("\n")[1] == seq, "L=%d: a3m query row != target sequence" % L
    h16 = hashlib.sha256(seq.encode()).hexdigest()[:16]
    (msa_dir / ("%s.a3m" % h16)).write_text(a3m)
    (OUT / ("cdk2_msa_%d.yaml" % L)).write_text(
        "# Tiled CDK2 (PDB 1HCL) at %d residues, %d copies of the 298 aa monomer cut to length.\n"
        "# Its MSA is the monomer's alignment tiled the same way, seeded at msa/%s.a3m.\n"
        "sequences:\n  - protein:\n      id: A\n      sequence: %s\n" % (L, reps, h16, seq))
    print("L=%-5d copies=%d depth=%d seq_hash=%s -> %s"
          % (L, reps, a3m.count(">"), h16, OUT / ("cdk2_msa_%d.yaml" % L)))

# self-check: at one unit the a3m must be the source byte for byte
one = msa_dir / ("%s.a3m" % hashlib.sha256(query.encode()).hexdigest()[:16])
if UNIT in SIZES:
    ref = "\n".join("%s\n%s" % (h, r) for h, r in zip(heads, rows)) + "\n"
    print("L=%d reproduces prot300.a3m byte for byte: %s" % (UNIT, one.read_text() == ref))
