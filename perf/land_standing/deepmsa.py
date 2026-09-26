#!/usr/bin/env python3
"""A DEEP alignment for the CDK2 monomer, extracted from the deep tiled-512 one on this box.

The precondition fold said the tiled-CDK2 family reads pLDDT 0.602 at 298 aa with its committed
35-sequence alignment, against 0.503 single-sequence. So the MSA helps and 35 sequences is not
enough; CDK2 is a heavily covered kinase and a real search returns thousands.

There is no deep alignment of the 298 monomer on this box, but there is one of the tiled 512
sequence at depth 13232, and the tiled 512 is CDK2[0:298] + CDK2[0:214]. **Its first 298 match
columns ARE the monomer's deep alignment.** Cut there with the repo's own column rule, drop the
rows that become all-gap, and the result is a deep monomer a3m that tiles to any length by
exactly the rule `build_sweep_fixtures.py` already uses.

Depth is capped because it costs fold time, not because more would be wrong; the cap is printed
with the result so a reader knows what was folded.
"""
import hashlib
import pathlib
import sys

WT = pathlib.Path("/home/ttuser/.coworker/wt/land-standing")
sys.path.insert(0, str(WT / "scripts"))
from capacity_fixture import cut                                   # noqa: E402

SRC = pathlib.Path("/home/ttuser/fastab_std_4/msa/"
                   "4e5c2b391bde62d4_unpaired_tmp_env/bfd.mgnify30.metaeuk30.smag30.a3m")
OUT = pathlib.Path(sys.argv[1])
SIZES = [int(x) for x in sys.argv[2].split(",")]
MAX_DEPTH = int(sys.argv[3])

MONOMER = (WT / "scripts/gpu_vs_tt/fixtures/prot300.a3m").read_text().split("\n")[1]
UNIT = len(MONOMER)

def read_a3m(path):
    """Header/sequence pairs, joining wrapped sequence lines.

    The committed fixtures are strict one-line-per-row, this search output is not, and assuming
    the strict form here raised on the first read.
    """
    heads, rows, buf = [], [], []
    for line in path.read_text().split("\n"):
        if line.startswith(">"):
            if buf:
                rows.append("".join(buf))
                buf = []
            heads.append(line)
        elif line:
            buf.append(line)
    if buf:
        rows.append("".join(buf))
    assert len(heads) == len(rows), "a3m has %d headers and %d rows" % (len(heads), len(rows))
    return heads, rows


heads, rows = read_a3m(SRC)
print("source %s: depth=%d" % (SRC.name, len(rows)))

# The claim this whole extraction rests on, asserted rather than assumed.
assert cut(rows[0], UNIT) == MONOMER, "the deep alignment's first 298 columns are NOT the monomer"
print("VERIFIED: the deep query's first %d match columns are the CDK2 monomer, byte for byte" % UNIT)

mono_heads, mono_rows = [], []
for h, r in zip(heads, rows):
    try:
        c = cut(r, UNIT)
    except AssertionError:
        continue                       # a row that does not span 298 match columns is unusable
    if len(mono_rows) and set(c) <= {"-"}:
        continue                       # aligned only to the second half; carries nothing here
    mono_heads.append(h)
    mono_rows.append(c)
print("monomer alignment: %d rows span the first %d columns and are not all-gap"
      % (len(mono_rows), UNIT))

if len(mono_rows) > MAX_DEPTH:
    mono_heads, mono_rows = mono_heads[:MAX_DEPTH], mono_rows[:MAX_DEPTH]
print("capped to depth %d for fold time" % len(mono_rows))

msa_dir = OUT / "msa"
msa_dir.mkdir(parents=True, exist_ok=True)
for L in SIZES:
    reps = -(-L // UNIT)
    seq = (MONOMER * reps)[:L]
    a3m = "\n".join("%s\n%s" % (h, cut(r * reps, L)) for h, r in zip(mono_heads, mono_rows)) + "\n"
    assert a3m.split("\n")[1] == seq, "L=%d: a3m query row != target sequence" % L
    h16 = hashlib.sha256(seq.encode()).hexdigest()[:16]
    (msa_dir / ("%s.a3m" % h16)).write_text(a3m)
    (OUT / ("cdk2_deep_%d.yaml" % L)).write_text(
        "# Tiled CDK2 (PDB 1HCL) at %d residues, %d copies of the 298 aa monomer cut to length.\n"
        "# MSA: the monomer's DEEP alignment (extracted from the tiled-512 bfd search, depth %d)\n"
        "# tiled the same way, seeded at msa/%s.a3m.\n"
        "sequences:\n  - protein:\n      id: A\n      sequence: %s\n"
        % (L, reps, len(mono_rows), h16, seq))
    print("  L=%-5d copies=%d depth=%d seq_hash=%s" % (L, reps, a3m.count(">"), h16))
