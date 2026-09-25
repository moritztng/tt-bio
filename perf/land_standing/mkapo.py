#!/usr/bin/env python3
"""Apo inputs at an exact token count, for reading which rung a real fold picks.

OpenFold3 pads the pair axis to a multiple of 64 (measured: 684 aa -> a (704, 704) pick), so the
only padded lengths a fold can present are multiples of 64. `N` here is chosen equal to the target
because a multiple of 64 pads to itself. The sequence is the committed aa684 apo fixture's own
tiled CDK2, repeated and cut -- the residues are not the subject, the token count is.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "perf/sizegate/inputs/apo/aa684/cdk2apo_684.yaml"
base = re.search(r"sequence:\s*(\S+)", SRC.read_text()).group(1)

out_root = pathlib.Path(sys.argv[1])
for n in (int(x) for x in sys.argv[2].split(",")):
    seq = (base * (n // len(base) + 1))[:n]
    assert len(seq) == n
    d = out_root / f"aa{n}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"cdk2apo_{n}.yaml").write_text(
        f"# Generated from {SRC.relative_to(ROOT)} for a {n}-token apo fold.\n"
        f"sequences:\n  - protein:\n      id: A\n      sequence: {seq}\n")
    print(d / f"cdk2apo_{n}.yaml", n)
