#!/usr/bin/env python3
"""Compose `PairformerLayer.__call__` once more: main's `add_to_input` form against of3t-msafwd's fp32 pair residual.

Runs after resolve_pairformer_add_to_input.py has composed main with the transition masks. msafwd
(D250) routes every pair op through `_z_compute` / `_z_residual` so a taped step can carry `z` in
fp32; untaped, `wide` is False and `_z_residual` is the same in-place bf16 add. Its block cannot
use `add_to_input`: the op would add into the bf16 compute copy, not the fp32 residual.

So the layer takes main's block when it is plain (not `wide`, no transition mask), which is every
inference call since `TT_BIO_MASK_TRANS` defaults off and `wide` needs a tape; otherwise msafwd's.

Usage: resolve_pairformer_z_residual.py <file>  ->  0 resolved, 2 not this shape.
"""
import ast
import pathlib
import re
import sys
import textwrap

p = pathlib.Path(sys.argv[1])
s = p.read_text()
for m in re.finditer(r"<<<<<<< [^\n]*\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n", s, re.S):
    a, b = m.group(1), m.group(2)
    head, row = (a, b) if "add_to_input=True" in a else (b, a)
    if ("z = self.triangle_multiplication_start(z, mask, add_to_input=True)" in head
            and "if trans_mask_z is None:" in head
            and row.lstrip().startswith("wide = self.z_fp32_residual and ops.taping()")
            and "_z_residual(z, zc, z_update, wide)" in row):
        break
else:
    sys.exit(2)

# main's plain block, without its own mask branch: the else below owns every masked call.
plain = head.split("        rmc = (")[0] + (
    "            z = self.transition_z(\n"
    "                z, memory_config=_residual_update_memory_config(z.shape, z.dtype)\n"
    "                if _RESIDUAL_L1 else None, add_to_input=True)\n")
wide_line, rest = row.lstrip().split("\n", 1)
new = ("        " + wide_line + "\n"
       "        if not wide and trans_mask_z is None:\n"
       + textwrap.indent(textwrap.dedent(plain.split("            z = self.transition_z(")[0]), " " * 12)
       + plain[plain.index("            z = self.transition_z("):]
       + "        else:\n"
       + textwrap.indent(textwrap.dedent(rest.lstrip("\n")), " " * 12))
out = s[:m.start()] + new + s[m.end():]
if "<<<<<<< " not in out:
    try:
        ast.parse(out)
    except SyntaxError:
        sys.exit(2)
p.write_text(out)
