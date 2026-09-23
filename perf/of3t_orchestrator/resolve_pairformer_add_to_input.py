#!/usr/bin/env python3
"""Compose `PairformerLayer.__call__`: main's `add_to_input` residuals with the row's transition masks.

main (MGX, 2026-09-23) makes every pair-track op return `z + update` through `add_to_input=True`, so
a pair too big to sit beside its update still runs. The of3t line (D174) keeps the explicit
`z_update` / `add_` form and masks the transition's update with `trans_mask_z`. The composed
`Transition` refuses `mask` together with `add_to_input` (resolve_transition_mask.py), because
masking `x + t(x)` would zero the residual on pads. So:

- the four triangle ops take main's form; they carry no transition mask and the row changed
  nothing in them but the residual idiom. The row's docstring is kept.
- `transition_z` takes main's form when unmasked, which is every inference caller, so inference
  runs main's bytes; when masked it takes the row's update-then-add.

Usage: resolve_pairformer_add_to_input.py <file>  ->  0 resolved, 2 not this shape.
"""
import ast
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
HUNK = re.compile(r"<<<<<<< [^\n]*\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n", re.S)


def sides(m):
    a, b = m.group(1), m.group(2)
    return (a, b) if "add_to_input=True" in a else (b, a)


done = 0
for m in HUNK.finditer(s):
    head, row = sides(m)
    if ("self.triangle_multiplication_start(z, mask, add_to_input=True)" in head
            and "z_update = self.triangle_multiplication_start(z, mask)" in row
            and "trans_mask_z" in row):
        doc = re.search(r'(\s*""".*?""")\n', row, re.S)
        if not doc:
            sys.exit(2)
        s = s[:m.start()] + doc.group(1).lstrip("\n") + "\n" + head + s[m.end():]
        done += 1
        break

ctx = ("        z = self.transition_z(\n"
       "            z, memory_config=_residual_update_memory_config(z.shape, z.dtype)\n")
for m in HUNK.finditer(s):
    head, row = sides(m)
    if (head.strip() == "if _RESIDUAL_L1 else None, add_to_input=True)"
            and "mask=trans_mask_z)" in row and "ttnn.add_(z, z_update)" in row
            and s[:m.start()].endswith(ctx)):
        new = ("        rmc = (_residual_update_memory_config(z.shape, z.dtype)\n"
               "               if _RESIDUAL_L1 else None)\n"
               "        if trans_mask_z is None:\n"
               "            z = self.transition_z(z, memory_config=rmc, add_to_input=True)\n"
               "        else:\n"
               "            # Upstream masks t(z) only, so the update is masked before the residual.\n"
               "            z_update = self.transition_z(z, memory_config=rmc, mask=trans_mask_z)\n"
               "            z = ttnn.add_(z, z_update)\n"
               "            ttnn.deallocate(z_update)\n")
        s = s[:m.start() - len(ctx)] + new + s[m.end():]
        done += 1
        break

if done != 2:
    sys.exit(2)
if "<<<<<<< " not in s:
    try:
        ast.parse(s)
    except SyntaxError:
        sys.exit(2)
p.write_text(s)
