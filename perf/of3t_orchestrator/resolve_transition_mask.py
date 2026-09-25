#!/usr/bin/env python3
"""Compose `Transition.__call__`: main's `add_to_input` + host-refusal wrapper with the row's `mask`.

main (MGX, 2026-09-23) moved the body to `_transition(x, memory_config, add_to_input)` behind
`host_acc_after_refusal`; the of3t line renamed the same body `_swiglu_all` and masks its output
in `__call__`. The only `add_to_input` caller (protenix.py) passes no mask and every masked caller
(PairformerLayer) passes no `add_to_input`, so both are kept and the combination is refused:
masking `x + t(x)` would zero the residual on pads, which is not upstream's `_mask_trans`.
HEAD's `_transition` def survives; the row's `_swiglu_all` name does not.

Usage: resolve_transition_mask.py <file>  ->  0 resolved, 2 not this shape (caller must stop).
"""
import ast
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
for m in re.finditer(r"<<<<<<< [^\n]*\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n", s, re.S):
    a, b = m.group(1), m.group(2)
    if "add_to_input" in a and "def _transition(" in a and "_swiglu_all" in b and "mask" in b:
        head, row = a, b
        break
    if "add_to_input" in b and "def _transition(" in b and "_swiglu_all" in a and "mask" in a:
        head, row = b, a
        break
else:
    sys.exit(2)
doc = re.search(r'(\s*""".*?""")', row, re.S)
tail = re.search(r"\n(\s*def _transition\(.*)", head, re.S)
if not doc or not tail:
    sys.exit(2)
ind = " " * 8
new = ("                 add_to_input: bool = False,\n"
       "                 mask: ttnn.Tensor | None = None) -> ttnn.Tensor:"
       + doc.group(1).rstrip() + "\n"
       + ind + "if mask is not None and add_to_input:\n"
       + ind + "    raise ValueError(\"Transition: mask with add_to_input would mask the residual \"\n"
       + ind + "                     \"on padded positions; upstream masks t(x) only\")\n"
       + ind + "out = host_acc_after_refusal((\"transition\", tuple(x.padded_shape)), x,\n"
       + ind + "                             lambda: self._transition(x, memory_config, add_to_input))\n"
       + ind + "if mask is None:\n"
       + ind + "    return out\n"
       + ind + "masked = ttnn.multiply(out, mask)\n"
       + ind + "if not ops.taping():\n"
       + ind + "    # Under a tape the multiply's backward reads its operands (freeing `out` was a\n"
       + ind + "    # storage.cpp:60 TT_THROW); inference keeps the free, 48 MB at 384 aa.\n"
       + ind + "    ttnn.deallocate(out)\n"
       + ind + "return masked\n"
       + tail.group(1))
if not new.endswith("\n"):
    new += "\n"
out = s[:m.start()] + new + s[m.end():]
if "_swiglu_all" in out:
    sys.exit(2)
if "<<<<<<< " not in out:
    try:
        ast.parse(out)
    except SyntaxError:
        sys.exit(2)
p.write_text(out)
