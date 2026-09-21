#!/usr/bin/env python3
"""Which of this row's reshapes are free, decided by ttnn's own condition rather than by analogy.

The atom half of this lever rests on two claims that were asserted for four passes and never
checked:

  1. `ttnn.reshape` CANNOT turn `[1, K, 32, 128]` into `[K, 4, 32, 32]` for free, even though the
     two are the same tile buffer in the same order. That is why the atom q destination has to be
     written by the matmul instead of reshaped after a `ttnn.linear` -- if the reshape were free the
     whole q half of this row would be unnecessary and one op-class change would come off the bet.
  2. The reshapes `atom_qkv_heads` DOES rely on, `[K, H, rows, 32] -> [1, K*H, rows, 32]`, ARE free.

Both are decided by one predicate in the installed wheel, `this_is_view` in
`reshape_view/reshape.cpp`, and its first clause is the whole answer: **the last dimension must be
unchanged**. 128 -> 32 fails it; 32 -> 32 passes.

The predicate is transcribed here and anchored against the wheel's source text, then validated
two-sidedly against the executed capture (`dm_prof` at `a63d6d8e3`), which is the part that makes
this evidence rather than a reading: a reshape the predicate calls a view emits NO device program
and must be ABSENT from the capture, and one it calls a real reshape must be PRESENT. The token
transformer's head merge is the positive control -- lead L2, 0.10483 s over 4800 programs -- and it
fails the predicate for a reason worth naming: its logical second-last dim is the head_dim of 48,
and 48 % 32 != 0.
"""

from __future__ import annotations

import csv
import collections
import gzip
import io
import json
import re
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CSV_REF = "a63d6d8e3:perf/c12_profiled_fold/runs/dm_prof/ops_perf_results.csv.gz"
TILE = 32

# The clauses `this_is_view` is built from, anchored so a wheel bump that changes the rule fails
# here instead of silently making a "free" reshape cost a kernel.
ANCHORS = [
    "bool this_is_view =",
    "(tensor_shape_last_dim == shape_last_dim)",
    "(tensor.layout() == ttnn::ROW_MAJOR_LAYOUT) ||",
    "(tensor_shape_second_last_dim == shape_second_last_dim) ||",
    "(shape_second_last_dim % tile_second_dim == 0 &&",
    "tensor_shape_second_last_dim % tile_first_dim == 0));",
]

# (name, in_logical, out_logical, shipped, why it matters)
#
# `shipped` is whether the fold actually ISSUES this reshape. Only a shipped case can be
# cross-checked against the capture: a view emits no program and must be absent, a real reshape
# must be present. A COUNTERFACTUAL is absent because nobody issues it, so its absence says nothing
# and the predicate's verdict is the whole result -- which is the point of asking at all.
CASES = [
    ("atom_q_metadata", (1, 140, 32, 128), (140, 4, 32, 32), False,
     "COUNTERFACTUAL: the same 560-tile buffer in the same order. NOT a view, which is why the "
     "matmul has to write the destination rather than a ttnn.linear plus a reshape -- if this were "
     "free the whole q half of this row would be unnecessary"),
    ("atom_q_return", (140, 4, 32, 32), (1, 560, 32, 32), True,
     "what atom_qkv_heads returns q as; must be free"),
    ("atom_kv_return", (140, 4, 128, 32), (1, 560, 128, 32), True,
     "what it returns k and v as; must be free"),
    ("atom_fallback_kv", (1, 140, 128, 128), (140, 1, 128, 128), True,
     "the shipped chain's own reshape, free -- which is why the atom cost is the pad, the split "
     "and the slice and not this"),
    ("l2_token_merge", (1, 16, 48, 512), (1, 768, 512), True,
     "POSITIVE CONTROL: lead L2's merge, measured 0.10483 s over 4800 programs. Fails on the "
     "logical head_dim of 48, since 48 % 32 != 0"),
    ("l2_token_merge_24", (1, 16, 24, 512), (1, 384, 512), True,
     "the same site at a 24-wide head, 4 programs in the capture"),
]


def _anchor_check():
    sys.path.insert(0, str(ROOT))
    from tt_bio.mm_generic import ttnn_cpp_root
    src = (ttnn_cpp_root() / "cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp")
    assert src.is_file(), f"wheel reshape source not found at {src}"
    text = src.read_text()
    missing = [a for a in ANCHORS if a not in text]
    return str(src), missing


def is_view(in_shape, out_shape, row_major=False):
    """`this_is_view`, transcribed. Both sides DRAM-interleaved here, so the two memory-config
    clauses hold and are not modelled."""
    last_in = in_shape[-1] if in_shape else 1
    last_out = out_shape[-1] if out_shape else 1
    sl_in = in_shape[-2] if len(in_shape) >= 2 else 1
    sl_out = out_shape[-2] if len(out_shape) >= 2 else 1
    if last_in != last_out:
        return False, "last dim changes"
    if row_major:
        return True, "row major"
    if sl_in == sl_out:
        return True, "second-last dim unchanged"
    if sl_out % TILE == 0 and sl_in % TILE == 0:
        return True, "both second-last dims are whole tiles"
    return False, f"second-last {sl_in} -> {sl_out}, and {sl_in} % {TILE} = {sl_in % TILE}"


def _capture_reshapes():
    """Every executed ReshapeView signature, keyed on LOGICAL in/out shapes."""
    blob = subprocess.run(["git", "show", CSV_REF], check=True, capture_output=True,
                          cwd=ROOT).stdout
    rows = list(csv.DictReader(gzip.open(io.BytesIO(blob), "rt")))
    seen = collections.Counter()
    for r in rows:
        if r["OP CODE"] != "ReshapeViewDeviceOperation":
            continue

        def logical(key):
            out = []
            for a in "WZYX":
                m = re.search(r"\[(\d+)\]", r[f"{key}_{a}_PAD[LOGICAL]"])
                out.append(int(m.group(1)) if m else 0)
            return tuple(out)
        seen[(logical("INPUT_0"), logical("OUTPUT_0"))] += 1
    return seen


def _matches(seen, in_shape, out_shape):
    """Programs in the capture whose logical shapes are these, ignoring leading 1s."""
    def norm(s):
        s = list(s)
        while len(s) > 1 and s[0] == 1:
            s.pop(0)
        return tuple(s)
    want = (norm(in_shape), norm(out_shape))
    return sum(n for (i, o), n in seen.items() if (norm(i), norm(o)) == want)


def _negctrl(seen):
    """The cross-check has to be able to fail, in both directions.

    Forcing every case to VIEW must break the two that emit programs; forcing every case to KERNEL
    must break the three that emit none. A cross-check that survives both is reading nothing.
    """
    shipped = [(n, i, o) for n, i, o, s, _w in CASES if s]
    forced_view = [n for n, i, o in shipped if _matches(seen, i, o) > 0]
    forced_kernel = [n for n, i, o in shipped if _matches(seen, i, o) == 0]
    print(f"NEGCTRL always-view breaks:   {forced_view}")
    print(f"NEGCTRL always-kernel breaks: {forced_kernel}")
    good = len(forced_view) == 2 and len(forced_kernel) == 3
    print("NEGCTRL PASS" if good else "NEGCTRL FAIL -- the capture cross-check is not two-sided")
    return 0 if good else 1


def main() -> int:
    src, missing = _anchor_check()
    print(f"SOURCE {src}")
    if missing:
        print("FAIL -- the wheel's this_is_view no longer contains:")
        for a in missing:
            print(f"  {a}")
        return 1
    print(f"anchors: all {len(ANCHORS)} clauses present\n")

    seen = _capture_reshapes()
    if "--negctrl" in sys.argv[1:]:
        return _negctrl(seen)
    rows, ok = [], True
    for name, i, o, shipped, why in CASES:
        view, reason = is_view(i, o)
        progs = _matches(seen, i, o)
        # A view emits no device program, so it must be ABSENT; a real reshape must be PRESENT.
        # Only meaningful for a reshape the fold issues.
        control = None if not shipped else ((progs == 0) if view else (progs > 0))
        if control is False:
            ok = False
        if not shipped and progs:
            ok = False          # a "counterfactual" that the fold does issue is a mislabel
        rows.append({"case": name, "in_logical": list(i), "out_logical": list(o),
                     "shipped": shipped, "is_view": view, "reason": reason,
                     "programs_in_capture": progs, "capture_agrees": control,
                     "why_it_matters": why})
        print(f"{name:<18} {'VIEW  ' if view else 'KERNEL'} progs={progs:<5} "
              f"{'shipped' if shipped else 'counterfactual':<14} "
              f"agrees={'n/a' if control is None else control}  ({reason})")

    (OUT / "reshape_view.json").write_text(json.dumps(
        {"all_pass": bool(ok), "source": src, "anchors": ANCHORS, "cases": rows}, indent=1) + "\n")
    print(f"\n{'PASS' if ok else 'FAIL'} -- {OUT / 'reshape_view.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
