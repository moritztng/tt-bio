#!/usr/bin/env python3
"""Is `fp32_softmax=True` the blocker for M18, or is it the branch M18 lives in?

The gates brief says two things that cannot both be acted on. Its synthesis paragraph asks whether
the kwarg is "load-bearing, satisfiable or incidental" and calls deciding that the best leverage in
the campaign; a later section says, in bold, **KEEP `fp32_softmax=True`** and **do not flip it**.
A row reading top-down meets the first one first. `train-i-run` ran an overruled schedule for a day
on exactly this kind of disagreement, so it gets resolved from the source rather than from whichever
paragraph is nearer the top.

Six claims, all AST or structural, no `tt_bio` import and no device. Each prints its own evidence.
"""
import ast
import re
import sys
from pathlib import Path

TT = Path(__file__).resolve().parents[2] / "tt_bio"
SRC = TT / "tenstorrent.py"
SDPA = TT / "triatt_sdpa.py"


def fn_src(tree, src, name):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return ast.get_source_segment(src, n)
    return None


def main() -> int:
    src = SRC.read_text()
    tree = ast.parse(src)
    ok = []

    # 1 + 2. M18's route is called INSIDE the fp32_softmax branch, with the materialised path as
    # its fallback. If that holds, the kwarg is not M18's blocker -- it is M18's home.
    att = fn_src(tree, src, "_attend_heads")
    branch = None
    if att:
        m = re.search(r"if _FP32_SOFTMAX or self\.fp32_softmax:(.*?)(?=\n            (?:el)?if |\Z)",
                      att, re.S)
        branch = m.group(1) if m else None
    c1 = bool(branch and "_tri_att_sdpa_hifi(" in branch)
    c2 = bool(branch and re.search(r"if o is None:\s*\n\s*o = _fp32_softmax_attention\(", branch))
    ok += [("1. `_tri_att_sdpa_hifi` is called INSIDE the `fp32_softmax` branch", c1),
           ("2. ...and `_fp32_softmax_attention` is its FALLBACK when it declines", c2)]

    # 3 + 4. The other two levers really are refused by the same kwarg, on the other route.
    ep = fn_src(tree, src, "_tri_att_gated_sdpa")
    fuse = fn_src(tree, src, "_tri_att_fused_qkv_sdpa")
    c3 = bool(ep and re.search(r"if att\.biased or .*?att\.fp32_softmax:", ep))
    c4 = bool(fuse and re.search(r"if att\.biased or .*?att\.fp32_softmax:", fuse))
    ok += [("3. the gate epilogue IS refused by `att.fp32_softmax`", c3),
           ("4. the qkv+SDPA fusion IS refused by `att.fp32_softmax`", c4)]

    # 5. ...and if the clause were relaxed, those two would run at the OP DEFAULT config unless
    #    `sdpa_hifi` is also set -- the config the tree's own corrected data calls WORSE than the
    #    materialised route. So relaxing the kwarg is not a free unlock; it is a precision trade.
    c5 = bool(fuse and re.search(r"_TRIATT_FUSED_HIFI_CKC if att\.sdpa_hifi else None", fuse))
    ok += [("5. the fusion takes HiFi4 only when `sdpa_hifi` is set, else the op default", c5)]

    # 6. The op default is on the record as WORSE than the route those sites run today.
    # ...recorded in triatt_sdpa.py, which is where the corrected error-structure table lives.
    c6 = bool(re.search(r"op default \(HiFi2, approx, -\)\s+1\.08 - 1\.50x WORSE", SDPA.read_text()))
    ok += [("6. the tree records the op default as 1.08-1.50x WORSE than materialised", c6)]

    for label, good in ok:
        print(f"  [{'OK ' if good else 'FAIL'}] {label}")
    if not all(g for _, g in ok):
        print("\nA premise moved. Re-read the call site before acting on the synthesis.")
        return 2

    print("""
CONCLUSION
  M18 does NOT require touching `fp32_softmax`. The fused HiFi route is tried FIRST inside that
  very branch and falls back to `_fp32_softmax_attention` when it declines, so plumbing the four
  OpenFold3 sites is ADDITIVE and the kwarg stays exactly as it is. The brief's "KEEP
  fp32_softmax=True" is right and its synthesis paragraph is the one to correct.

  The other two levers are genuinely refused by the kwarg -- but relaxing it would route those
  sites to the fused kernel at its OP DEFAULT config, which the tree's own corrected error-structure
  data records as 1.08-1.50x WORSE than the materialised path they take today. So "one kwarg, three
  levers" is not one unlock: it is one additive plumb that needs no kwarg change, and two levers
  whose unlock is an ACCURACY TRADE that must be scored against the 0.60 A bar with the 1.84 A seed
  floor -- `allm-safety`'s question, not a gate fix.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
