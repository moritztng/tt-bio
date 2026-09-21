#!/usr/bin/env python3
"""Why the RFdiffusion3 audit arm reports output_ok=False, checked by AST rather than argued.

`perf/allm_audit/rfd3_page.py` fails every design on `na != EXP_ATOMS` with EXP_ATOMS = 6051,
while its own validate() docstring says to test "residue topology and finiteness, never atom
equality between siblings". Both statements cannot hold. This decides which one the code supports.

Three claims, all about tt_bio/rfd3/design.py, none of which needs the model to run:

  A. `DesignResult.n_atoms` is set to `int(X.shape[1])` -- the FEATURISED atom axis, a padded
     width that is constant across designs of one spec. 6051 is that number, and the same
     expression is what the "(N atoms, batch=)" progress line prints.
  B. `_write_cif` does not write that many atoms. It builds a `keep` list and skips two kinds of
     atom: those `_is_virtual` (synthetic atom14 pad slots) and `CB` on a residue the sequence
     head predicted GLY.
  C. the GLY skip reads `gly_tok`, which is built from `pred_restype` -- the DESIGNED SEQUENCE.
     So the written atom count varies per design, by construction.

A+B mean 6051 and the CIF count are different quantities, so the equality can never pass. C means
the sibling-to-sibling variation the check also trips on is the model designing different
sequences. The harness's docstring is right and its constant is wrong.

exit 0  all three hold -- the check is the defect, and the arm's timing stands
exit 1  a claim does not hold -- the finding is wrong, re-read before acting on it
exit 2  the file or the symbols moved -- this test is stale, not the code
"""
import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "tt_bio" / "rfd3" / "design.py"


def main() -> int:
    if not SRC.is_file():
        print(f"STALE: {SRC} does not exist")
        return 2
    tree = ast.parse(SRC.read_text())

    write_cif = next((n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == "_write_cif"), None)
    if write_cif is None:
        print("STALE: no _write_cif in design.py")
        return 2

    bad = []

    # A. n_atoms is the featurised width X.shape[1], not a count of what was written.
    n_atoms = [k for k in ast.walk(tree)
               if isinstance(k, ast.keyword) and k.arg == "n_atoms"]
    if not n_atoms:
        print("STALE: no n_atoms= keyword anywhere in design.py")
        return 2
    srcs = [ast.unparse(k.value) for k in n_atoms]
    if not any("shape[1]" in s for s in srcs):
        bad.append(f"A: n_atoms is not a .shape[1] -- it is {srcs}")

    # B. _write_cif drops atoms: a `keep` list, and two `continue`s inside the loop that fills it.
    keeps = [n for n in ast.walk(write_cif)
             if isinstance(n, ast.Name) and n.id == "keep"]
    conts = [n for n in ast.walk(write_cif) if isinstance(n, ast.Continue)]
    if not keeps:
        bad.append("B: _write_cif has no `keep` list, so it may write every featurised atom")
    if len(conts) < 2:
        bad.append(f"B: _write_cif skips atoms at {len(conts)} sites, expected 2 "
                   f"(virtual pad, GLY CB)")
    body = ast.unparse(write_cif)
    if "_is_virtual" not in body:
        bad.append("B: _write_cif never consults _is_virtual, so the pad-atom skip is gone")
    if "AtomArray(len(keep))" not in body:
        bad.append("B: the AtomArray is not sized from len(keep) -- the written count may no "
                   "longer be the kept count")

    # C. the GLY skip is driven by the designed sequence, so the count varies per design.
    if "gly_tok" not in body:
        bad.append("C: no gly_tok -- the per-design GLY CB skip is gone")
    if "pred_restype" not in body:
        bad.append("C: _write_cif does not read pred_restype, so the count would NOT vary by "
                   "designed sequence and the sibling spread needs another explanation")

    for line in bad:
        print("FAIL " + line)
    if bad:
        return 1
    print("OK  A: DesignResult.n_atoms = int(X.shape[1]), the featurised atom axis (6051 at R4)")
    print("OK  B: _write_cif writes len(keep), after skipping virtual pads and GLY CB")
    print("OK  C: the GLY skip reads pred_restype, so the written count varies per designed "
          "sequence")
    print()
    print("=> EXP_ATOMS = 6051 in perf/allm_audit/rfd3_page.py compares a CIF's real atom count "
          "against a padded featurised width. It cannot pass for any design, and the sibling "
          "spread it also trips on is the model designing different sequences -- which the "
          "validator's own docstring says never to test. The rfd3 arm's seconds are unaffected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
