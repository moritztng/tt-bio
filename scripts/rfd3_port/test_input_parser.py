"""Unit tests for tt_bio/rfd3_input.py — the contig + InputSelection parser.

Every assertion is grounded in a worked example from
RosettaCommons/foundry models/rfd3/docs/input.md (production, 2026-07-23), cited
inline. Run:

  python3 scripts/rfd3_port/test_input_parser.py
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3_input import (
    parse_contig, parse_input_selection, InputSpecification,
    ChainBreak, Indexed, Designed, DesignedRange, UnindexedOffset, AtomSelection,
    contig_summary,
)


def check(label, got, want):
    ok = got == want
    print(f"[{'OK ' if ok else 'FAIL'}] {label}: {got}")
    if not ok:
        print(f"      expected: {want}")
        raise SystemExit(1)


# --- Contig Strings section of input.md -----------------------------------
# "A40-60,70,A120-170,A203,/0,B3-45,60-80"
comps = parse_contig("A40-60,70,A120-170,A203,/0,B3-45,60-80")
check("A40-60 -> Indexed(A40-60)", comps[0], Indexed("A", 40, 60))
check("70 -> Designed(70)", comps[1], Designed(70))
check("A120-170 -> Indexed(A120-170)", comps[2], Indexed("A", 120, 170))
check("A203 -> Indexed(A203)", comps[3], Indexed("A", 203, 203))
check("/0 -> ChainBreak", comps[4], ChainBreak())
check("B3-45 -> Indexed(B3-45)", comps[5], Indexed("B", 3, 45))
check("60-80 -> DesignedRange(60-80)", comps[6], DesignedRange(60, 80))
check("contig has 7 components", len(comps), 7)
summ = contig_summary(comps)
check("summary n_indexed=4", summ["n_indexed"], 4)
check("summary n_chain_breaks=1", summ["n_chain_breaks"], 1)
check("summary n_designed_fixed=1", summ["n_designed_fixed"], 1)
check("summary n_designed_range=1", summ["n_designed_range"], 1)

# Quick start example: "50-80,/0,A1-100"
comps = parse_contig("50-80,/0,A1-100")
check("50-80 -> DesignedRange", comps[0], DesignedRange(50, 80))
check("/0 -> ChainBreak", comps[1], ChainBreak())
check("A1-100 -> Indexed", comps[2], Indexed("A", 1, 100))

# --- Unindexing Specifics: offset syntax (unindex field) -------------------
# "A11-12" ties two unindexed components together (0 offset).
check("A11-12 -> Indexed(A11-12)", parse_contig("A11-12", unindex=True), [Indexed("A", 11, 12)])
# "A11,0,A12" == "A11-12" (0 sequence offset)
check("A11,0,A12 -> [Indexed(A11), Offset(0), Indexed(A12)]",
      parse_contig("A11,0,A12", unindex=True),
      [Indexed("A", 11, 11), UnindexedOffset(0), Indexed("A", 12, 12)])
# "A11,3,A12" = 3-residue separation
check("A11,3,A12 -> [Indexed(A11), Offset(3), Indexed(A12)]",
      parse_contig("A11,3,A12", unindex=True),
      [Indexed("A", 11, 11), UnindexedOffset(3), Indexed("A", 12, 12)])
# "A244,A274,A320,A329,A375" — multiple unindexed components, NO offsets (all
# chain-prefixed single residues; input.md: "internal breakpoints are inferred
# and logged"). Confirm they all parse as Indexed.
check("A244,A274,A320 -> three Indexed",
      parse_contig("A244,A274,A320", unindex=True),
      [Indexed("A", 244, 244), Indexed("A", 274, 274), Indexed("A", 320, 320)])

# A bare integer between two designed/indexed-is-not-offset when neighbors are
# not both indexed: e.g. "10,/0,B5-12" — the 10 is a Designed length, NOT an
# offset (its right neighbor is a chain break).
check("10,/0,B5-12 -> [Designed(10), ChainBreak, Indexed(B5-12)]",
      parse_contig("10,/0,B5-12"),
      [Designed(10), ChainBreak(), Indexed("B", 5, 12)])

# --- InputSelection mini-language -----------------------------------------
check("bool True -> True", parse_input_selection(True), True)
check("bool False -> False", parse_input_selection(False), False)
check("contig string -> parsed list",
      parse_input_selection("A1-10,B5-8"),
      [Indexed("A", 1, 10), Indexed("B", 5, 8)])

# Dict examples from input.md (The InputSelection Mini-Language):
#   A1-2: BKBN   ; A3: N,CA,C,O,CB ; B5-7: ALL ; B10: TIP ; LIG: ''
d = parse_input_selection({
    "A1-2": "BKBN",
    "A3": "N,CA,C,O,CB",
    "B5-7": "ALL",
    "B10": "TIP",
    "LIG": "",
})
check("dict A1-2 -> BKBN", d["A1-2"], AtomSelection.bkbn())
check("dict A3 -> explicit {N,CA,C,O,CB}",
      d["A3"], AtomSelection.explicit(["N", "CA", "C", "O", "CB"]))
check("dict B5-7 -> ALL", d["B5-7"], AtomSelection.all_())
check("dict B10 -> TIP", d["B10"], AtomSelection.tip())
check("dict LIG -> none (empty string)", d["LIG"], AtomSelection.none_())

# --- InputSpecification round-trip + validation ---------------------------
spec = InputSpecification.from_dict({
    "input": "path/to/pdb",
    "contig": "A1-80,10,/0,B5-12",
    "select_unfixed_sequence": "A20-35",
    "ligand": "HAX,OAA",
    "partial_t": 15.0,
    "allow_ligand_on_existing_chain": True,   # passthrough -> extra_fields
})
spec.validate()
check("spec.contig parsed", spec.parsed_contig(),
      [Indexed("A", 1, 80), Designed(10), ChainBreak(), Indexed("B", 5, 12)])
check("spec.partial_t kept", spec.partial_t, 15.0)
check("unknown field -> extra_fields",
      spec.extra_fields.get("allow_ligand_on_existing_chain"), True)

# Safer parsing: malformed contig raises early.
try:
    InputSpecification.from_dict({"contig": "A1-80,bogus-"}).validate()
    raise SystemExit("FAIL: malformed contig did not raise")
except ValueError as e:
    print(f"[OK ] malformed contig raised: {e}")

# Bad dialect raises.
try:
    InputSpecification.from_dict({"dialect": 3}).validate()
    raise SystemExit("FAIL: bad dialect did not raise")
except ValueError as e:
    print(f"[OK ] bad dialect raised: {e}")

print("\nALL INPUT-PARSER TESTS PASSED")
