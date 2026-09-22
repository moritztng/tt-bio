#!/usr/bin/env python3
"""Check the EXTRA entry this row proposes for `_of3t_donecheck.py`, without editing that file.

`of3t-modelframe` has no entry in the gate's EXTRA table, so the check refuses every possible
document and only the orchestrator can add one. Two things have to be true of the entry before
it is pasted, and neither is true by construction:

  - it must PASS on this row's honest report. A bar written blind can be unsatisfiable by a
    perfect artifact, and the first draft of this entry listed the `worst_case` and
    `denominator` checks without the GRADIENTS and MODELS fields those checks read, so it could
    not have passed any document this row would write.
  - it must REFUSE a document that drops the measurement. A gate that passes everything has
    tested nothing.

So this imports the real gate module, injects the entry at runtime, runs it against the real
state doc, then runs it again on eleven mutations of that doc that each remove one thing the
entry is supposed to require. The gate file on disk is never written, and the state doc is
restored after every mutation.

Run it where `~moritz/.coworker` lives, which is the orchestrator host, not qb1: qb1 has a
staged copy of the gate script but no state directory.

    python3 perf/of3t_modelframe/gate_entry_check.py

Exit 0 only if the unmutated document passes and every mutation is refused.
"""
import contextlib
import importlib.util
import io
import pathlib
import re
import sys

GATE = pathlib.Path("/home/moritz/.coworker/workstreams/_of3t_donecheck.py")
DOC = pathlib.Path("/home/moritz/.coworker/state/of3t-modelframe.md")

ENTRY = {"min": 2400, "req": [
    (r"^PREDICT:\s*\S", "PREDICT: the trunk allowance and the projection, fixed BEFORE the arm "
                        "ran, in a committed artifact with its commit and timestamp"),
    (r"^FRAME:\s*\S", "FRAME: the two float64 references' disagreement as a measured number, "
                      "both sides float64, with the trunk gradient mass on each boundary"),
    (r"^CONTROL:\s*\S", "CONTROL: the loss bit-identity and the trunk-gradient witness against "
                        "the published reference, both before any arm is read"),
    (r"^CLAUSE:\s*\S", "CLAUSE: the recomposed clause value with pairformer_stack from the "
                       "frame-matched arm and every other section untouched, against "
                       "0.15210099830945006"),
    (r"^MULTIPLE:\s*\S", "MULTIPLE: the frame-matched multiple of upstream's own bf16 on THIS "
                         "boundary, said explicitly whether or not it is near 2.2341x"),
    (r"^GRADIENTS:\s*\S", "GRADIENTS: the trunk's worst case LOCATED by parameter path, and the "
                          "share of the squared gradient norm the compared set holds"),
    (r"^MODELS:\s*\S", "MODELS: our leaves placed AGAINST their tensors as x of y, and the "
                       "compared parameters against the model's total"),
    (r"^HOST:\s*\S", "HOST: the host and card of the device arm, and the board class, because "
                     "the banked arm it must stay comparable with is p300c"),
    (r"^VERDICT:\s*\S", "VERDICT:")],
    "measured": ["FRAME", "CLAUSE", "MULTIPLE", "GRADIENTS"],
    "checks": ["f64", "worst_case", "denominator", "named_host", "clock", "bound"]}


def _sub_field(t, name, fn):
    m = re.search(rf"^{name}:(.*?)(?=^[A-Z][A-Z-]+:)", t, re.M | re.S)
    return t[:m.start(1)] + fn(m.group(1)) + t[m.end(1):]


MUTATIONS = {
    "CLAUSE says a measurement is owed and carries no number":
        lambda t: _sub_field(t, "CLAUSE", lambda b: " pending the arm.\n\n"),
    "MULTIPLE quietly dropped":
        lambda t: re.sub(r"^MULTIPLE:", "Multiple:", t, flags=re.M),
    "FRAME says a measurement is owed":
        lambda t: _sub_field(t, "FRAME", lambda b: " not yet measured.\n\n"),
    "GRADIENTS with every dotted parameter path removed":
        lambda t: _sub_field(t, "GRADIENTS",
                             lambda b: re.sub(r"\b[A-Za-z_]\w*(?:\.\w+){2,}\b", "that tensor", b)),
    "GRADIENTS with no worst case at all":
        lambda t: _sub_field(t, "GRADIENTS", lambda b: re.sub(r"worst", "typical", b, flags=re.I)),
    "MODELS with no denominator":
        lambda t: _sub_field(t, "MODELS", lambda b: " one model, all tensors placed.\n\n"),
    "HOST names a host but no card":
        lambda t: t.replace("HOST: qb2 (tt-quietbox2) card 2,", "HOST: qb2 (tt-quietbox2),"),
    "HOST is pc card 0":
        lambda t: t.replace("HOST: qb2 (tt-quietbox2) card 2,", "HOST: pc card 0,"),
    "no DURING-sampled AICLK recorded":
        lambda t: t.replace("AICLK", "clock").replace(" MHz", " megahertz"),
    "DOESNOT drops the stability bound":
        lambda t: _sub_field(t, "DOESNOT", lambda b: " nothing.\n\n"),
    "no float64 reference named anywhere":
        lambda t: re.sub(r"float64|fp64|finite difference", "high precision", t, flags=re.I),
}


def run(dc):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        sys.argv = ["_of3t_donecheck.py", "of3t-modelframe"]
        rc = dc.main()
    return rc, buf.getvalue()


def main() -> int:
    spec = importlib.util.spec_from_file_location("of3t_gate", GATE)
    dc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dc)
    if "of3t-modelframe" in dc.EXTRA:
        print("the entry is already in the gate; this check is for before it is pasted")
        return 0
    dc.EXTRA["of3t-modelframe"] = ENTRY
    dc._STAGE_HINTS.append(str(DOC))

    original = DOC.read_text()
    bad = []

    rc, out = run(dc)
    print(("PASS    " if rc == 0 else "REFUSED ") + "the state doc as written")
    if rc != 0:
        print(out.rstrip())
        bad.append("the entry refuses this row's own honest report")

    for name, fn in MUTATIONS.items():
        mutated = fn(original)
        if mutated == original:
            bad.append(f"mutation {name!r} changed nothing, so it tested nothing")
            print("INERT   " + name)
            continue
        DOC.write_text(mutated)
        try:
            rc, out = run(dc)
        finally:
            DOC.write_text(original)
        why = ""
        lines = [l.strip() for l in out.strip().splitlines()[1:] if l.strip()]
        if rc != 0 and lines:
            why = "  <- " + lines[0].lstrip("- ")[:96]
        print(("REFUSED " if rc != 0 else "PASSED  ") + name + why)
        if rc == 0:
            bad.append(f"the entry PASSES a document with {name}")

    assert DOC.read_text() == original, "the state doc was left mutated"
    if bad:
        print("\nthe proposed entry is not ready:")
        for b in bad:
            print("  - " + b)
        return 1
    print(f"\nthe proposed entry passes this row's report and refuses all "
          f"{len(MUTATIONS)} mutations of it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
