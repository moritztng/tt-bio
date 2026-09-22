#!/usr/bin/env python3
"""D56 runtime half: the module that defines TT_BIO_SOFTMAX_BW_RENORM is never LOADED by a fold.

`renorm_reach.py` proves the static half by AST -- every branch on the flag is inside a backward
closure. The runtime half was supposed to be "the counter reads zero over a real fold", and the
three-leg fold A/B came back with an EMPTY `TT_BIO_RENORM_STATS_DIR` rather than zero counts. An
empty directory is not a reading, so this asks the question the counter was standing in for, one
level up: does the fold path import `tt_bio.autograd` at all?

It does not. `tt_bio.main`, `tt_bio.tenstorrent`, `tt_bio.openfold3_trunk` and `tt_bio.ops` all
import clean of it, so the flag is never parsed, `softmax_bw_inner` is never defined in the
process, and the atexit dump is never registered. That is why the directory is empty, and it is a
stronger fact than a zero counter: a counter at zero could mean the code ran a branch that does
not increment, and this cannot.

Negative control, so a pass cannot come from the module being broken: after asserting absence, it
imports `tt_bio.autograd` directly and asserts the flag, the counter and the helper are all there.
A tree where autograd fails to import would fail that half loudly instead of passing this one
quietly.

Run with no device. Exit 0 on PASS.
"""
import json
import sys

FOLD_ENTRY_POINTS = ("tt_bio.main", "tt_bio.tenstorrent", "tt_bio.openfold3_trunk", "tt_bio.ops")
FLAG_MODULES = ("tt_bio.autograd", "tt_bio.taped_ttnn")


def main():
    failures = []
    for name in FOLD_ENTRY_POINTS:
        __import__(name)
    loaded = {m: (m in sys.modules) for m in FLAG_MODULES}
    for m, was in loaded.items():
        if was:
            failures.append(f"{m} is imported by the fold path ({', '.join(FOLD_ENTRY_POINTS)})")

    # Negative control: the module exists, imports, and carries what the static proof scored.
    import tt_bio.autograd as ag
    control = {
        "flag_present": hasattr(ag, "SOFTMAX_BW_RENORM"),
        "counter_present": hasattr(ag, "SOFTMAX_BW_RENORM_STATS"),
        "helper_present": hasattr(ag, "softmax_bw_inner"),
        "helper_exported": "softmax_bw_inner" in getattr(ag, "__all__", []),
        "counter_at_import": dict(ag.SOFTMAX_BW_RENORM_STATS),
    }
    for k in ("flag_present", "counter_present", "helper_present", "helper_exported"):
        if not control[k]:
            failures.append(f"negative control failed: {k}")
    if control["counter_at_import"] != {"applied": 0, "declined": 0}:
        failures.append("the counter is non-zero at import, so something ran a backward")

    out = {"fold_entry_points": list(FOLD_ENTRY_POINTS),
           "flag_module_loaded_by_fold_path": loaded,
           "control": control,
           "failures": failures,
           "verdict": "PASS" if not failures else "FAIL"}
    print(json.dumps(out, indent=1))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
