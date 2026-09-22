#!/usr/bin/env python3
"""What the tape gate costs an inference softmax call, measured, with an A/A floor.

Moritz: "make sure regular inference is not changed to softmax fp64, not made slower." The
digest arm answers the first half. This answers the second at the resolution where the gate
actually lives -- the wrapper around `ttnn.softmax` -- and it runs on CPU, so it does not wait
for a card.

Two readings, and the first is the one that settles it.

OPCODES. The shipped inference path is `site_softmax(..., host_f64=False)`. Both the pre-gate
and the gated wrapper are traced at the opcode level with `f_trace_opcodes`, so what is compared
is the instruction sequence the interpreter actually executed, not the source. If the two
sequences are equal the gate cannot have cost anything, because no additional instruction ran.

TIME. The same two wrappers timed against a stub `ttnn.softmax`, arms interleaved, A/A first.
The A/A floor is two runs of the SAME wrapper: without it a cross-wrapper difference has no
scale and any number looks like a result. `ttnn.softmax` is stubbed because the kernel is
common to both arms and is ~54.5 us at [1,16,384,384] -- leaving it in would bury the thing
being measured under six orders of magnitude of shared work.

Neither reading replaces the fold-level A/B, which needs a card. Both bound it: a fold's cost
is this per-call number times the number of calls the fold makes, which
`inference_ab_with_aa_floor.py --stats` reads off the fold itself.
"""
import argparse
import json
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The tree BEFORE the gate, pinned by SHA. A branch name is the wrong handle here and this
# harness learned it the expensive way: `origin/wk/of3t` was the pre-gate tree when this file was
# written and stopped being it 20 minutes later, when the campaign branch took the gate commit in.
# The run that followed compared the gated wrapper against itself, found no opcode difference and
# no timing difference, and reported FREE -- a perfect A/A wearing an A/B's label. `_source_of`
# now refuses when the two arms are the same text, which is the control that would have caught it.
BASE_REF = "6d7f32dc0"               # wk/of3t immediately before 7f386f768 landed the gate


def _source_of(ref: str, name: str) -> str:
    """Lift one top-level function's source out of `tenstorrent.py` at `ref`."""
    import ast
    src = subprocess.run(["git", "show", f"{ref}:tt_bio/tenstorrent.py"],
                         cwd=str(ROOT), capture_output=True, text=True, check=True).stdout
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, fn)


class _Stats(dict):
    """The census dict, standing in for the module global both wrappers bump."""


def _build_arms():
    """The two wrappers, each compiled into its own namespace with a stub `ttnn`."""
    class _StubTtnn:
        @staticmethod
        def softmax(x, **kw):
            return x

    def _ns():
        stats = _Stats({"served": 0, "declined": 0, "refused": 0, "elements": 0})
        return {"ttnn": _StubTtnn, "HOST_F64_SOFTMAX_STATS": stats,
                "host_f64_softmax": lambda x, dim=-1: x, "int": int}

    arms, texts = {}, {}
    for label, ref in (("pre_gate", BASE_REF), ("gated", None)):
        ns = _ns()
        if ref is None:
            src = raw = _source_of("HEAD", "site_softmax")
            # `from . import ops` cannot run outside the package; the gate reads
            # `ops.host_softmax_hook()`, and an inference process has nothing installed, so the
            # stub returns None exactly as the real slot does.
            class _StubOps:
                @staticmethod
                def host_softmax_hook():
                    return None
            src = src.replace("from . import ops", "ops = _STUB_OPS")
            ns["_STUB_OPS"] = _StubOps
        else:
            src = raw = _source_of(ref, "site_softmax")
        exec(compile(textwrap.dedent(src), f"<{label}>", "exec"), ns)
        arms[label] = (ns["site_softmax"], ns["HOST_F64_SOFTMAX_STATS"])
        texts[label] = raw
    if texts["pre_gate"] == texts["gated"]:
        raise SystemExit(
            "REFUSING: site_softmax is the same text at %s and at HEAD, so the 'before' arm is "
            "not before anything and every reading below would be an A/A wearing an A/B's "
            "label. Repoint BASE_REF at a commit that predates the gate." % BASE_REF)
    return arms


def opcode_trace(fn):
    """The opcodes `fn` executes on one shipped inference call, in order."""
    seen = []
    code = fn.__code__

    def local(frame, event, arg):
        if event == "opcode":
            seen.append(frame.f_lasti)
        return local

    def tracer(frame, event, arg):
        if event == "call" and frame.f_code is code:
            frame.f_trace_opcodes = True
            return local
        return None

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        fn("SC", dim=-1, numeric_stable=True, compute_kernel_config="CKC", host_f64=False)
    finally:
        sys.settrace(old)

    import dis
    by_off = {i.offset: i.opname for i in dis.get_instructions(code)}
    return [by_off.get(o, "?") for o in seen]


def time_arm(fn, reps):
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        fn("SC", dim=-1, numeric_stable=True, compute_kernel_config="CKC", host_f64=False)
    return (time.perf_counter_ns() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200000)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    arms = _build_arms()
    pre, gated = arms["pre_gate"][0], arms["gated"][0]

    tr_pre, tr_gated = opcode_trace(pre), opcode_trace(gated)
    diffs = [{"i": i, "pre_gate": x, "gated": y}
             for i, (x, y) in enumerate(zip(tr_pre, tr_gated)) if x != y]
    # A conditional jump is a conditional jump whichever way it tests. `if not host_f64` became
    # `if host_f64` with the arms swapped, so the one instruction that differs is the same
    # instruction with its sense inverted -- not an instruction the gate added.
    JUMPS = {"POP_JUMP_IF_TRUE", "POP_JUMP_IF_FALSE"}
    polarity_only = bool(diffs) and all(
        d["pre_gate"] in JUMPS and d["gated"] in JUMPS for d in diffs)
    added = len(tr_gated) - len(tr_pre)
    no_extra_work = added <= 0 and (not diffs or polarity_only)

    # Interleaved, and each A/A arm is the SAME function object twice, so its two legs differ
    # by nothing but the machine. Every round times all four legs, and the comparison is made on
    # the PER-ROUND PAIRED DIFFERENCE rather than on a difference of two medians: the legs drift
    # together, and a floor taken as |median - median| reads 0.4 ns against a round-to-round
    # stdev of 8-21 ns, which would make a few ns of nothing look like a result.
    legs = {"AA_pre_1": pre, "AA_pre_2": pre, "AA_gated_1": gated, "AA_gated_2": gated}
    samples = {k: [] for k in legs}
    order = list(legs)
    for r in range(a.rounds):
        for k in (order if r % 2 == 0 else order[::-1]):
            samples[k].append(time_arm(legs[k], a.reps))

    def q95(xs):
        s = sorted(abs(x) for x in xs)
        return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]

    aa_pre = [x - y for x, y in zip(samples["AA_pre_1"], samples["AA_pre_2"])]
    aa_gated = [x - y for x, y in zip(samples["AA_gated_1"], samples["AA_gated_2"])]
    ab = [(g1 + g2) / 2 - (p1 + p2) / 2 for g1, g2, p1, p2 in
          zip(samples["AA_gated_1"], samples["AA_gated_2"],
              samples["AA_pre_1"], samples["AA_pre_2"])]

    floor = max(q95(aa_pre), q95(aa_gated))
    ab_med = statistics.median(ab)
    pre_ns = statistics.median(samples["AA_pre_1"] + samples["AA_pre_2"])
    gated_ns = statistics.median(samples["AA_gated_1"] + samples["AA_gated_2"])
    readable = abs(ab_med) > floor

    rep = {
        "opcodes": {
            "pre_gate": tr_pre, "gated": tr_gated,
            "n_pre": len(tr_pre), "n_gated": len(tr_gated),
            "instructions_added": added,
            "differences": diffs,
            "branch_polarity_only": polarity_only,
            "no_extra_work": no_extra_work,
            "reading": ("the shipped inference call executes %d instructions before the gate "
                        "and %d after; the only difference is the sense of one conditional "
                        "jump, so the gate added no work" % (len(tr_pre), len(tr_gated)))
                       if no_extra_work else
                       ("the gate changed what the inference call executes: %+d instructions, "
                        "%d differing positions" % (added, len(diffs))),
        },
        "time_ns_per_call": {
            "pre_gate_median": round(pre_ns, 2), "gated_median": round(gated_ns, 2),
            "AA_floor_ns": round(floor, 2),
            "AA_pre_median_diff": round(statistics.median(aa_pre), 2),
            "AA_gated_median_diff": round(statistics.median(aa_gated), 2),
            "AB_median_diff": round(ab_med, 2),
            "AB_min": round(min(ab), 2), "AB_max": round(max(ab), 2),
            "round_stdev_ns": {k: round(statistics.stdev(v), 2) for k, v in samples.items()},
            "readable_above_the_floor": readable,
            "slower": readable and ab_med > 0,
            "reps_per_leg": a.reps, "rounds": a.rounds,
            "note": ("A/A floor is the 95th percentile of |same-arm per-round difference|, "
                     "read before the A/B. A NEGATIVE AB_median_diff means the gated wrapper "
                     "was the faster one."),
        },
        "verdict": (
            "NOT FREE: the gate costs %.2f ns per shipped inference softmax call, above the "
            "%.2f ns A/A floor" % (ab_med, floor) if (readable and ab_med > 0) else
            "FREE: the gate adds no instruction to the shipped inference call (%d before, %d "
            "after, one conditional jump inverted), and the A/B is %.2f ns against a %.2f ns "
            "A/A floor, so it is not slower"
            % (len(tr_pre), len(tr_gated), ab_med, floor) if no_extra_work else
            "UNDECIDED: the instruction sequence changed; see the fields above"),
    }
    print(json.dumps(rep, indent=1), flush=True)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
        print("wrote", a.out)
    return 0 if (no_extra_work and not (readable and ab_med > 0)) else 3


if __name__ == "__main__":
    sys.exit(main())
