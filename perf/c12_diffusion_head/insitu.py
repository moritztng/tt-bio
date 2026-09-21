#!/usr/bin/env python3
"""What the head-major arms actually delete, and what they move, from the profiled fold's own CSV.

`perf/c12_tail_screen/leads.json` priced this lever as 0.31376 s of `NlpCreateHeads` and
`NLPConcatHeads`. That is the split and the merge themselves. It does NOT include the two ops that
exist ONLY to feed the atom split -- the q pad from the 32-row window up to ATOM_DIM and the slice
straight back down after it -- because the screen enumerated by op class and those land in Pad and
Slice. They go with the split, so they belong to the arm.

It also does not state what the arm MOVES. Both sites run `ttnn.linear` today and serving them puts
the projection on `minimal_matmul`, so the arm's net is a deletion minus whatever that op-class
change costs on matmuls of a known size. Naming that size is the difference between an A/B with a
prediction and an A/B with a hope.

Source: `perf/c12_profiled_fold/runs/dm_prof/ops_perf_results.csv.gz` at
`origin/wk/c12-profiled-fold @ a63d6d8e3`, the DiffusionModule unit fenced under the profiler at a
during-sampled 1350 MHz, read through `git show`. Eight replay reps, and `--controls` proves it two
ways rather than assuming it.

CONTROL, and it matters: a unit CSV is scaled to the fold by the unit's own fold call count, which
is only valid for an op that runs EVERY call. Every row this script uses has an integer per-call
count. The rows it refuses are the fractional ones -- the same-shape create at 0.5/call is the
once-per-fold hoisted conditioning bias, and multiplying that by 200 would inflate it ~400x.
"""

from __future__ import annotations

import collections
import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
CSV_REF = "a63d6d8e3:perf/c12_profiled_fold/runs/dm_prof/ops_perf_results.csv.gz"
PFL_REF = "a63d6d8e3:perf/c12_profiled_fold/runs/pfl_prof/ops_perf_results.csv.gz"
CLK_MHZ = 1350.0
DM_CALLS, DM_REPS = 200, 8            # DiffusionModule calls per fold; replay reps in the CSV
PFL_CALLS, PFL_REPS = 264, 9          # PairformerLayer, calibrated the same way

# What each arm deletes. (op, in0, out, per_call, arm, note)
DELETES = [
    ("NlpCreateHeadsDeviceOperation", (1, 1, 512, 3072), (1, 16, 512, 64), 24, "token",
     "the token transformer's split, 24 layers"),
    ("NlpCreateHeadsDeviceOperation", (140, 1, 128, 128), (140, 4, 128, 32), 6, "atom",
     "the atom block's split"),
    ("PadDeviceOperation", (1, 140, 32, 128), (1, 140, 128, 128), 6, "atom",
     "q padded 32 -> ATOM_DIM rows, only so the split sees matching row counts"),
    ("SliceDeviceOperation", (1, 560, 128, 32), (1, 560, 32, 32), 6, "atom",
     "and sliced straight back to 32 after it"),
    ("NLPConcatHeadsDeviceOperation", (140, 4, 32, 32), (140, 1, 32, 128), 6, "atom_tail",
     "the atom merge -- NOT built, see MOVES"),
]

# What the arms move off ttnn.linear, priced as the signature's mean per-program cost times the
# programs the arm actually converts. (in0, in1, programs_per_fold, arm, note)
#
# Priced this way rather than by a share of the signature because the [4480,128]x[128,128] signature
# is 222 programs over 8 reps = 27.75 per call, and a fractional count is exactly the trap this
# script's control refuses: 27 of them run every call and 6 run ONCE per fold, because
# `DiffusionTransformerLayer` caches the atom-level `output_projection_linear` in `self.s_o`
# (tenstorrent.py:9668-9678, `s` is t-independent there). Dividing by the mean is sound here because
# every program in the signature has the same shape AND the same core count; it would not be sound
# across a phase total.
MOVES = [
    ((1, 1, 512, 768), (1, 1, 768, 3072), 24 * 200, "token", "the token qkv projection"),
    ((1, 140, 128, 128), (1, 1, 128, 256), 6 * 200, "atom", "the atom kv projection"),
    ((1, 140, 32, 128), (1, 1, 128, 128), 6 * 200, "atom",
     "the atom q projection, 6 of the 27 per-call programs in this signature"),
    ((1, 140, 32, 128), (1, 1, 128, 128), 6 * 200, "atom_tail",
     "the atom gate projection, the only per-call one the tail would also convert -- its `out` "
     "sibling is the cached s_o, 6 programs per FOLD, so the tail barely touches it"),
]


def _rows(ref):
    blob = subprocess.run(["git", "show", ref], check=True, capture_output=True,
                          cwd=OUT.parents[1]).stdout
    return list(csv.DictReader(gzip.open(__import__("io").BytesIO(blob), "rt")))


def _shape(r, key):
    """The padded extent, or None where the op has no such operand (a single-input op's in1)."""
    vals = []
    for a in "WZYX":
        raw = r.get(f"{key}_{a}_PAD[LOGICAL]", "").split("[")[0].strip()
        if not raw:
            return None
        vals.append(int(raw))
    return tuple(vals)


def _index(rows):
    """(op, in0, in1, out) -> [total_ns, count, cores]."""
    ix = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        k = (r["OP CODE"], _shape(r, "INPUT_0"), _shape(r, "INPUT_1"), _shape(r, "OUTPUT_0"))
        e = ix[k]
        e[0] += int(r["DEVICE KERNEL DURATION [ns]"])
        e[1] += 1
        e[2] = int(r["CORE COUNT"])
    return ix


def _find(ix, op, in0=None, out=None, in1=None):
    hits = [(k, v) for k, v in ix.items()
            if k[0] == op and (in0 is None or k[1] == in0)
            and (in1 is None or k[2] == in1) and (out is None or k[3] == out)]
    assert len(hits) == 1, f"{op} {in0}->{out} matched {len(hits)} signatures"
    return hits[0][1]


def controls(ix, pfl_ix):
    """Two independent checks on the 8-rep, x200 scaling before any number is used."""
    out = []
    # 1. the atom split reproduces the tail screen's own independent pricing
    ns, n, _c = _find(ix, "NlpCreateHeadsDeviceOperation", (140, 1, 128, 128), (140, 4, 128, 32))
    got = ns / 1e9 / DM_REPS * DM_CALLS
    out.append({"control": "atom split vs leads.json", "measured_s": round(got, 5),
                "screen_s": 0.09016, "rel_err_pct": round((got / 0.09016 - 1) * 100, 2)})
    # 2. the trunk signature, scaled off a DIFFERENT unit with its own rep count
    ns, n, _c = _find(pfl_ix, "NlpCreateHeadsDeviceOperation", (1, 1, 512, 1536), (1, 16, 512, 32))
    got = ns / 1e9 / PFL_REPS * PFL_CALLS
    out.append({"control": "trunk split vs leads.json", "measured_s": round(got, 5),
                "screen_s": 0.00581, "rel_err_pct": round((got / 0.00581 - 1) * 100, 2),
                "per_call": n / PFL_REPS})
    return out


def main() -> int:
    ix, pfl_ix = _index(_rows(CSV_REF)), _index(_rows(PFL_REF))
    ctl = controls(ix, pfl_ix)
    print("CONTROLS")
    for c in ctl:
        print(f"  {c['control']:<28} {c['measured_s']:.5f} s vs {c['screen_s']:.5f} s  "
              f"{c['rel_err_pct']:+.2f} %")
    bad = [c for c in ctl if abs(c["rel_err_pct"]) > 1.0]
    if bad:
        print("FAIL -- a control is off by more than 1 %")
        return 1

    rec = {"source": CSV_REF, "clock_MHz": CLK_MHZ, "controls": ctl,
           "deletes": [], "moves": [], "totals": {}}
    print("\nDELETES (the ops that stop running)")
    for op, in0, o, per_call, arm, note in DELETES:
        ns, n, cores = _find(ix, op, in0, o)
        assert abs(n / DM_REPS - per_call) < 1e-9, \
            f"{op} {in0}: {n / DM_REPS} per call, expected {per_call} -- fractional counts refused"
        s = ns / 1e9 / DM_REPS * DM_CALLS
        rec["deletes"].append({"op": op, "in0": list(in0), "out": list(o), "arm": arm,
                               "per_call": per_call, "programs_per_fold": per_call * DM_CALLS,
                               "s_per_fold": round(s, 5), "Mcycles": round(s * CLK_MHZ, 1),
                               "cores": cores, "note": note})
        print(f"  {arm:<9} {s:8.5f} s {s * CLK_MHZ:7.1f} Mc {per_call * DM_CALLS:6} progs "
              f"{cores:4}c  {op.replace('DeviceOperation', '')}  {note}")

    print("\nMOVES (matmul that changes op class, ttnn.linear -> minimal_matmul)")
    for in0, in1, progs, arm, note in MOVES:
        ns, n, cores = _find(ix, "MatmulDeviceOperation", in0, in1=in1)
        mean_ns = ns / n
        s = mean_ns * progs / 1e9
        rec["moves"].append({"in0": list(in0), "in1": list(in1), "arm": arm,
                             "programs_per_fold": progs,
                             "signature_programs_in_csv": n,
                             "signature_per_call": round(n / DM_REPS, 3),
                             "mean_us_per_program": round(mean_ns / 1e3, 3),
                             "s_per_fold": round(s, 5), "Mcycles": round(s * CLK_MHZ, 1),
                             "cores": cores, "note": note})
        print(f"  {arm:<9} {s:8.5f} s {s * CLK_MHZ:7.1f} Mc {progs:6} progs {cores:4}c  "
              f"{mean_ns / 1e3:7.1f} us/prog  {note}")

    for arm in ("token", "atom", "atom_tail"):
        d = sum(x["s_per_fold"] for x in rec["deletes"] if x["arm"] == arm)
        m = sum(x["s_per_fold"] for x in rec["moves"] if x["arm"] == arm)
        rec["totals"][arm] = {"deletes_s": round(d, 5), "Mcycles": round(d * CLK_MHZ, 1),
                              "moves_s": round(m, 5), "delete_over_move": round(d / m, 3) if m else None}
    built = rec["totals"]["token"]["deletes_s"] + rec["totals"]["atom"]["deletes_s"]
    moved = rec["totals"]["token"]["moves_s"] + rec["totals"]["atom"]["moves_s"]
    rec["totals"]["built"] = {"deletes_s": round(built, 5), "Mcycles": round(built * CLK_MHZ, 1),
                              "moves_s": round(moved, 5),
                              "delete_over_move": round(built / moved, 3)}
    print("\nPER ARM")
    for arm, v in rec["totals"].items():
        print(f"  {arm:<9} delete {v['deletes_s']:.5f} s ({v['Mcycles']:.1f} Mc)  "
              f"move {v['moves_s']:.5f} s  ratio {v['delete_over_move']}")
    (OUT / "insitu.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"\nPASS -- {OUT / 'insitu.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
