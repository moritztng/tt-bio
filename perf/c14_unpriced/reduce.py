#!/usr/bin/env python3
"""Close the fold's accounting, and say which of the three campaign numbers is wrong.

Three numbers cannot all be right:

    device term in situ        13.2090 s   c12-profiled-fold, one session, during-sampled 1350
    census-priced 70 keys      10.5368 s   c10-fold-census, a REPLAY SUBSTITUTION, not a fold
    generic_op in situ          3.3743 s   c12-genericop-rate (3.3830 s in c12-profiled-fold)

    10.5368 + 3.3743 = 13.9111  against 13.2090, over-subscribed by 0.7021 s, with 51 refused
    keys still owed a second each.

The resolution is in `c12-profiled-fold`'s own artifact and nobody did the subtraction. Its
`per_class_s` field holds the in-situ cost of the eight census classes, and its `unsplit_s`
field holds the device ops it could NOT attribute to a ttnn class -- which is exactly this
block. So the in-situ split already exists for every term and it closes; what was wrong was
mixing a replay estimate into an in-situ total.

This reducer does that subtraction against the artifact, then puts the standalone arms measured
by `replay.py` beside the in-situ per-device-op table as an independent second instrument.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# c12-profiled-fold/runs/composed.json, quoted with its path so every number is traceable
FOLD_S = 14.8810
DEVICE_S = 13.2090
NONDEVICE_S = 1.6720
GENERIC_INSITU_S = 3.3830          # GenericOpDeviceOperation
GENERIC_RATE_ROW_S = 3.3743        # c12-genericop-rate's independent weighting, 0.26 % apart
CENSUS_REPLAY = {"linear": 4.6604, "matmul": 1.4790, "multiply_": 1.7510, "add_": 0.8473,
                 "add": 0.1403, "multiply": 0.1256, "layer_norm": 1.5330}
CENSUS_INSITU = {"linear": 3.3992, "matmul": 0.5713, "multiply_": 1.4925, "add_": 0.6348,
                 "add": 0.1309, "multiply": 0.1318, "layer_norm": 1.3808}
# the 15 device ops outside the 8 census classes and generic_op: THE BLOCK, measured in situ
BLOCK_INSITU = {"TransposeDeviceOperation": 0.5174, "SDPAOperation": 0.4335,
                "NlpCreateHeadsDeviceOperation": 0.3007, "SliceDeviceOperation": 0.1917,
                "ReshapeViewDeviceOperation": 0.1571, "PadDeviceOperation": 0.0864,
                "PermuteDeviceOperation": 0.0823, "ConcatDeviceOperation": 0.0748,
                "UntilizeDeviceOperation": 0.0590, "EmbeddingsDeviceOperation": 0.0571,
                "TilizeDeviceOperation": 0.0536, "SoftmaxDeviceOperation": 0.0323,
                "CopyDeviceOperation": 0.0243, "NLPConcatHeadsDeviceOperation": 0.0131,
                "UnaryNgDeviceOperation": 0.0015}
# EmbeddingsDeviceOperation is in NO launch-key census: the graph capture records no
# ttnn.embedding top-level call, so it is in the block by device op but in none of the 133 keys.
NOT_IN_ANY_KEY = "EmbeddingsDeviceOperation"
# which arm class each device op collects, so the standalone table can be put beside it
ARM_TO_OP = {"permute": "TransposeDeviceOperation", "transpose": "TransposeDeviceOperation",
             "sdpa": "SDPAOperation", "nlp_create_qkv_heads": "NlpCreateHeadsDeviceOperation",
             "nlp_concat_heads": "NLPConcatHeadsDeviceOperation",
             "slice": "SliceDeviceOperation", "chunk": "SliceDeviceOperation",
             "reshape": "ReshapeViewDeviceOperation", "pad": "PadDeviceOperation",
             "concat": "ConcatDeviceOperation", "to_layout": "TilizeDeviceOperation",
             "softmax": "SoftmaxDeviceOperation", "to_memory_config_l1": "CopyDeviceOperation",
             "cos": "UnaryNgDeviceOperation", "matmul": "MatmulDeviceOperation"}
# Tilize and Untilize are one to_layout class; Permute and Transpose are one permute class
_M = [["TilizeDeviceOperation", "UntilizeDeviceOperation"],
      ["TransposeDeviceOperation", "PermuteDeviceOperation"]]
# keyed on BOTH members: the loop walks device ops by descending seconds, and Untilize (0.0590)
# comes before Tilize (0.0536), so a one-way map silently splits the to_layout class in the
# printed table while still summing it correctly -- a display lie is still a lie.
MERGE = {m: g for g in _M for m in g}
LAYOUT_OPS = {"TransposeDeviceOperation", "PermuteDeviceOperation", "SliceDeviceOperation",
              "ReshapeViewDeviceOperation", "PadDeviceOperation", "ConcatDeviceOperation",
              "UntilizeDeviceOperation", "TilizeDeviceOperation", "CopyDeviceOperation"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=HERE / "runs/s1/replay.json")
    ap.add_argument("--out", type=Path, default=HERE / "closure.json")
    a = ap.parse_args()
    R = json.loads(a.run.read_text())
    keys = json.loads((HERE / "keys_512.json").read_text())

    roof = R["roofs"]["dram"]["GBps"]
    priced_insitu = sum(CENSUS_INSITU.values())
    priced_replay = sum(CENSUS_REPLAY.values())
    block_insitu = sum(BLOCK_INSITU.values())
    embed = BLOCK_INSITU[NOT_IN_ANY_KEY]
    refused_insitu = block_insitu - embed
    total = priced_insitu + GENERIC_INSITU_S + block_insitu

    # standalone arms, calls-weighted, only blocks whose during-samples qualified at 1350
    rows = [r for r in R["rows"] if r.get("s_per_call_qualified")]
    unq = [r for r in R["rows"] if not r.get("s_per_call_qualified")]
    st_s, st_calls, st_B = defaultdict(float), defaultdict(float), defaultdict(float)
    for r in rows:
        op = ARM_TO_OP.get(r["arm"], r["arm"])
        st_s[op] += r["s_per_call_qualified"] * r["calls"]
        st_calls[op] += r["calls"]
        st_B[op] += r["B"]
    st_total = sum(v for k, v in st_s.items() if k != "MatmulDeviceOperation")

    print("CLOSURE, every term in situ, one session (c12-profiled-fold/runs/composed.json):")
    print("  %-46s %9.4f s" % ("70 census-priced keys, IN SITU", priced_insitu))
    print("  %-46s %9.4f s" % ("generic_op (1 key, no LAUNCH_ARM), IN SITU", GENERIC_INSITU_S))
    print("  %-46s %9.4f s" % ("51 remaining refused keys, IN SITU", refused_insitu))
    print("  %-46s %9.4f s" % ("Embeddings, in NO launch-key census", embed))
    print("  %-46s %9.4f s" % ("sum", total))
    print("  %-46s %9.4f s" % ("in-situ device term of record", DEVICE_S))
    print("  %-46s %+9.4f s" % ("RESIDUAL", total - DEVICE_S))
    print("  %-46s %9.4f s" % ("non-device remainder", NONDEVICE_S))
    print("  %-46s %9.4f s" % ("fold", DEVICE_S + NONDEVICE_S))
    print("  %-46s %+9.4f s" % ("RESIDUAL against the 14.8810 s fold",
                                DEVICE_S + NONDEVICE_S - FOLD_S))
    over = priced_replay - priced_insitu
    print()
    print("WHICH NUMBER IS WRONG: the 10.5368 s census figure, by +%.4f s (%.4fx too high)."
          % (over, priced_replay / priced_insitu))
    decomp = over - refused_insitu - embed - (GENERIC_INSITU_S - GENERIC_RATE_ROW_S)
    print("  the orchestrator's 0.7021 s over-subscription decomposes exactly:")
    print("    +%.4f  replay overprice on the 70 keys" % over)
    print("    -%.4f  real in-situ cost of the 51 refused keys" % refused_insitu)
    print("    -%.4f  Embeddings, in neither census" % embed)
    print("    -%.4f  generic_op, profiled-fold minus genericop-rate"
          % (GENERIC_INSITU_S - GENERIC_RATE_ROW_S))
    print("    =%.4f  against the orchestrator's 0.7021 s" % decomp)
    naive = DEVICE_S - priced_replay
    print()
    print("THE ADVERTISED 2.68 s: %.4f s, and it is right to within %.4f s for the WRONG "
          "reasons -- it adds a +%.4f s replay overprice to a -%.4f s omission of generic_op "
          "from the residual it computed, and the two nearly cancel."
          % (naive, naive - (block_insitu), over, GENERIC_INSITU_S))

    print()
    print("TABLE: the block per device op, IN SITU, beside this session's standalone arms.")
    print("%-32s %8s %8s %9s %8s %7s %8s"
          % ("device op (in situ)", "insitu s", "stand s", "calls", "GB", "GB/s", "% roof"))
    tbl = []
    done = set()
    for op, s in sorted(BLOCK_INSITU.items(), key=lambda kv: -kv[1]):
        if op in done:
            continue
        group = MERGE.get(op, [op])
        if any(g in done for g in group):
            continue
        done.update(group)
        s_ins = sum(BLOCK_INSITU[g] for g in group)
        s_std = sum(st_s.get(g, 0.0) for g in group)
        calls = sum(st_calls.get(g, 0.0) for g in group)
        B = sum(st_B.get(g, 0.0) for g in group)
        gbps = (B / s_ins / 1e9) if s_ins and B else None
        row = {"op": "+".join(g.replace("DeviceOperation", "") for g in group),
               "insitu_s": s_ins, "standalone_s": s_std or None, "calls": calls,
               "GB": B / 1e9, "GBps_insitu": gbps,
               "pct_roof": (100 * gbps / roof) if gbps else None,
               "layout": all(g in LAYOUT_OPS for g in group)}
        tbl.append(row)
        print("%-32s %8.4f %8s %9d %8.1f %7s %8s"
              % (row["op"], s_ins, "%.4f" % s_std if s_std else "-", calls, row["GB"],
                 "%.1f" % gbps if gbps else "-",
                 "%.0f%%" % row["pct_roof"] if row["pct_roof"] else "-"))
    lay = sum(r["insitu_s"] for r in tbl if r["layout"])
    cmp_ = sum(r["insitu_s"] for r in tbl if not r["layout"])
    print("%-32s %8.4f" % ("layout/data-movement subtotal", lay))
    print("%-32s %8.4f" % ("real-compute subtotal", cmp_))
    print("%-32s %8.4f  standalone %.4f  ratio %.3fx"
          % ("BLOCK", block_insitu, st_total, st_total / refused_insitu))
    print()
    print("DIRECTIONAL PREDICTION (registered before the session): the standalone sum comes out "
          "AT OR ABOVE the in-situ %.4f s. Measured %.4f s, ratio %.3fx, predicted band "
          "1.0-2.5x -> %s" % (refused_insitu, st_total, st_total / refused_insitu,
                              "HELD" if 1.0 <= st_total / refused_insitu <= 2.5 else "REFUTED"))
    out = {"fold_s": FOLD_S, "device_s": DEVICE_S, "nondevice_s": NONDEVICE_S,
           "priced_insitu_s": priced_insitu, "priced_replay_s": priced_replay,
           "replay_overprice_s": over, "replay_overprice_x": priced_replay / priced_insitu,
           "generic_op_insitu_s": GENERIC_INSITU_S,
           "block_insitu_s": block_insitu, "refused51_insitu_s": refused_insitu,
           "embeddings_s": embed, "sum_s": total, "residual_s": total - DEVICE_S,
           "naive_residual_s": naive,
           "standalone_total_s": st_total,
           "standalone_over_insitu_x": st_total / refused_insitu,
           "dram_roof_GBps": roof,
           "cube_open_TFLOPs": R["roofs"]["cube_open"]["TFLOPs"],
           "cube_close_TFLOPs": R["roofs"]["cube_close"]["TFLOPs"],
           "clock_min_MHz": R["clock_min_MHz"], "clock_max_MHz": R["clock_max_MHz"],
           "clock_samples": R["clock_samples"],
           "qualified_keys": len(rows), "unqualified_keys": [r["key"] for r in unq],
           "layout_subtotal_s": lay, "compute_subtotal_s": cmp_,
           "table": tbl, "n_keys_total": keys["n_keys"],
           "buckets": keys["buckets"]}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
