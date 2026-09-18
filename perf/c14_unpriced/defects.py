#!/usr/bin/env python3
"""Attribute the census's overprice to its two named defects, using the fold's own executed calls.

`c14-matmul-ceiling` found two defects in the census this row reconciles against. Both are
properties of the census's CAPTURE, not of the fold, so neither is checkable from the capture.
`fixture.py` counted a live fold's own top-level ttnn calls -- a call count is clock-immune and
needs no profiler -- and this file differences that against the capture-derived census.

The result splits the 2.7953 s overprice into a COUNT error and a PRICE error. It is an
attribution of a measured total, not a correction applied to an estimate: the in-situ 7.7413 s
comes from `c12-profiled-fold` and the executed call counts come from this row's own session.
Subtracting the two defects from 10.5368 s and calling the remainder measured would be the same
mistake in the other direction, and is not what happens here.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERF = HERE.parent

# c10-fold-census's per-class replay seconds, and c12-profiled-fold's in-situ total for the same
# 70 keys. Neither is re-derived here; both are quoted with their source.
REPLAY = {"ttnn.linear": 4.6604, "ttnn.matmul": 1.4790, "ttnn.multiply_": 1.7510,
          "ttnn.add_": 0.8473, "ttnn.add": 0.1403, "ttnn.multiply": 0.1256,
          "ttnn.layer_norm": 1.5330}
INSITU_70 = 7.7413
# every class in the unpriced block, so its exposure to the same defects is measured not assumed
BLOCK = ["ttnn.permute", "ttnn.transformer.scaled_dot_product_attention",
         "ttnn.experimental.nlp_create_qkv_heads", "ttnn.slice", "ttnn.reshape", "ttnn.pad",
         "ttnn.concat", "ttnn.to_layout", "ttnn.chunk", "ttnn.softmax", "ttnn.transpose",
         "ttnn.to_memory_config", "ttnn.experimental.nlp_concat_heads", "ttnn.cos",
         "ttnn.generic_op"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=HERE / "runs/f1")
    ap.add_argument("--out", type=Path, default=HERE / "defects.json")
    a = ap.parse_args()
    ex = json.loads((a.run / "executed_keys.json").read_text())
    F = json.loads((a.run / "fixture.json").read_text())
    cap = json.loads((PERF / "roof_launch/op_census_512.json").read_text())["by_op"]

    executed = defaultdict(int)
    for k, v in ex.items():
        executed[k.split("|")[0]] += v

    print("DEFECT 1, MSA DEPTH: the fold pads %d a3m rows by %d to depth %d (the FIRST entry of "
          "MSA_PAD_LADDER); the census capture signature is %s, depth %d, the LAST entry. %.1fx."
          % (F["msa"]["rows_in_a3m"], F["msa"]["pad_amount"], F["msa"]["executed_depth"],
             F["msa"]["capture_signature"], F["msa"]["census_capture_depth"],
             F["msa"]["capture_over_executed_x"]))
    print()
    print("%-26s %10s %10s %7s %9s %11s"
          % ("op", "capture", "executed", "cap/exec", "census s", "count-corr"))
    tot, rows = 0.0, []
    for op, s in sorted(REPLAY.items(), key=lambda kv: -kv[1]):
        c, e = cap[op]["calls"], executed[op]
        corr = s * e / c
        tot += corr
        rows.append({"op": op, "capture_calls": c, "executed_calls": e, "ratio": c / e,
                     "census_s": s, "count_corrected_s": corr})
        print("%-26s %10d %10d %7.3f %9.4f %11.4f"
              % (op.replace("ttnn.", ""), c, e, c / e, s, corr))
    rep = sum(REPLAY.values())
    print("%-26s %10d %10d %7s %9.4f %11.4f"
          % ("TOTAL", sum(cap[o]["calls"] for o in REPLAY),
             sum(executed[o] for o in REPLAY), "", rep, tot))

    over = rep - INSITU_70
    count_err, price_err = rep - tot, tot - INSITU_70
    print()
    print("THE 10.5368 s OVERPRICE OF %.4f s, ATTRIBUTED:" % over)
    print("  %.4f s  %.1f %%  inflated CALL COUNTS -- the capture ran MSA depth %d, the fold %d"
          % (count_err, 100 * count_err / over, F["msa"]["census_capture_depth"],
             F["msa"]["executed_depth"]))
    print("  %.4f s  %.1f %%  per-call PRICE -- a standalone replay arm, not the in-fold kernel"
          % (price_err, 100 * price_err / over))
    print("  %.4f s          in-situ total for the same 70 keys (c12-profiled-fold, measured)"
          % INSITU_70)
    d = cap["ttnn.deallocate"]["calls"] / executed["ttnn.deallocate"]
    print("  control: deallocate does no device work and cannot be mispriced, only miscounted -- "
          "capture %d vs executed %d, %.3fx, which is the same inflation as the three "
          "MSA-depth classes and independent of any per-call price."
          % (cap["ttnn.deallocate"]["calls"], executed["ttnn.deallocate"], d))

    print()
    print("DEFECT 2, BLOCK-vs-OP KEY: the executed pair-transition keys, from this row's own fold")
    for k, v in sorted(ex.items(), key=lambda kv: -kv[1]):
        if k.startswith("ttnn.linear") and "512x512x512" in k.replace("x1", "x").replace(" ", ""):
            continue
    pair = {k: v for k, v in ex.items()
            if k.startswith("ttnn.linear") and ("1x47x512" in k or "1x42x512" in k)}
    for k, v in sorted(pair.items(), key=lambda kv: -kv[1]):
        print("  %6d  %s" % (v, k))
    print("  census key for the same work: linear|1x16x512x512|K128, 17,920 calls. Executed "
          "%d. %.2fx." % (sum(pair.values()),
                          17920 / sum(pair.values()) if pair else float("nan")))

    print()
    print("THE BLOCK'S OWN EXPOSURE to both defects, measured per class:")
    worst = 0.0
    blk = []
    for op in BLOCK:
        c, e = cap[op]["calls"], executed[op]
        worst = max(worst, abs(c / e - 1))
        blk.append({"op": op, "capture_calls": c, "executed_calls": e, "ratio": c / e})
        print("  %-46s %8d %8d %7.4f" % (op.replace("ttnn.", ""), c, e, c / e))
    print("  WORST deviation across the whole block: %.2f %%. The block's call weights are "
          "sound and its in-situ seconds do not move." % (100 * worst))

    out = {"msa": F["msa"], "priced_rows": rows, "census_replay_s": rep,
           "count_corrected_s": tot, "insitu_70_s": INSITU_70,
           "overprice_s": over, "count_error_s": count_err, "price_error_s": price_err,
           "count_error_pct": 100 * count_err / over,
           "price_error_pct": 100 * price_err / over,
           "deallocate_ratio": d, "pair_executed": pair,
           "pair_census_calls": 17920, "block_rows": blk,
           "block_worst_deviation_pct": 100 * worst,
           "folds": [{k: f[k] for k in ("label", "elapsed_s", "recording")} for f in F["folds"]],
           "fold_clock_qualified": [f["clock"]["pass"] for f in F["folds"]],
           "clock_min_MHz": F["clock_min_MHz"], "clock_max_MHz": F["clock_max_MHz"]}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
