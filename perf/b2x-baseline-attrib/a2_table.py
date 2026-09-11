#!/usr/bin/env python3
"""A1 + A2: the phase table on clean times, and what the remainder is made of.

`roof_table.py` ranks every phase on per-call times taken with a device sync on both sides of
every bracket. `remainder_probe.py` re-takes the same tree with the sync only at depth 1, and its
instrumented fold costs nothing measurable (23.924 s against 23.906 s plain), so its top-level
times are clean. This reconciles the two and prices the residual.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STREAM = 429.9e9
COMPUTE = 85.96e12
FLOP = {"pairformer block": 502796386304, "MSA block": 595167543296,
        "token DiT layer": 12280922112, "atom transformer layer": 4580179968}

rd = json.loads((HERE / "roof_deficit_512_qb2c1.json").read_text())
rem = json.loads((HERE / "remainder_512_qb2c1.json").read_text())
by_sig = {r["sig"]: r for r in rd["rows"]}
inst = rem["instrumented"]
top = inst["top_level"]

MB = lambda s: by_sig[s]["real_MB_per_call"]                                  # noqa: E731

# --- bytes each top-level class moves, from the corrected per-call capture counts -------------
trunk_GB = (256 * MB("PairformerLayer|1x512x384,1x512x512x128")
            + 16 * MB("MSALayer|1x512x512x128,1x1024x512x64")) / 1e3
conf_GB = 8 * MB("PairformerLayer|1x512x384,1x512x512x128") / 1e3
diff_GB = 200 * MB("DiffusionModule|") / 1e3
trunk_FLOP = 256 * FLOP["pairformer block"] + 16 * FLOP["MSA block"]
conf_FLOP = 8 * FLOP["pairformer block"]
diff_FLOP = 200 * (24 * FLOP["token DiT layer"] + 6 * FLOP["atom transformer layer"])

rows = []
for name, GB, F in (("TrunkModule (4 recycles: 256 pairformer + 16 MSA)", trunk_GB, trunk_FLOP),
                    ("DiffusionModule (200 sampler steps)", diff_GB, diff_FLOP),
                    ("PairformerModule (8 confidence blocks)", conf_GB, conf_FLOP)):
    key = name.split(" ")[0]
    s = top[key]["incl_s"]
    rows.append({"phase": name, "s_per_fold": round(s, 3), "GB_per_fold": round(GB, 1),
                 "ops_per_fold": top[key]["ops"],
                 "GBps": round(GB * 1e9 / s / 1e9, 1),
                 "pct_stream": round(100 * GB * 1e9 / s / STREAM, 1),
                 "TFLOPs": round(F / s / 1e12, 2),
                 "pct_compute": round(100 * F / s / COMPUTE, 1),
                 "us_per_op": round(1e6 * s / top[key]["ops"], 1),
                 "deficit_s": round(s - GB * 1e9 / (0.8 * STREAM), 3)})
rows.append({"phase": "residual: everything outside every device class",
             "s_per_fold": inst["residual_s"], "GB_per_fold": 0.0,
             "ops_per_fold": inst["outside_every_class"]["ops"],
             "GBps": 0.0, "pct_stream": 0.0, "TFLOPs": 0.0, "pct_compute": 0.0,
             "us_per_op": None, "deficit_s": inst["residual_s"]})

# --- how much the pass-1 per-call syncs inflated each phase -----------------------------------
trunk_synced = (256 * by_sig["PairformerLayer|1x512x384,1x512x512x128"]["ms_per_call"]
                + 16 * by_sig["MSALayer|1x512x512x128,1x1024x512x64"]["ms_per_call"]) / 1e3
diff_inner_synced = (24 * by_sig["DiffusionTransformerLayer|1x512x768,1x512x768"]["ms_per_call"]
                     + 6 * by_sig["DiffusionTransformerLayer|1x224x32x128,1x224x32x128"]["ms_per_call"])
diff_clean = top["DiffusionModule"]["median_ms"]
k = diff_clean / diff_inner_synced

sync = {
    "trunk_sum_of_synced_per_call_s": round(trunk_synced, 3),
    "trunk_clean_s": top["TrunkModule"]["incl_s"],
    "trunk_inflation_pct": round(100 * (trunk_synced / top["TrunkModule"]["incl_s"] - 1), 1),
    "diffusion_step_synced_ms": by_sig["DiffusionModule|"]["ms_per_call"],
    "diffusion_step_clean_ms": diff_clean,
    "diffusion_step_inflation_pct": round(100 * (by_sig["DiffusionModule|"]["ms_per_call"] / diff_clean - 1), 1),
    "diffusion_inner_sum_synced_ms": round(diff_inner_synced, 3),
    "deflation_factor_for_diffusion_layers": round(k, 4),
}
for s in ("DiffusionTransformerLayer|1x512x768,1x512x768",
          "DiffusionTransformerLayer|1x224x32x128,1x224x32x128"):
    r = by_sig[s]
    ms = r["ms_per_call"] * k
    sync[s] = {"synced_ms": r["ms_per_call"], "clean_ms": round(ms, 4),
               "pct_stream_synced": r["pct_stream_roof"],
               "pct_stream_clean": round(100 * r["real_MB_per_call"] * 1e6 / (ms / 1e3) / STREAM, 1)}

# --- inside the pairformer block ---------------------------------------------------------------
blk = by_sig["PairformerLayer|1x512x384,1x512x512x128"]
sub = []
for s, per_block in (("TriangleMultiplication|1x512x512x128,1x512x512", 2),
                     ("TriangleAttention|1x512x512x128,1x1x1x512", 2),
                     ("Transition|1x512x512x128", 1),
                     ("AttentionPairBias|1x512x384,1x512x512x128", 1),
                     ("Transition|1x512x384", 1)):
    r = by_sig[s]
    sub.append({"sub_unit": s, "per_block": per_block,
                "ms_per_block": round(per_block * r["ms_per_call"], 3),
                "MB_per_block": round(per_block * r["real_MB_per_call"], 1),
                "ops_per_block": per_block * r["ops_per_call"],
                "pct_stream": r["pct_stream_roof"], "us_per_op": r["us_per_op"]})
sub.sort(key=lambda r: -r["ms_per_block"])
acc_ms = sum(r["ms_per_block"] for r in sub)
acc_ops = sum(r["ops_per_block"] for r in sub)

out = {"clean_phase_table": rows, "sync_inflation": sync,
       "pairformer_block": {"block_ms": blk["ms_per_call"], "block_ops": blk["ops_per_call"],
                            "sub_units": sub, "accounted_ms": round(acc_ms, 3),
                            "accounted_ops": acc_ops},
       "fold": {"plain_median_s": rem["plain_median_s"], "instrumented_s": inst["fold_s"],
                "instrument_cost_pct": round(100 * (inst["fold_s"] / rem["plain_median_s"] - 1), 2),
                "n_ttnn_calls_charged": inst["n_ttnn_calls"],
                "top_level_s": inst["top_level_s"], "residual_s": inst["residual_s"]}}
(HERE / "a2_remainder_512_qb2c1.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
