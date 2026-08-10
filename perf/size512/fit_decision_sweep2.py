#!/usr/bin/env python3
"""All four L1 gates, decision read as a function of token count. Stub operands, validated.

Gate 1 C2FIX  `_transpose_memory_config` -> `_l1_memory_config_if_it_fits(x, 2.5)`
Gate 2 X10    `_l1_layer_norm`           -> `_l1_memory_config_if_it_fits(x, 1.5)`
Gate 3 X7     `_pair_proj_config(x, w, bw_cap=_PAIR_PROJ_L1_BW, out_l1=True)` is not None
Gate 4 X2     `_pair_proj_config(x, w, bw_cap=_NARROW_PROJ_BW, out_l1=...)` for the narrow class
"""
import json, sys
from pathlib import Path
import ttnn
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tt_bio.tenstorrent as T

C_Z = 256


class Stub:
    def __init__(self, shape, dtype=None):
        self.shape = list(shape)
        self.dtype = dtype or ttnn.bfloat16
        self.padded_shape = tuple(self.shape)


def pair(n):
    return Stub([n, n, C_Z])


def gates(n):
    x = pair(n)
    w_wide = Stub([C_Z, C_Z])
    w_narrow = Stub([C_Z, 1])       # PWA per-head z->bias
    w_bias16 = Stub([C_Z, 16])      # AttentionPairBias z->bias
    w_tmpl = Stub([C_Z, 64])        # template z projection
    g = {
        "c2fix_transpose_l1": T._l1_memory_config_if_it_fits(x, 2.5) is ttnn.L1_MEMORY_CONFIG,
        "x10_l1_layer_norm": T._l1_memory_config_if_it_fits(x, 1.5) is ttnn.L1_MEMORY_CONFIG,
        "x7_pair_proj_l1_out": T._pair_proj_config(x, w_wide, bw_cap=T._PAIR_PROJ_L1_BW,
                                                   out_l1=True) is not None,
        "pair_proj_dram_tuned": T._pair_proj_config(x, w_wide) is not None,
    }
    for lbl, w in (("narrow_c1", w_narrow), ("bias_c16", w_bias16), ("tmpl_c64", w_tmpl)):
        g[f"x2_narrow_{lbl}_l1out"] = T._pair_proj_config(x, w, bw_cap=T._NARROW_PROJ_BW,
                                                          out_l1=True) is not None
        g[f"x2_narrow_{lbl}_dram"] = T._pair_proj_config(x, w, bw_cap=T._NARROW_PROJ_BW,
                                                          out_l1=False) is not None
    return g


def main():
    T.get_device()
    grid = tuple(T.COMPUTE_GRID_MAIN)
    keys = list(gates(298).keys())
    rows = {n: gates(n) for n in range(64, 1153)}
    cliffs = {}
    for k in keys:
        ok = [n for n, r in rows.items() if r[k]]
        cliffs[k] = {
            "last_passing_n": max(ok) if ok else None,
            "first_failing_above_298": next((n for n in sorted(rows) if n >= 298 and not rows[n][k]), None),
            "monotone_above_last": all(not rows[n][k] for n in sorted(rows) if ok and n > max(ok)),
        }
    out = {
        "ttnn": __import__("importlib.metadata", fromlist=["version"]).version("ttnn"),
        "grid": list(grid), "num_cores": grid[0] * grid[1],
        "per_core_unreserved": int(ttnn.get_max_worker_l1_unreserved_size()),
        "l1_bank_bytes": int(T._l1_bank_bytes()),
        "flags": {"_PAIR_PROJ_BW": T._PAIR_PROJ_BW, "_PAIR_PROJ_L1_BW": T._PAIR_PROJ_L1_BW,
                  "_NARROW_PROJ_BW": T._NARROW_PROJ_BW, "_PAIR_PROJ_L1_OUT": T._PAIR_PROJ_L1_OUT,
                  "_PWA_L1_NORM": T._PWA_L1_NORM, "_TEMPLATE_L1_NORM": T._TEMPLATE_L1_NORM},
        "cliffs": cliffs,
        "at": {str(n): rows[n] for n in (117, 298, 320, 352, 366, 384, 448, 457, 480, 509, 512, 1095)},
    }
    print(json.dumps(out, indent=2))
    Path(__file__).with_name("fit_decision2_qb2c0.json").write_text(json.dumps(out, indent=2))


main()
