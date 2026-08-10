#!/usr/bin/env python3
"""qb1 card 0: the L1 budget, every capacity gate swept over token count, and the copy roofs.

Nothing here is inherited. The per-bank budget, the core grid and the two bandwidth roofs are read
or measured on this card in this process, because the whole question of this leg is whether qb1s
13x10 grid (18 % more L1 than qb2s 11x10) changes the 448 aa verdict, and an 18 % that came from
someone elses table is not evidence.
"""
import json, statistics as st, sys, time
from pathlib import Path
import torch, ttnn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tt_bio.tenstorrent as T

C_Z = 256
HEADROOMS = {"transpose_c2fix_h2.5": 2.5, "l1_layer_norm_h1.5": 1.5}
SIZES = [117, 256, 298, 320, 352, 353, 384, 385, 416, 448, 449, 457, 458, 464, 480, 496,
         506, 507, 509, 512, 640, 1095]


class Stub:
    """Duck type: _l1_memory_config_if_it_fits reads only .shape and .dtype."""
    def __init__(self, n, c=C_Z):
        self.shape = [n, n, c]
        self.dtype = ttnn.bfloat16


def roof(dev, n, dst):
    """Measured clone bandwidth at the pair shape, GB/s counting read+write."""
    mc = ttnn.L1_MEMORY_CONFIG if dst == "L1" else ttnn.DRAM_MEMORY_CONFIG
    t = ttnn.from_torch(torch.zeros(n, n, C_Z), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    nbytes = n * ((n + 31) // 32 * 32) * C_Z * 2
    ms = []
    for i in range(7):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = ttnn.clone(t, memory_config=mc)
        ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(o)
    ttnn.deallocate(t)
    med = st.median(ms[2:])
    return {"n": n, "dst": dst, "bytes": nbytes, "ms_median": round(med, 4),
            "gbps_rw": round(2 * nbytes / (med * 1e-3) / 1e9, 1), "ms_all": [round(x, 4) for x in ms]}


def main():
    dev = T.get_device()
    a = dev.compute_with_storage_grid_size()
    grid = tuple(int(x) for x in T.COMPUTE_GRID_MAIN)
    per_core = int(ttnn.get_max_worker_l1_unreserved_size())
    bank = int(T._l1_bank_bytes())
    import importlib.metadata as im
    out = {
        "host": "qb1", "card": 0, "ttnn": im.version("ttnn"),
        "device_compute_grid": [int(a.x), int(a.y)],
        "compute_grid_main": list(grid), "num_cores": grid[0] * grid[1],
        "get_max_worker_l1_unreserved_size": per_core,
        "l1_allocator_bytes_per_bank": bank,
        "fit_budget_bytes": per_core * grid[0] * grid[1],
        "trimul_l1_max_seq": int(T._trimul_l1_max_seq()),
        "TRIANGLE_MULT_L1_MAX_SEQ": int(T.TRIANGLE_MULT_L1_MAX_SEQ),
        "TRIANGLE_MULT_L1_MAX_SEQ_FAST": int(T.TRIANGLE_MULT_L1_MAX_SEQ_FAST),
        "TRIANGLE_MULT_L1_MAX_SEQ_FAST_13X10": int(T.TRIANGLE_MULT_L1_MAX_SEQ_FAST_13X10),
        "TRIANGLE_MULT_L1_CHUNK_BUDGET": int(T.TRIANGLE_MULT_L1_CHUNK_BUDGET),
    }

    # stub validated against real device tensors before any of the sweep is believed
    val = []
    for n in (298, 384, 448):
        t = ttnn.from_torch(torch.zeros(n, n, C_Z), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        row = {"n": n, "real_shape": [int(d) for d in t.shape]}
        for name, h in HEADROOMS.items():
            real = T._l1_memory_config_if_it_fits(t, h) is ttnn.L1_MEMORY_CONFIG
            stub = T._l1_memory_config_if_it_fits(Stub(n), h) is ttnn.L1_MEMORY_CONFIG
            row[name] = {"real": real, "stub": stub, "agree": real == stub}
        ttnn.deallocate(t)
        val.append(row)
    out["stub_validation"] = val
    out["stub_valid"] = all(r[k]["agree"] for r in val for k in HEADROOMS)

    sweep = {}
    for name, h in HEADROOMS.items():
        dec = [(n, T._l1_memory_config_if_it_fits(Stub(n), h) is ttnn.L1_MEMORY_CONFIG)
               for n in range(64, 1200)]
        passing = [n for n, ok in dec if ok]
        sweep[name] = {"headroom": h, "last_passing_n": max(passing) if passing else None,
                       "at": {str(n): T._l1_memory_config_if_it_fits(Stub(n), h)
                              is ttnn.L1_MEMORY_CONFIG for n in SIZES}}
    out["fit_sweep"] = sweep

    per_n = {}
    for n in SIZES:
        nt = (n + 31) // 32
        nbytes = n * (nt * 32) * C_Z * 2
        pc = T._triangle_mul_program_config(nt)
        per_n[str(n)] = {
            "pair_bytes": nbytes,
            "per_bank_bytes": round(nbytes / (grid[0] * grid[1]), 1),
            "pct_of_bank": round(100.0 * nbytes / (grid[0] * grid[1]) / bank, 2),
            "trimul_chunk_size": int(T._trimul_chunk_size(n, 128)),
            "trimul_memcfg": "L1" if T._triangle_mul_memory_config(n).buffer_type
                             == ttnn.BufferType.L1 else "DRAM",
            "tri_pc": {"in0_block_w": int(pc.in0_block_w), "per_core_M": int(pc.per_core_M),
                       "per_core_N": int(pc.per_core_N)},
        }
    out["per_size"] = per_n

    out["roofs"] = [roof(dev, n, d) for n in (320, 448) for d in ("DRAM", "L1")]
    Path(__file__).with_name("probe_qb1c0.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


main()
