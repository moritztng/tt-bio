#!/usr/bin/env python3
"""Which branch does each L1 gate take, as a function of token count? Read, not inferred.

`_l1_memory_config_if_it_fits` reads only `t.shape` and `t.dtype`, so a duck-typed stub gives
the SAME decision the live helper makes without allocating a 134 MB tensor 350 times. Three real
device tensors validate the stub against the helper before the sweep is believed.
"""
import json, sys
from pathlib import Path
import torch, ttnn
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tt_bio.tenstorrent as T

C_Z = 256
HEADROOMS = {"transpose_c2fix": 2.5, "l1_layer_norm_x10": 1.5}


class Stub:
    def __init__(self, n, c=C_Z):
        self.shape = [n, n, c]
        self.dtype = ttnn.bfloat16


def main():
    dev = T.get_device()
    grid = tuple(T.COMPUTE_GRID_MAIN)
    per_core = int(ttnn.get_max_worker_l1_unreserved_size())
    bank = int(T._l1_bank_bytes())
    a = dev.compute_with_storage_grid_size()
    out = {"ttnn": __import__("importlib.metadata", fromlist=["version"]).version("ttnn"),
           "device_grid": [int(a.x), int(a.y)], "compute_grid_main": list(grid),
           "num_cores": grid[0] * grid[1],
           "get_max_worker_l1_unreserved_size": per_core,
           "l1_bank_bytes": bank,
           "budget_bytes": per_core * grid[0] * grid[1]}

    # validate the stub against real device tensors at three sizes
    val = []
    for n in (298, 352, 384):
        t = ttnn.from_torch(torch.zeros(n, n, C_Z), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
               for n in range(64, 1025)]
        passing = [n for n, ok in dec if ok]
        first_fail = next((n for n, ok in dec if not ok and n > min(passing or [0])), None)
        sweep[name] = {
            "headroom": h,
            "last_passing_n": max(passing) if passing else None,
            "first_failing_n_above_298": next((n for n, ok in dec if not ok and n >= 298), None),
            "monotone": all(not ok for n, ok in dec if n > (max(passing) if passing else 0)),
            "at": {str(n): T._l1_memory_config_if_it_fits(Stub(n), h) is ttnn.L1_MEMORY_CONFIG
                   for n in (117, 256, 298, 320, 352, 353, 384, 416, 448, 457, 458, 480, 509, 512, 640, 1095)},
        }
        sweep[name]["bytes_at"] = {str(n): n * ((n + 31) // 32 * 32) * C_Z * 2
                                   for n in (298, 320, 352, 384, 448, 480, 512, 1095)}
    out["sweep"] = sweep
    print(json.dumps(out, indent=2))
    Path(__file__).with_name("fit_decision_qb2c0.json").write_text(json.dumps(out, indent=2))


main()
