#!/usr/bin/env python3
"""The grid facts, read AFTER the device is open, plus the two fit verdicts.

`COMPUTE_GRID_MAIN` and `CORE_GRID_MAIN` are rebound inside `get_device()`, so a module-scope
`from tt_bio.tenstorrent import CORE_GRID_MAIN` freezes the pre-open default. Read as module
attributes here, and the by-value copy is printed beside them to show the two differ.
"""
import json, sys
from pathlib import Path
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio.tenstorrent import CORE_GRID_MAIN as BY_VALUE   # deliberately the wrong way

dev = T.get_device()
dg = dev.compute_with_storage_grid_size()
res = {"ttnn": "0.67.4" if "tt-bio-dev" in sys.executable else sys.executable,
       "python": sys.executable,
       "device_compute_with_storage_grid": [dg.x, dg.y],
       "COMPUTE_GRID_MAIN_after_open": list(T.COMPUTE_GRID_MAIN),
       "CORE_GRID_MAIN_after_open": [T.CORE_GRID_MAIN.x, T.CORE_GRID_MAIN.y],
       "CORE_GRID_MAIN_imported_by_value": [BY_VALUE.x, BY_VALUE.y],
       "cores": T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1],
       "trimul_chunk_size_298_128": T._trimul_chunk_size(298, 128),
       "trimul_l1_max_seq": T._trimul_l1_max_seq(),
       "l1_bank_bytes": T._l1_bank_bytes(),
       "max_worker_l1_unreserved": int(ttnn.get_max_worker_l1_unreserved_size())}
z = ttnn.from_torch(torch.randn(1, 298, 320, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
res["pair_tensor_MB"] = 298 * 320 * 256 * 2 / 1e6
res["fit_verdict_headroom_1.5"] = str(T._l1_memory_config_if_it_fits(z, 1.5).buffer_type)
res["fit_verdict_headroom_2.5"] = str(T._l1_memory_config_if_it_fits(z, 2.5).buffer_type)
print(json.dumps(res, indent=1), flush=True)
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
