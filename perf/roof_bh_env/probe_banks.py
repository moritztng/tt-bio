#!/usr/bin/env python3
"""The DRAM bank count TRIMUL_GP_BANK_SPLIT's neutrality argument rests on, read off the part."""
import json
import ttnn
import tt_bio.tenstorrent as T
dev = T.get_device()
out = {"arch": str(dev.arch())}
for name in ("dram_grid_size", "num_dram_channels", "dram_size_per_channel",
             "compute_with_storage_grid_size", "l1_size_per_core"):
    try:
        v = getattr(dev, name)()
        out[name] = str(v)
    except Exception as e:                                                  # noqa: BLE001
        out[name] = "n/a (%s)" % type(e).__name__
out["l1_bank_bytes"] = T._l1_bank_bytes()
out["max_worker_l1_unreserved"] = ttnn.get_max_worker_l1_unreserved_size()
print(json.dumps(out), flush=True)
T.cleanup()
