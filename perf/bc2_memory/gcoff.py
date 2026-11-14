"""Issue 18: what a checkpoint-switch round leaves to the cycle collector.

Runs test_bindcraft2_hw's switching rounds with the collector off, reads held DRAM after each,
then collects once with DEBUG_SAVEALL and names the reference loops it freed.
"""
from __future__ import annotations

import gc
import json
import sys

sys.path.insert(0, "tests")
sys.path.insert(0, "perf/bc2_memory")

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()

import ttnn  # noqa: E402
import test_bindcraft2_hw as t  # noqa: E402
from boundary import _cycles  # noqa: E402
from bindcraft.af2 import MULTIMER_POOL  # noqa: E402
from tt_bio import bindcraft2  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402


def held():
    mv = ttnn.get_memory_view(get_device(), ttnn.BufferType.DRAM)
    return int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)


rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
params, protein_states, losses = t._pdl1_draw()
names = MULTIMER_POOL[:2]
out = {"held": []}
with bindcraft2.predictor(trunk="device", checkpoints=params, resident=1) as build:
    model = build(presets=names, models=names, data_dir=str(params), max_cache_size=1,
                  num_recycle=1, length_bucket_size=32)
    gc.collect()
    gc.disable()
    for i in range(rounds):
        model.sequence_gradients(protein_states, losses, model=names[i % 2])
        out["held"].append(held())
        print("round", i, out["held"][-1] / 1e9, flush=True)
    gc.set_debug(gc.DEBUG_SAVEALL)
    gc.collect()
    gc.set_debug(0)
    out["cycles"] = _cycles(gc.garbage, ttnn, paths=8)
    gc.garbage.clear()
    gc.collect()
    out["after_collect"] = held()
    gc.enable()
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:6000])
