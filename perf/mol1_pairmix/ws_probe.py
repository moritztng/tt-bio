"""Which Mol-1 transition calls the weight-stationary kernel serves: one short fold (1 cycle, 2 steps), every
Transition call counted by (width, rows, dtype, destination, served). Usage: same env as e2e_tt.py;
python perf/mol1_pairmix/ws_probe.py CKPT SEQ  (TT_BIO_LEVERS picks the set)."""
import collections
import sys

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import torch  # noqa: E402
import ttnn  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.protenix import Protenix  # noqa: E402
from tt_bio.protenix_data import build_protein_features  # noqa: E402

seen = collections.Counter()
orig = T.Transition._weight_stationary


def ws(self, x, memory_config, add_to_input=False):
    out = orig(self, x, memory_config, add_to_input)
    dest = "dram" if memory_config is None or memory_config.buffer_type == ttnn.BufferType.DRAM else "l1"
    seen[(int(x.shape[-1]), int(x.shape[1]), str(x.dtype).split(".")[-1], len(x.shape), dest, add_to_input,
          out is not None)] += 1
    return out


T.Transition._weight_stationary = ws
model = Protenix.load_from_checkpoint(sys.argv[1])
print("levers:", sorted(T._LEVERS) if hasattr(T, "_LEVERS") else "?", flush=True)
with torch.no_grad():
    model.fold(build_protein_features(sys.argv[2]), n_step=2, n_sample=1, seed=101, n_cycles=1)
print("(width, rows, dtype, rank, dest, add_to_input, served): calls")
for k, v in sorted(seen.items()):
    print(k, v)
