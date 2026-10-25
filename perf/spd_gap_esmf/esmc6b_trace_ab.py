"""ESMC-6B embed: eager forward vs a captured trace of the same stack, one process, one load.

The 6B embed path (ESMCLanguageModel, batch 1) runs eager. phase_prof on Blackhole puts its host enqueue at 44 ms against
~83 ms of Tracy device time and ~96-104 ms of wall, so the forward is not host-bound end to end, but the ~1,100 ops a
forward each leave a dispatch gap on the device that a trace replay shortens. This measures what a trace path would pay
before one is built into the model: arms alternate eager / traced for --rounds, each --repeat timed calls of the full
page step (upload, forward, readback of the final-norm state), and the traced output is compared with the eager one.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--fast", action="store_true")
ap.add_argument("--seq-file", default="scripts/gpu_vs_tt/fixtures/prot512.seq")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--repeat", type=int, default=7)
ap.add_argument("--region-mb", type=int, default=256)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.environ.setdefault("TT_BIO_TRACE_REGION_MB", str(a.region_mb))

import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio import esmc as E  # noqa: E402
from tt_bio.tenstorrent import capture_trace, get_device  # noqa: E402

torch.set_grad_enabled(False)
dev = get_device(trace="esmc")
model = E.load_esmc("esmc-6b", fast=a.fast)
m = model.module
seq = Path(a.seq_file).read_text().strip()
tokens = E.tokenize(seq)
Lm = tokens.shape[1]
Lb = -(-Lm // E.BUCKET) * E.BUCKET

# the inputs ESMCLanguageModel.forward builds for this sequence, kept on the device for the trace
ids = torch.nn.functional.pad(tokens, (0, Lb - Lm), value=E.PAD_TOKEN)
mask = torch.zeros(1, Lb, Lb)
mask[:, :, Lm:] = float("-inf")
lens = E._key_lens(mask)
kv = torch.ones(1, 1, Lb, 1)
kv[:, :, Lm:, :] = 0.0
host_ids = lambda: ttnn.from_torch(ids.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
tok_d = ttnn.from_torch(ids.to(torch.int32), device=dev, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
mask_d = E._lm_mask(mask, lens, dev)
kv_d = ttnn.from_torch(kv.to(torch.bfloat16), device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
cos, sin = E.rope_tables(Lb, m.norm_weight.shape[-1] // m.n_heads, device=dev)


def stack():   # ESMCHiddenStatesModel.__call__ with last_hidden_only, minus the host readback
    x = m.embed(tok_d)
    for b in m.blocks:
        x = b(x, cos, sin, mask_d, kv_d)
    y = ttnn.layer_norm(x, weight=m.norm_weight, epsilon=1e-5, compute_kernel_config=m.compute_kernel_config)
    ttnn.deallocate(x)
    return y


eager = lambda: model(tokens, last_hidden_only=True)[-1, 0]
for _ in range(2):
    ref = eager()
    ttnn.deallocate(stack())
ttnn.synchronize_device(dev)
tid, out_d = capture_trace(dev, stack)


def traced():
    ttnn.copy_host_to_device_tensor(host_ids(), tok_d)
    E._lm_mask_into(mask, lens, mask_d)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    return torch.Tensor(ttnn.to_torch(out_d)).float()[0, :Lm]


got = traced()
diff = (got - ref).abs().max().item()
rel = ((got - ref).norm() / ref.norm()).item()
rows = []
for r in range(a.rounds):
    for arm, fn in (("eager", eager), ("trace", traced)):
        ts = []
        for _ in range(a.repeat):
            t = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t) * 1e3)
        rows.append({"round": r, "arm": arm, "median_ms": round(statistics.median(ts), 2), "all": [round(x, 2) for x in ts]})
        print(json.dumps(rows[-1]), flush=True)
med = {arm: statistics.median(x["median_ms"] for x in rows if x["arm"] == arm) for arm in ("eager", "trace")}
res = {"model": "esmc-6b", "fast": a.fast, "L": Lm, "Lb": Lb, "max_abs_diff": diff, "rel_l2": rel, "median_ms": med,
       "rows": rows}
Path(a.out).write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("fast", "max_abs_diff", "rel_l2", "median_ms")}))
ttnn.release_trace(dev, tid)
sys.exit(0)
