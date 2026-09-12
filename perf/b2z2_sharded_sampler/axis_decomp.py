"""Which axis of a token-DiT matmul actually carries its cost?

The row shard measured 1.0411x because 56.3 percent of the token DiT does not depend on the
token count (token_layer_shard.py). That leaves one claim in the state doc unmeasured: that the
CHANNEL axis is the one that does carry it, so a Megatron-style column shard would divide what a
row shard could not. This measures it directly on the four linear shapes the census names inside
one token layer, instead of asserting it.

For each shape, four arms, all on one chip, traced, so nothing is a dispatch artifact:
  full      M x K x N as the fold runs it
  half-M    the ROW shard: half the token rows, same weights
  half-N    the COLUMN shard: same rows, half the output columns (half the weight)
  half-K    the down-projection direction: half the contraction depth (half the weight)

A cost that is weight-bound halves on N and on K and barely moves on M. A cost that is
row-work-bound does the opposite.
"""

import json, os, time
import torch

REPS = int(os.environ.get("AD_REPS", "15"))
ITERS = int(os.environ.get("AD_ITERS", "24"))
OUT = os.environ.get("AD_OUT", "perf/b2z2_sharded_sampler/axis_decomp.json")

import ttnn  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402

dev = tt.get_device()
KC = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


# (name, M, K, N, calls per token layer) straight from the census table.
SHAPES = [
    ("qkv 768x3072", 512, 768, 3072, 1),
    ("proj 768x768", 512, 768, 768, 5),
    ("up 768x1536", 512, 768, 1536, 3),
    ("down 1536x768", 512, 1536, 768, 1),
]


def timed(M, K, N):
    a = ttnn.from_torch(torch.randn(1, M, K).to(torch.bfloat16).float(), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev)
    w = ttnn.from_torch(torch.randn(K, N).to(torch.bfloat16).float(), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev)

    def chain():
        outs = []
        for _ in range(ITERS):
            outs.append(ttnn.linear(a, w, compute_kernel_config=KC, core_grid=tt.CORE_GRID_MAIN))
        return outs

    for _ in range(2):
        for o in chain():
            ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    chain()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    per = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        per.append(time.perf_counter() - t0)
    per.sort()
    ttnn.release_trace(dev, tid)
    ttnn.deallocate(a)
    ttnn.deallocate(w)
    return per[len(per) // 2] / ITERS * 1e6  # us per call


res = {"iters": ITERS, "reps": REPS, "shapes": []}
for name, M, K, N, ncalls in SHAPES:
    arms = {"full": (M, K, N), "half_M": (M // 2, K, N),
            "half_N": (M, K, N // 2), "half_K": (M, K // 2, N)}
    row = {"name": name, "M": M, "K": K, "N": N, "calls_per_layer": ncalls, "us": {}}
    for arm, (m, k, n) in arms.items():
        row["us"][arm] = timed(m, k, n)
    f = row["us"]["full"]
    row["ratio"] = {a: f / v for a, v in row["us"].items()}
    res["shapes"].append(row)
    log("%-16s full %7.2f us | half-M %7.2f (%.3fx) | half-N %7.2f (%.3fx) | half-K %7.2f (%.3fx)"
        % (name, f, row["us"]["half_M"], row["ratio"]["half_M"],
           row["us"]["half_N"], row["ratio"]["half_N"],
           row["us"]["half_K"], row["ratio"]["half_K"]))

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1)
log("wrote %s" % OUT)
