#!/usr/bin/env python3
"""Is a bfp8 destination on our own matmul transcription a ROUNDING of the right answer?

The standing hard stop is a transform that is WRONG rather than imprecise, verified against a
float64 reference and never against another approximation. `TT_BIO_TRIATT_B8` changes exactly one
thing in the arithmetic: the format the packer writes at the end of `generic_minimal_matmul`. The
accumulator is untouched (`interm_fmt` is fp32 under `fp32_dest_acc_en`), so the claim to check is
that the narrowed destination differs from the float64 answer by bfp8's own spacing and nothing
more -- a mis-sized output page or a corrupted shared-exponent block would show up here as an
error orders of magnitude larger, not as a rounding.

Three arms against ONE float64 reference computed from the device tensors read back, so the
reference is the same numbers the device saw:

  our_b16   generic_minimal_matmul, bf16 destination   -- the shipped path, the error floor
  our_b8    generic_minimal_matmul, bfp8 destination   -- the change
  ttnn_b8   ttnn.linear(dtype=bfloat8_b)               -- upstream's own bfp8 pack, the cross-check
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
from tt_bio import mm_generic as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "qkv_ref.json"))
ap.add_argument("--m", type=int, default=512)
ap.add_argument("--k", type=int, default=128)
ap.add_argument("--n", type=int, default=512)
A = ap.parse_args()

MGD = ensure_p300_mesh_descriptor()
dev = ttnn.open_device(device_id=0)
from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, _mm_block_for, _MM_DEFAULT  # noqa: E402

B16, B8 = ttnn.bfloat16, ttnn.bfloat8_b
DRAM = ttnn.DRAM_MEMORY_CONFIG
ckc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)

xt = torch.randn(A.m, A.k, dtype=torch.float32)
wt = torch.randn(A.k, A.n, dtype=torch.float32) * 0.05
x = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=B16, memory_config=DRAM)
w = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=B16, memory_config=DRAM)

# the reference is built from what the DEVICE holds, not from the fp32 draw, so the bf16 rounding
# of the operands is not charged to any arm
ref = (ttnn.to_torch(x).double() @ ttnn.to_torch(w).double())

blk = _mm_block_for(w) or _MM_DEFAULT
cfg = (blk, tuple(COMPUTE_GRID_MAIN))
res = {"m": A.m, "k": A.k, "n": A.n, "mgd": MGD, "block_entry": blk is not _MM_DEFAULT,
       "grid": list(COMPUTE_GRID_MAIN), "arms": {}}


def score(name, t):
    d = (t.double() - ref)
    res["arms"][name] = {
        "max_abs": float(d.abs().max()),
        "rel_rmse": float((d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())),
        "dtype": str(t.dtype), "shape": list(t.shape),
    }


for name, dt in (("our_b16", B16), ("our_b8", B8)):
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([A.m, A.n]), dt, ttnn.TILE_LAYOUT, dev, DRAM)
    try:
        G.generic_minimal_matmul(dev, x, w, out, cfg, G.ckc_args(ckc), (), None)
        ttnn.synchronize_device(dev)
        score(name, ttnn.to_torch(out))
    except Exception as e:                                                   # noqa: BLE001
        res["arms"][name] = {"error": "%s: %s" % (type(e).__name__, str(e)[:300])}
    ttnn.deallocate(out)

for name, dt in (("ttnn_b16", B16), ("ttnn_b8", B8)):
    try:
        y = ttnn.linear(x, w, compute_kernel_config=ckc, dtype=dt, memory_config=DRAM)
        ttnn.synchronize_device(dev)
        score(name, ttnn.to_torch(y))
        ttnn.deallocate(y)
    except Exception as e:                                                   # noqa: BLE001
        res["arms"][name] = {"error": "%s: %s" % (type(e).__name__, str(e)[:300])}

b16 = res["arms"].get("our_b16", {}).get("rel_rmse")
b8 = res["arms"].get("our_b8", {}).get("rel_rmse")
t8 = res["arms"].get("ttnn_b8", {}).get("rel_rmse")
if b16 and b8:
    res["b8_over_b16"] = b8 / b16
if t8 and b8:
    res["ours_over_upstream_b8"] = b8 / t8
Path(A.out).write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
ttnn.close_device(dev)
