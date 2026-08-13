"""Two API questions lever A's implementation turns on. Untimed, so co-tenancy is irrelevant.

  1. `ttnn.embedding` from a 4225 x 128 table -- the fused (bin_d, bin_s) pair table.
  2. `ttnn.multiply([B,I,I,128], [B,I,I,1])` -- the per-row rms scale, broadcast on the last dim.

Both checked for VALUE, not speed: the gather against a torch index, the multiply against a torch
broadcast.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ttnn  # noqa: E402

B, I, C, NB = 2, 96, 128, 65


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


dev = ttnn.open_device(device_id=0)
g = torch.Generator().manual_seed(7)

# --- 1. gather from the fused pair table ------------------------------------
table = torch.randn(NB * NB, C, generator=g) * 0.1
bd = torch.randint(0, NB, (B, I, I), generator=g)
bs = torch.randint(0, NB, (B, I, I), generator=g)
idx = (bd * NB + bs).to(torch.int32)
ref = table[idx.reshape(-1)].reshape(B, I, I, C)

t_dev = ttnn.from_torch(table, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
i_dev = ttnn.from_torch(idx.reshape(1, -1), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                        dtype=ttnn.uint32)
try:
    out = ttnn.embedding(i_dev, t_dev, layout=ttnn.ROW_MAJOR_LAYOUT,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.to_layout(ttnn.reshape(out, (B, I, I, C)), ttnn.TILE_LAYOUT)
    got = ttnn.to_torch(out).float()
    print(f"[1] pair-table gather   PCC {pcc(ref, got):.10f}   "
          f"max|d| {float((ref - got.double()).abs().max()):.3e}   "
          f"exact_vs_bf16 {torch.equal(got, ref.bfloat16().float())}")
except Exception as e:
    print(f"[1] pair-table gather   FAILED: {type(e).__name__}: {e}")

# --- 2. broadcast multiply on the last dim ----------------------------------
x = torch.randn(B, I, I, C, generator=g)
r = torch.rand(B, I, I, 1, generator=g) + 0.5
x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
for tag, rr in (("[B,I,I,1]", r), ("[B,I,I,32] replicated", r.expand(B, I, I, 32).contiguous())):
    r_dev = ttnn.from_torch(rr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    try:
        got = ttnn.to_torch(ttnn.multiply(x_dev, r_dev)).float()
        print(f"[2] multiply {tag:22s} shape {tuple(got.shape)}  "
              f"PCC {pcc(x * r, got):.10f}")
    except Exception as e:
        print(f"[2] multiply {tag:22s} FAILED: {type(e).__name__}: {str(e)[:120]}")
    ttnn.deallocate(r_dev)

# --- 3. the fp32 residual question: does add mix fp32 const with bf16 gather? ---
c32 = ttnn.from_torch(torch.randn(B, I, I, C, generator=g), dtype=ttnn.float32,
                      layout=ttnn.TILE_LAYOUT, device=dev)
try:
    s = ttnn.add(c32, x_dev)
    print(f"[3] add fp32 + bf16     out dtype {s.dtype}")
except Exception as e:
    print(f"[3] add fp32 + bf16     FAILED: {type(e).__name__}: {str(e)[:120]}")

ttnn.close_device(dev)
