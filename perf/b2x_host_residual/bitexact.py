#!/usr/bin/env python3
"""Is the collapsed form of each candidate host lever BIT-EXACT against the shipped form?

Small tensors, one thread. Answers three questions before any of them is worth building:

  ln_split   `F.layer_norm(x, (C,), w, b)`  vs  `F.layer_norm(x, (C,)) * w + b`
             If exact, the 24 bias projections in DiffusionConditioning can share ONE set of
             layer-norm statistics over the 134 MB `z` instead of recomputing them 24 times.
  ln_fused   ... vs one matmul with the affine folded into the Linear weight.
             Mathematically equal, almost certainly not bit-exact; measured, not assumed.
  onehot_mm  `one_hot(idx, K).float() @ W.T`  vs  `W[idx]` (a gather).
             If exact, RelativePositionEncoder stops materialising an N*N*139 float tensor.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.manual_seed(0)
torch.set_grad_enabled(False)


def report(tag, a, b):
    same = torch.equal(a, b)
    d = (a.float() - b.float()).abs()
    print(f"{tag:24s} bit-exact {str(same):5s}  max|d| {d.max().item():.3e}  "
          f"mean|d| {d.mean().item():.3e}  rel {(d.max()/a.float().abs().max()).item():.2e}")
    return same


C, H = 128, 8
for shape in [(1, 64, 64, C), (1, 200, 200, C)]:
    x = torch.randn(*shape)
    w = torch.randn(C)
    b = torch.randn(C)
    W = torch.randn(H, C)          # nn.Linear(C, H, bias=False).weight

    ship = F.linear(F.layer_norm(x, (C,), w, b), W)

    # (1) share the statistics, keep the per-layer affine and matmul
    zn = F.layer_norm(x, (C,))
    split = F.linear(zn * w + b, W)

    # (2) fold the affine into the Linear weight: one matmul for all layers
    Wf = W * w                      # [H, C]
    bf = W @ b                      # [H]
    fused = F.linear(zn, Wf) + bf

    print(f"--- shape {tuple(shape)} ---")
    report("ln_split", ship, split)
    report("ln_fused", ship, fused)

# (3) one-hot matmul vs gather
for N, K in ((64, 66), (200, 139)):
    idx = torch.randint(0, K, (1, N, N))
    Wl = torch.randn(C, K)
    ship = F.linear(F.one_hot(idx, K).float(), Wl)
    gather = Wl.t()[idx]
    print(f"--- one_hot N={N} K={K} ---")
    report("onehot_mm", ship, gather)

# (4) the real rel_pos shape: four concatenated one-hot blocks through one Linear vs
#     the sum of four gathers, in the SAME channel order the cat uses.
N = 200
sizes = [66, 66, 1, 6]
Wl = torch.randn(C, sum(sizes))
idxs = [torch.randint(0, s, (1, N, N)) for s in sizes]
blocks = [F.one_hot(i, s).float() for i, s in zip(idxs, sizes)]
ship = F.linear(torch.cat(blocks, dim=-1), Wl)
off = 0
acc = None
for i, s in zip(idxs, sizes):
    g = Wl[:, off:off + s].t()[i]
    acc = g if acc is None else acc + g
    off += s
print(f"--- rel_pos-shaped, 4 blocks, N={N} ---")
report("relpos_gathersum", ship, acc)

# (5) the EXACT rel_pos layout: block 3 is not a one-hot, it is a single 0/1 channel, so the
#     gather form has to add its row conditionally. Signed zero is the only tolerated difference.
N = 256
sizes = [66, 66, 1, 6]
Wl = torch.randn(C, sum(sizes))
d_res = torch.randint(0, 66, (1, N, N))
d_tok = torch.randint(0, 66, (1, N, N))
same_ent = torch.rand(1, N, N) > 0.5
d_ch = torch.randint(0, 6, (1, N, N))
ship = F.linear(torch.cat([F.one_hot(d_res, 66).float(), F.one_hot(d_tok, 66).float(),
                           same_ent.unsqueeze(-1).float(), F.one_hot(d_ch, 6).float()],
                          dim=-1), Wl)
acc = Wl[:, 0:66].t()[d_res]
acc = acc + Wl[:, 66:132].t()[d_tok]
acc = acc + same_ent.unsqueeze(-1).float() * Wl[:, 132]
acc = acc + Wl[:, 133:139].t()[d_ch]
print(f"--- rel_pos exact layout, N={N}, conditional channel ---")
report("relpos_conditional", ship, acc)
print("  signbit differences:", int((torch.signbit(ship) != torch.signbit(acc)).sum()))
