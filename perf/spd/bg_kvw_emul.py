#!/usr/bin/env python3
"""Host-only fp64 proof that atom_kvw (zero-framed stream, superset bias, in-place kv_window read) is the windowed atom
attention `_atom_shift_gather` + SDPA compute, on every real window. No device: python perf/spd/bg_kvw_emul.py"""

import torch
torch.manual_seed(0)
W, A, S, L = 32, 128, 48, 64; Wk = A + 2 * (L - S)
K, H, dh, D = 7, 4, 32, 128; K_real = 5
N = K * W
s = torch.randn(N, D, dtype=torch.float64)
wq, wk, wv = (torch.randn(D, H * dh, dtype=torch.float64) for _ in range(3)); bq = torch.randn(H * dh, dtype=torch.float64)
atom_real = torch.zeros(N); atom_real[:K_real * W - 7] = 1          # ragged real atoms
# today's: keys of window i = atoms [32i-48, 32i+80), zero outside [0, K_real*W) (one-hot), mask -1e9 where not real
def key_ids(i): return torch.arange(32 * i - S, 32 * i - S + A)
z = torch.randn(K, H, W, A, dtype=torch.float64)
ref = torch.zeros(K, H, W, dh, dtype=torch.float64)
for i in range(K):
    ids = key_ids(i); ok = (ids >= 0) & (ids < N)
    xs = torch.zeros(A, D, dtype=torch.float64); xs[ok] = s[ids[ok]]
    real = torch.zeros(A); real[ok] = atom_real[ids[ok]]
    if i >= K_real: real[:] = 0
    m = torch.where(real > 0, 0.0, -1e9).double()
    q = (s[32 * i:32 * i + 32] @ wq + bq).view(W, H, dh).transpose(0, 1)
    k = (xs @ wk).view(A, H, dh).transpose(0, 1); v = (xs @ wv).view(A, H, dh).transpose(0, 1)
    zz = z[i] + m
    ref[i] = torch.softmax((q @ k.transpose(-1, -2) + zz) * dh ** -0.5, -1) @ v
    z[i] = zz                                                      # the bias today's kernel reads carries the mask
# atom_kvw
frame = torch.cat([torch.zeros(L, D, dtype=torch.float64), s, torch.zeros(L, D, dtype=torch.float64)])
qkv = frame @ torch.cat([wq, wk, wv], 1) + torch.cat([bq, torch.zeros(2 * H * dh, dtype=torch.float64)])
Q, Kf, V = (qkv[:, j * H * dh:(j + 1) * H * dh].view(-1, H, dh).transpose(0, 1) for j in range(3))
fill = torch.full((H, K, W, L - S), -1e9, dtype=torch.float64)
sup = torch.cat([fill, z.permute(1, 0, 2, 3), fill], 3)            # [H, K, W, Wk]
out = torch.zeros(H, K, W, dh, dtype=torch.float64)
for h in range(H):
    for i in range(K):
        q = Q[h, L + 32 * i:L + 32 * i + W]; k = Kf[h, 32 * i:32 * i + Wk]; v = V[h, 32 * i:32 * i + Wk]
        out[h, i] = torch.softmax((q @ k.T + sup[h, i]) * dh ** -0.5, -1) @ v
out = out.permute(1, 0, 2, 3)
d = (out - ref)[:K_real].abs().max().item()
print("max |kvw - ref| over real windows:", d)
assert d < 1e-12
