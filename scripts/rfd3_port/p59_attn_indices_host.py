"""p59: host-side screen of the per-step attn_indices chain at the page fixture shape.

CPU only, no device. Question: how much of the 51.9 + 19.8 ms/step that P3.7 bounded at
54.3 ms/step end-to-end is recoverable by (a) thread count and (b) fusing cdist + mask +
topk into row blocks so the [L,L] fp32 distance matrix never round-trips through DRAM.
Bit-exactness of every arm is checked with torch.equal against the shipped chain.
"""
import json, os, sys, time
import torch

L = 6051
K = 128
NSEQ = 2
REPS = 5

torch.manual_seed(0)
x = torch.randn(1, L, 3, dtype=torch.float32) * 12.0

# A mask with the same density as the shipped one: sequence neighbours within NSEQ
# tokens of each other are excluded from the distance topk (they are already in seq_idx).
tok = torch.arange(L) // 9          # ~9 atoms per token, 685 tokens ~ 6051 atoms
allowed = (tok[:, None] - tok[None, :]).abs() <= NSEQ
mask = allowed.contiguous()
rows = torch.arange(L).unsqueeze(0).expand(L, L)
seq_idx = torch.where(mask, rows, torch.tensor(L)).topk(K, dim=1, largest=False, sorted=True).values
inf = torch.tensor(float("inf"))


def shipped(x):
    D = torch.cdist(x, x, p=2)
    D = D.masked_fill_(mask, inf)
    fill = torch.topk(D, K, dim=-1, largest=False).indices.flip(dims=[-1])
    idx = torch.where((seq_idx == L).expand_as(fill), fill, seq_idx.expand_as(fill))
    return torch.sort(idx.long(), dim=-1)[0]


def blocked(x, R):
    out = torch.empty(1, L, K, dtype=torch.long)
    for r0 in range(0, L, R):
        r1 = min(r0 + R, L)
        D = torch.cdist(x[:, r0:r1], x, p=2)
        D = D.masked_fill_(mask[r0:r1], inf)
        fill = torch.topk(D, K, dim=-1, largest=False).indices.flip(dims=[-1])
        s = seq_idx[r0:r1].unsqueeze(0)
        out[:, r0:r1] = torch.where((s == L).expand_as(fill), fill, s.expand_as(fill))
    return torch.sort(out, dim=-1)[0]


def timeit(fn, reps=REPS):
    fn()                       # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], min(ts), max(ts)


res = {"L": L, "K": K, "cpu_count": os.cpu_count(), "arms": []}
ref = None
for nthr in (int(sys.argv[1]) if len(sys.argv) > 1 else 8,):
    torch.set_num_threads(nthr)
    med, lo, hi = timeit(lambda: shipped(x.clone()))
    ref = shipped(x.clone())
    res["arms"].append(dict(arm="shipped", threads=nthr, ms=med, min=lo, max=hi, exact=True))
    print(f"shipped      thr={nthr:2d}  {med:8.2f} ms  [{lo:.2f}, {hi:.2f}]", flush=True)
    for R in (256, 512, 1024, 2048):
        med, lo, hi = timeit(lambda: blocked(x.clone(), R))
        ok = torch.equal(blocked(x.clone(), R), ref)
        res["arms"].append(dict(arm=f"blocked{R}", threads=nthr, ms=med, min=lo, max=hi, exact=bool(ok)))
        print(f"blocked R={R:5d} thr={nthr:2d}  {med:8.2f} ms  [{lo:.2f}, {hi:.2f}]  exact={ok}", flush=True)
print(json.dumps(res))
