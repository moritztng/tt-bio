"""Can a gather replace the dense attention-bias scatter? No -- it is 8x worse.

`bench_bias_scatter.py` shows `ttnn.scatter` into the dense (1,4,L,L) bias is
7.3x off bandwidth with no knob to fix it. The natural restructure is to invert
the op: precompute a dense slot map `m[i,j] = position of j in idx[i], else K`
once per step (it depends only on the neighbour indices, which all 9 atom-block
invocations of a step share), then per block gather the padded pair bias through
it, with slot K holding -1e4. Pure data movement, so bit-exact by construction.

Measured at L=3359: gather 38.9 ms against the shipped scatter's 4.7 ms. It also
needs an index tensor the size of its output (181 MB uint32 for a 90 MB bf16
result -- the index does not broadcast across heads), so even a perfect kernel
would move 1.5x the bytes. Direction closed; kept so nobody re-derives it.
"""
import sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ttnn
L,H,K = 3359,4,128
d = ttnn.open_device(device_id=0)
def up(t,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT): return ttnn.from_torch(t,layout=layout,device=d,dtype=dtype)
def timed(label, fn, reps=5):
    try:
        for _ in range(2): o=fn()
        ttnn.synchronize_device(d)
    except Exception as e:
        print(f"  {label:<46s} unsupported: {type(e).__name__}: {str(e)[:90]}"); return None
    ts=[]
    for _ in range(reps):
        t0=time.perf_counter_ns(); o=fn(); ttnn.synchronize_device(d); ts.append((time.perf_counter_ns()-t0)/1e6)
    ts.sort(); print(f"  {label:<46s} {ts[len(ts)//2]:8.3f} ms"); return o

g=torch.Generator().manual_seed(0)
idx = torch.stack([torch.sort(torch.randperm(L,generator=g)[:K])[0] for _ in range(L)])  # [L,K]
# dense slot map: m[i,j] = position of j in idx[i], else K (sentinel -> -1e4 slot)
m = torch.full((L,L), K, dtype=torch.int32)
rows = torch.arange(L).unsqueeze(1).expand(L,K)
m[rows.reshape(-1), idx.reshape(-1)] = torch.arange(K,dtype=torch.int32).repeat(L)
padded = torch.randn(1,H,L,K+1); padded[...,K] = -1e4
pad_t = up(padded)
m4 = up(m.view(1,1,L,L).expand(1,H,L,L).contiguous(), dtype=ttnn.uint32)
m1 = up(m.view(1,1,L,L), dtype=ttnn.uint32)
pad1 = up(padded[:,:1])
print(f"L={L} index (1,4,L,L) uint32 = {H*L*L*4/1e6:.0f} MB, output bf16 = {H*L*L*2/1e6:.0f} MB")
out = timed("gather(pad(1,4,L,129), dim=3, idx(1,4,L,L))", lambda: ttnn.gather(pad_t,3,m4))
timed("gather H=1 (1,1,L,129) idx(1,1,L,L)", lambda: ttnn.gather(pad1,3,m1))
dense_bf = up(torch.full((1,H,L,L),-1e4)); src = up(torch.randn(1,H,L,K)); i4=up(idx.view(1,1,L,K).expand(1,H,L,K).contiguous().to(torch.int32),dtype=ttnn.uint32)
timed("scatter [SHIPPED baseline]", lambda: ttnn.scatter(dense_bf,3,i4,src))
timed("ttnn.add on the 90MB dense [bandwidth ref]", lambda: ttnn.add(dense_bf,dense_bf))
if out is not None:
    # correctness: gather form must equal the scatter form exactly
    sc = ttnn.to_torch(ttnn.scatter(up(torch.full((1,H,L,L),-1e4)),3,i4,up(padded[...,:K].contiguous())))
    ga = ttnn.to_torch(ttnn.gather(up(padded),3,m4))
    print(f"  gather == scatter bit-exact: {torch.equal(sc,ga)} maxabs={(sc-ga).abs().max().item():.3e}")
ttnn.close_device(d)
