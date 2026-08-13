"""E2/L3 screen: triangle-attention SDPA at the bg_R3 production shape, k_chunk sweep + K2.

Shipped today at seq 576: q_chunk=576 (wide-q serves), k_chunk=256. 256 does NOT divide 576, so
padded_Sk = 768: the kernel reads a 576x768 mask grid instead of 576x576, and `use_padded_mask`
also makes K2 (persistent mask) decline on 100% of calls.

Arms, all at the real shape (B=576, H=4, Sq=Sk=576, D=32), mask (1,4,576,576) bf16 DRAM TILE:
  stock ttnn SDPA at k_chunk in {256 (shipped), 192, 288, 576}
  K2 fused at the same k_chunks
Median of 7 after 2 warm. Accuracy vs the shipped arm reported as PCC + maxdiff.
"""
import sys, json, time
sys.path.insert(0, "/home/ttuser/.coworker/wt/boltzgen-optimize-on-fixture")
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import triatt_sdpa as PM

dev = ttnn.open_device(device_id=0)
B, H, S, D = 576, 4, 576, 32
torch.manual_seed(0)
tq = torch.randn(B, H, S, D, dtype=torch.bfloat16) * 0.5
tk = torch.randn(B, H, S, D, dtype=torch.bfloat16) * 0.5
tv = torch.randn(B, H, S, D, dtype=torch.bfloat16) * 0.5
tm = torch.randn(1, H, S, S, dtype=torch.bfloat16) * 0.3
to_dev = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
q, k, v, m = to_dev(tq), to_dev(tk), to_dev(tv), to_dev(tm)
scale = 32 ** -0.5
N = 7

def timeit(fn):
    for _ in range(2):
        o = fn(); ttnn.deallocate(o)
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        last = ttnn.to_torch(o).float()
        ttnn.deallocate(o)
    ts.sort()
    return ts[N // 2] * 1e3, last

def pcc(a, b):
    a, b = a.flatten(), b.flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])

res = {}
ref = None
for kc in (256, 192, 288, 576):
    def stock(kc=kc):
        return ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=m, is_causal=False, scale=scale,
            program_config=T._sdpa_program_config(q_chunk_size=576, k_chunk_size=kc))
    try:
        ms, out = timeit(stock)
        if kc == 256:
            ref = out
        res[f"stock_k{kc}"] = {"ms": round(ms, 4), "pcc_vs_shipped": pcc(out, ref),
                               "maxdiff": float((out - ref).abs().max())}
    except Exception as e:
        res[f"stock_k{kc}"] = {"error": str(e)[:200]}
    print(f"stock k_chunk={kc}", res[f"stock_k{kc}"], flush=True)

for kc in (192, 288, 576, 256):
    PM.REJECTS.clear(); PM.STATS[0] = PM.STATS[1] = 0
    def fused(kc=kc):
        o = PM.sdpa(q, k, v, m, scale, 576, kc)
        if o is None:
            raise RuntimeError(f"declined: {dict(PM.REJECTS)}")
        return o
    try:
        ms, out = timeit(fused)
        res[f"K2_k{kc}"] = {"ms": round(ms, 4), "pcc_vs_shipped": pcc(out, ref),
                            "maxdiff": float((out - ref).abs().max()),
                            "served": PM.STATS[0], "declined": PM.STATS[1]}
    except Exception as e:
        res[f"K2_k{kc}"] = {"error": str(e)[:300]}
    print(f"K2 k_chunk={kc}", res[f"K2_k{kc}"], flush=True)

print("RESULT " + json.dumps(res))
ttnn.close_device(dev)
