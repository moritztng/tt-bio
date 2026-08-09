#!/usr/bin/env python3
"""Can SDPA replace the RFD3 atom-block score chain? Chunk sweep + parity.

chain.py established that `rfd3/model.py:1346` is at 102.8% of the DRAM write roof and that
the 8.54 ms chain around it exists only to move a 130 MB score matrix through DRAM five more
times. SDPA keeps that matrix in L1. Two things have to hold before it can ship:

  1. a chunking that fits L1 and gets near the only roof SDPA can have -- reading the mask
     (130 MB bf16 / 388.4 GB/s = 335 us on qb1 card 2);
  2. numerics. The shipped path is bf16 matmul -> fp32 upcast -> scale -> +bias -> fp32 softmax
     -> bf16. The bias is itself an exact upcast of a bf16 tensor (`_sparse_bias_f32` typecasts
     a bf16 scatter), so a bf16 SDPA mask reads identical VALUES; what differs is the softmax
     accumulation order (flash rescaling vs one pass) and where the fp32 lives.

Parity is measured against the shipped chain ON DEVICE (not against torch), because that chain
is what ships. A realistic mask is used: -1e4 everywhere except `n_keys` neighbours per row,
which is what `_sparse_bias_f32` builds, and which makes the softmax denominator a sum over
128 terms rather than 4032.
"""
import argparse, json, statistics as st, time

import torch
import ttnn

from tt_bio.tenstorrent import get_device

DRAM = ttnn.DRAM_MEMORY_CONFIG

ap = argparse.ArgumentParser()
ap.add_argument("--mt", type=int, default=126)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--head-dim", type=int, default=32)
ap.add_argument("--n-keys", type=int, default=128)
ap.add_argument("--out", default=None)
a = ap.parse_args()

dev = get_device()
dg = dev.compute_with_storage_grid_size()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=False, packer_l1_acc=False)

H, D = a.heads, a.head_dim
M = N = a.mt * 32
SC = float(D) ** -0.5
res = {"shape": {"heads": H, "M": M, "N": N, "head_dim": D, "n_keys": a.n_keys}}
print(f"H={H} M={M} N={N} D={D} n_keys={a.n_keys}", flush=True)


def timed(fn, warm=3, pipe=3, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)


# ------------------------------------------------------------------ operands
torch.manual_seed(0)
tq = torch.randn(1, H, M, D) * 0.3
tk = torch.randn(1, H, M, D) * 0.3
tv = torch.randn(1, H, M, D) * 0.3
# realistic sparse bias: -1e4 background, a real pair bias at n_keys sorted neighbours/row
tb = torch.full((1, H, M, N), -1e4)
idx = torch.stack([torch.randperm(N)[: a.n_keys].sort().values for _ in range(M)])
local = torch.randn(1, H, M, a.n_keys) * 0.5
tb.scatter_(3, idx.view(1, 1, M, a.n_keys).expand(1, H, M, a.n_keys), local)

q = ttnn.from_torch(tq, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
k = ttnn.from_torch(tk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
v = ttnn.from_torch(tv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
bias16 = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
bias32 = ttnn.typecast(bias16, ttnn.float32, memory_config=DRAM)


def shipped():
    """Exactly what rfd3/model.py:1332-1358 issues, in the same order."""
    kT = ttnn.permute(k, (0, 1, 3, 2))
    s = ttnn.matmul(q, kT, compute_kernel_config=ckc)
    s = ttnn.typecast(s, ttnn.float32, memory_config=s.memory_config())
    s = ttnn.multiply(s, SC)
    s = ttnn.add(s, bias32)
    s = ttnn.softmax(s, dim=-1)
    s = ttnn.typecast(s, ttnn.bfloat16, memory_config=s.memory_config())
    return ttnn.matmul(s, v, compute_kernel_config=ckc, dtype=ttnn.bfloat16)


sdpa = ttnn.transformer.scaled_dot_product_attention
ref_t = ttnn.to_torch(shipped()).float()
t_ship = timed(lambda: ttnn.deallocate(shipped()), warm=2, pipe=2, reps=5)
print(f"shipped chain {t_ship*1e6:9.1f} us", flush=True)
res["shipped_us"] = round(t_ship * 1e6, 1)

MASK_READ_FLOOR_US = H * M * N * 2 / 388.4e9 * 1e6
print(f"mask-read floor {MASK_READ_FLOOR_US:.1f} us (130 MB bf16 / 388.4 GB/s)", flush=True)


def parity(x):
    d = (x - ref_t).abs()
    denom = ref_t.abs().max().item()
    xf, rf = x.flatten().double(), ref_t.flatten().double()
    pcc = torch.corrcoef(torch.stack([xf, rf]))[0, 1].item()
    return {"max_abs": round(d.max().item(), 6),
            "mean_abs": round(d.mean().item(), 8),
            "rel_to_peak": round(d.max().item() / denom, 6),
            "pcc": round(pcc, 9),
            "bit_exact": bool(torch.equal(x, ref_t))}


rows = []
cfgs = [None]
for qc in (32, 64, 128, 256):
    for kc in (128, 256, 512, 1024):
        cfgs.append((qc, kc))
for c in cfgs:
    if c is None:
        label, pc = "default", None
    else:
        label = f"q{c[0]}_k{c[1]}"
        try:
            pc = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=dg, q_chunk_size=c[0], k_chunk_size=c[1],
                exp_approx_mode=False)
        except Exception as e:
            print(f"  {label:12s} cfg ERR {str(e)[:70]}", flush=True)
            continue
    kw = {"attn_mask": bias16, "is_causal": False, "scale": SC, "compute_kernel_config": ckc}
    if pc is not None:
        kw["program_config"] = pc
    try:
        out = sdpa(q, k, v, **kw)
        p = parity(ttnn.to_torch(out).float())
        ttnn.deallocate(out)
        t = timed(lambda: ttnn.deallocate(sdpa(q, k, v, **kw)), warm=2, pipe=2, reps=5)
    except Exception as e:
        print(f"  {label:12s} ERR {str(e)[:90]}", flush=True)
        rows.append({"cfg": label, "err": str(e)[:200]})
        continue
    row = {"cfg": label, "us": round(t * 1e6, 1),
           "speedup": round(t_ship / t, 2),
           "pct_mask_read_roof": round(100 * MASK_READ_FLOOR_US / (t * 1e6), 1), **p}
    rows.append(row)
    print(f"  {label:12s} {t*1e6:8.1f} us  {t_ship/t:5.2f}x  "
          f"{row['pct_mask_read_roof']:5.1f}% of mask-read roof  "
          f"maxabs {p['max_abs']:.5f} pcc {p['pcc']:.9f}", flush=True)

res["sdpa"] = rows
ok = [r for r in rows if "us" in r]
if ok:
    best = min(ok, key=lambda r: r["us"])
    res["best"] = best
    print(f"BEST {best['cfg']} {best['us']} us  {best['speedup']}x  "
          f"{best['pct_mask_read_roof']}% of the mask-read roof", flush=True)

# how much of the shipped chain's own time is the mask/bias handling we would also delete
res["bias_typecast_us"] = round(
    timed(lambda: ttnn.deallocate(ttnn.typecast(bias16, ttnn.float32, memory_config=DRAM)),
          warm=2, pipe=2, reps=5) * 1e6, 1)
print(f"bias bf16->fp32 typecast (deletable with a bf16 mask) {res['bias_typecast_us']} us", flush=True)

if a.out:
    json.dump(res, open(a.out, "w"), indent=2)
    print("wrote " + a.out, flush=True)
