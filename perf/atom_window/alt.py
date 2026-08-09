#!/usr/bin/env python3
"""Candidate fixes for the serial-batch atom-window matmuls. qb2 card 1.

Baseline: the exact sub-graph of AtomTransformer._attention at 298 aa shapes
(nb=75, H=4, nq=32, nk=128, dh=32).

Candidates:
  pc     ttnn.MatmulMultiCoreReuseProgramConfig -- does an explicit program config put the
         batch on the core grid instead of a serial loop?
  chunk  batch split into G chunks, one matmul per chunk with a grid-sized program config
  sdpa   ttnn.transformer.scaled_dot_product_attention -- parallelises over (b, nqh, s_q) and
         fuses scale+mask+softmax+A@V, so the [75,4,32,128] logits never reach DRAM
"""
import argparse, itertools, json, statistics as st, time
import torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

NB, H, NQ, NK, DH = 75, 4, 32, 128, 32
dev = get_device()
CKC = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
GX, GY = CORE_GRID_MAIN.x, CORE_GRID_MAIN.y


def timed(fn, warm=3, reps=8, trials=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(trials):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) / reps)
    return st.median(o)


def tt(x, dt):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)


def probe_pc(dt, out):
    """Does MatmulMultiCoreReuseProgramConfig spread the batch across cores?"""
    print(f"--- program-config probe, QK^T shape [nb,1,32,32]@[nb,1,32,128], grid {GX}x{GY}", flush=True)
    rows = []
    for nb in (1, 4, 16, 64, 100, 110):
        g = torch.Generator().manual_seed(0)
        a = tt(torch.randn(nb, 1, NQ, DH, generator=g), dt)
        b = tt(torch.randn(nb, 1, DH, NK, generator=g), dt)
        base = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=CKC)))
        got = {}
        for gx, gy in ((GX, GY), (10, 10), (8, 8)):
            if nb > gx * gy:
                continue
            try:
                pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=(gx, gy), in0_block_w=1,
                    out_subblock_h=1, out_subblock_w=4, per_core_M=1, per_core_N=4)
                r = ttnn.matmul(a, b, program_config=pc, compute_kernel_config=CKC)
                ok = torch.equal(ttnn.to_torch(r), ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=CKC)))
                ttnn.deallocate(r)
                s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, program_config=pc,
                                                              compute_kernel_config=CKC)))
                got[f"{gx}x{gy}"] = {"us": round(s * 1e6, 2), "bit_exact": ok}
                print(f"  nb={nb:<4} base {base*1e6:8.2f} us   pc {gx}x{gy} {s*1e6:8.2f} us  "
                      f"({base/s:5.2f}x, bit-exact={ok})", flush=True)
            except Exception as e:
                got[f"{gx}x{gy}"] = {"err": str(e)[:120]}
                print(f"  nb={nb:<4} base {base*1e6:8.2f} us   pc {gx}x{gy} ERR {str(e)[:110]}", flush=True)
        rows.append({"nb": nb, "base_us": round(base * 1e6, 2), "pc": got})
        ttnn.deallocate(a); ttnn.deallocate(b)
    out["pc_probe"] = rows


def sdpa(dt, name, out):
    print(f"--- SDPA {name}: q[75,4,32,32] k/v[75,4,128,32] mask[75,4,32,128]", flush=True)
    g = torch.Generator().manual_seed(0)
    tq = torch.randn(NB, H, NQ, DH, generator=g)
    tk = torch.randn(NB, H, NK, DH, generator=g)
    tv = torch.randn(NB, H, NK, DH, generator=g)
    tz = torch.randn(NB, H, NQ, NK, generator=g)
    q, k, v, z = tt(tq, dt), tt(tk, dt), tt(tv, dt), tt(tz, dt)

    def baseline():
        sc = ttnn.matmul(q, ttnn.permute(k, (0, 1, 3, 2)), compute_kernel_config=CKC)
        sc = ttnn.multiply(sc, DH ** -0.5)
        sc = ttnn.add(sc, z)
        return ttnn.matmul(ttnn.softmax(sc, dim=-1), v, compute_kernel_config=CKC)

    bs = timed(baseline)
    ref = ttnn.to_torch(baseline()).float()
    print(f"  baseline chain      {bs*1e6:9.2f} us", flush=True)
    rows = []
    cfgs = [("auto", None)]
    for gx, gy in ((GX, GY), (8, 8)):
        for qc, kc in ((32, 128), (32, 32)):
            try:
                cfgs.append((f"{gx}x{gy}/q{qc}k{kc}", ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(gx, gy), q_chunk_size=qc, k_chunk_size=kc,
                    exp_approx_mode=False)))
            except Exception as e:
                print("  cfg build err", str(e)[:100], flush=True)
    for lbl, pc in cfgs:
        try:
            kw = {"attn_mask": z, "is_causal": False, "scale": DH ** -0.5,
                  "compute_kernel_config": CKC}
            if pc is not None:
                kw["program_config"] = pc
            r = ttnn.transformer.scaled_dot_product_attention(q, k, v, **kw)
            got = ttnn.to_torch(r).float()
            ttnn.deallocate(r)
            s = timed(lambda: ttnn.deallocate(
                ttnn.transformer.scaled_dot_product_attention(q, k, v, **kw)))
            d = (got - ref)
            pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
            rows.append({"cfg": lbl, "us": round(s * 1e6, 2), "speedup": round(bs / s, 2),
                         "pcc": round(pcc, 7), "max_abs": round(d.abs().max().item(), 6),
                         "bit_exact": bool(torch.equal(got, ref))})
            print(f"  sdpa {lbl:16s} {s*1e6:9.2f} us  ({bs/s:5.2f}x)  pcc={pcc:.7f} "
                  f"maxabs={d.abs().max().item():.3e}", flush=True)
        except Exception as e:
            rows.append({"cfg": lbl, "err": str(e)[:160]})
            print(f"  sdpa {lbl:16s} ERR {str(e)[:140]}", flush=True)
    out[f"sdpa_{name}"] = {"baseline_us": round(bs * 1e6, 2), "runs": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="pc,sdpa")
    ap.add_argument("--out", default="perf/atom_window/alt_card1.json")
    a = ap.parse_args()
    out = {"card": "qb2-card1", "grid": f"{GX}x{GY}"}
    if "pc" in a.modes:
        probe_pc(ttnn.bfloat16, out)
    if "sdpa" in a.modes:
        sdpa(ttnn.bfloat16, "bf16", out)
    json.dump(out, open(a.out, "w"), indent=2)
    print("wrote", a.out, flush=True)
