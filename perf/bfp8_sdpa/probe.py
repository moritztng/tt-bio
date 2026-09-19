#!/usr/bin/env python3
"""KERNEL-PATH for the two SDPA refusals: does our fused kernel SERVE a bfp8 operand?

Both gates are ours and both sit on machinery that sizes every CB from `tile_bytes(dtype)` per
operand. So the question is not whether the refusal can be deleted, it is whether the kernel the
refusal was hiding actually runs at bfp8 or falls through to something else. A fallback is the
failure mode, not a cost, so this reports SERVED vs DECLINED off the kernel own counters
(`triatt_sdpa.STATS` / `REJECTS` / `FUSE_REJECTS`) and never off the wall clock: `sdpa` returns
`None` when it declines, so a served call is proof our program ran.

Every arm is also scored against a float64 reference of the SAME logical operands, because an
imprecise transform and a wrong one are different things and only the second is a hard stop.
"""
from __future__ import annotations
import argparse, json, statistics, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "c12_orchestrator" / "relayed" / "c12_kblock"))
import clk  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio import triatt_sdpa as TS  # noqa: E402
from tt_bio import mm_generic as MG  # noqa: E402
from tt_bio import sdpa_generic as SG  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "probe.json"))
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--head-dim", type=int, default=32)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--acc-batch", type=int, default=4)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--mhz", type=int, default=1350)
A = ap.parse_args()

dev = ttnn.open_device(device_id=0)
nodes = clk.nodes_open_by_this_process()
grid = dev.compute_with_storage_grid_size()
cores = grid.x * grid.y
DRAM = ttnn.DRAM_MEMORY_CONFIG
B8, B16 = ttnn.bfloat8_b, ttnn.bfloat16
S, H, D, Bt = A.seq, A.heads, A.head_dim, A.batch
SCALE = D ** -0.5

res = {"host": "qb1", "seq": S, "heads": H, "head_dim": D, "batch": Bt, "grid": [grid.y, grid.x],
       "cores": cores, "mhz_requested": A.mhz, "reps": A.reps,
       "tile_bytes": {"bf16": MG.tile_bytes(B16), "bfp8": MG.tile_bytes(B8)}, "arms": {}}

held = clk.force(A.mhz, nodes)
for _ in range(200):
    if all(clk.aiclk(n) >= A.mhz - 5 for n in held):
        break
    time.sleep(0.05)
sampler = clk.Sampler(held[0])

# ---------------------------------------------------------------- host-only: what the mask dtype
# buys on REACH. The persistent mask CB is `k_num_chunks * Sq_chunk_t * Sk_chunk_t` tiles and it is
# the term that caps the shipped q split at 1024. bfp8 tiles are 1088 B against 2048, so the same
# CB budget holds 1.88x the tiles. Pure, no device.
reach = {}
for label, dt in (("bf16", B16), ("bfp8", B8)):
    per_len = {}
    for s in range(512, 2593, 32):
        per_len[s] = len(TS.fused_pairs(s, H, D, cores, dt))
    reach[label] = {"lengths_served": sum(1 for v in per_len.values() if v),
                    "lengths_total": len(per_len),
                    "pairs_at_512": per_len.get(512), "pairs_at_1024": per_len.get(1024),
                    "first_len_with_no_pair": next((s for s, v in sorted(per_len.items()) if not v),
                                                   None),
                    "served_above_1024": sorted(s for s, v in per_len.items() if v and s > 1024)}
res["reach"] = reach


def mk(shape, dt):
    h = torch.randn(*shape, dtype=torch.float32)
    return h, ttnn.from_torch(h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt, memory_config=DRAM)


def ref_sdpa(hq, hk, hv, hb):
    """float64 reference of the same logical operands. hb is [1,H,S,S], broadcast over batch."""
    q, k, v = (x.to(torch.float64) for x in (hq, hk, hv))
    b = hb.to(torch.float64)
    sc = (q @ k.transpose(-1, -2)) * SCALE + b
    return torch.softmax(sc, dim=-1) @ v


def rel_rms(got, ref):
    got = got.to(torch.float64)
    return float(torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref))


def bench(fn):
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(A.reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        if o is not None:
            ttnn.deallocate(o)
    return ts


def record(name, served_delta, declined_delta, out, ts, rr, extra=None):
    d = {"served": served_delta, "declined": declined_delta,
         "kernel_ran": bool(served_delta) and out is not None,
         "rejects_now": {str(k): v for k, v in TS.REJECTS.items()},
         "fuse_rejects_now": dict(TS.FUSE_REJECTS)}
    if out is not None:
        d["out_dtype"] = str(out.dtype)
    if ts:
        d.update(ms_med=statistics.median(ts), ms_min=min(ts), ms=ts)
    if rr is not None:
        d["rel_rms_vs_fp64"] = rr
    if extra:
        d.update(extra)
    res["arms"][name] = d
    return d


pairs = TS.fused_pairs(S, H, D, cores, B16)
if not pairs:
    sys.exit("no (q_chunk, k_chunk) pair at seq=%d on %d cores" % (S, cores))
QC, KC = pairs[0]
res["pair_used"] = [QC, KC]

# ---------------------------------------------------------------- arm 1-4: `sdpa`, the gate at :338
_orig_ok = TS.fast_dtypes_ok


def mixed_ok(*dts):
    """Uniformity relaxed, membership kept. `is_uniform_dataformat` is a compile-time HINT the
    kernel takes either way, so this asks the device whether a bfp8 q/k/v with a bf16 mask is
    legal instead of assuming it from the flag name."""
    return set(dts) <= MG.FAST_DTYPES


ARMS = [("bf16", B16, B16, _orig_ok), ("bfp8", B8, B8, _orig_ok),
        ("bfp8_qkv_bf16_mask", B8, B16, mixed_ok), ("bf16_aa", B16, B16, _orig_ok)]
for name, qdt, bdt, ok in ARMS:
    TS.fast_dtypes_ok = ok
    hq, q = mk([Bt, H, S, D], qdt)
    hk, k = mk([Bt, H, S, D], qdt)
    hv, v = mk([Bt, H, S, D], qdt)
    hb, bias = mk([1, H, S, S], bdt)
    b0 = (TS.STATS[0], TS.STATS[1])
    try:
        out = TS.sdpa(q, k, v, bias, SCALE, QC, KC)
    except Exception as e:  # noqa: BLE001
        record(name, 0, 0, None, None, None, {"error": "%s: %s" % (type(e).__name__, str(e)[:400])})
        TS.fast_dtypes_ok = _orig_ok
        for x in (q, k, v, bias):
            ttnn.deallocate(x)
        continue
    sd, dd = TS.STATS[0] - b0[0], TS.STATS[1] - b0[1]
    ts = bench(lambda: TS.sdpa(q, k, v, bias, SCALE, QC, KC)) if out is not None else None
    rr = None
    if out is not None:
        n = min(A.acc_batch, Bt)
        got = ttnn.to_torch(out)[:n]
        rr = rel_rms(got, ref_sdpa(hq[:n], hk[:n], hv[:n], hb))
    record(name, sd, dd, out, ts, rr)
    if out is not None:
        ttnn.deallocate(out)
    for x in (q, k, v, bias):
        ttnn.deallocate(x)
    TS.fast_dtypes_ok = _orig_ok

# ---------------------------------------------------------------- arm 5-6: `sdpa_fused_qkv`, :481
C = H * D
for name, dt in (("fuseqkv_bf16", B16), ("fuseqkv_bfp8", B8)):
    hx, x = mk([Bt, S, C], dt)
    hw, w = mk([C, 3 * C], dt)
    hb, bias = mk([1, H, S, S], dt)
    b0 = dict(TS.FUSE_REJECTS)
    try:
        out = TS.sdpa_fused_qkv(x, w, bias, SCALE, H, D, S, KC, force=True)
    except Exception as e:  # noqa: BLE001
        record(name, 0, 0, None, None, None, {"error": "%s: %s" % (type(e).__name__, str(e)[:400])})
        for t_ in (x, w, bias):
            ttnn.deallocate(t_)
        continue
    new = {k: v - b0.get(k, 0) for k, v in TS.FUSE_REJECTS.items() if v - b0.get(k, 0)}
    ts = bench(lambda: TS.sdpa_fused_qkv(x, w, bias, SCALE, H, D, S, KC, force=True)) \
        if out is not None else None
    rr = None
    if out is not None:
        n = min(A.acc_batch, Bt)
        qkv = (hx[:n].to(torch.float64) @ hw.to(torch.float64))
        hq, hk, hv = (qkv[..., i * C:(i + 1) * C].reshape(n, S, H, D).permute(0, 2, 1, 3)
                      for i in range(3))
        got = ttnn.to_torch(out)[:n]
        rr = rel_rms(got, ref_sdpa(hq, hk, hv, hb))
    record(name, 1 if out is not None else 0, sum(new.values()), out, ts, rr,
           {"new_fuse_rejects": new})
    if out is not None:
        ttnn.deallocate(out)
    for t_ in (x, w, bias):
        ttnn.deallocate(t_)

res["clock"] = sampler.stop()
a = res["arms"]
if a.get("bf16", {}).get("ms_med") and a.get("bf16_aa", {}).get("ms_med"):
    res["aa_floor_pct"] = abs(a["bf16_aa"]["ms_med"] - a["bf16"]["ms_med"]) / a["bf16"]["ms_med"] * 100
for lo, hi in (("bfp8", "bf16"), ("bfp8_qkv_bf16_mask", "bf16"), ("fuseqkv_bfp8", "fuseqkv_bf16")):
    if a.get(lo, {}).get("ms_med") and a.get(hi, {}).get("ms_med"):
        res.setdefault("ratios", {})[lo] = a[hi]["ms_med"] / a[lo]["ms_med"]
Path(A.out).write_text(json.dumps(res, indent=2))
print(json.dumps({"reach": res["reach"], "pair_used": res["pair_used"],
                  "clock": res["clock"], "aa_floor_pct": res.get("aa_floor_pct"),
                  "ratios": res.get("ratios"),
                  "arms": {k: {kk: vv for kk, vv in v.items()
                               if kk not in ("ms", "rejects_now", "fuse_rejects_now")}
                           for k, v in a.items()}}, indent=2))
ttnn.close_device(dev)
