#!/usr/bin/env python3
"""Job 1+2 of b2z-custom-sdpa: price the triangle-attention SDPA and find its floor.

Job 1 (share) is already MEASURED on the cell and must not be re-run: `subunit_floor_512_qb2c0.json`
(qb2 card 0, p300c, commit 072da10f, 2026-09-11) gives TriangleAttention = 4.391 ms/call at
560 calls/fold, against a PairformerLayer of 41.4152 ms at 280 calls/fold -- two triangle-attention
calls per layer, so 8.782 ms of 41.4152 ms = 21.2 % of the block. `triatt_fused/s6_gate.json` gives
the SDPA call inside it at 2.673 ms, so the SDPA ITSELF is 5.346 ms = 12.9 % of the block and
1.497 s of a 23.7 s fold.

This script answers Job 2, on whatever card it is given: what is the floor of that 2.673 ms? Both
roofs are MEASURED here in the same process on the same card rather than quoted, because a roof
carried across a card type is not a roof (`roofline-roof-must-be-measured-not-asserted`).

PREDICTED, written before the first number existed:

    The op moves 541.0 MB and does 137.44 GFLOP. On Blackhole's published campaign roofs
    (429.9 GB/s, 85.96 TFLOP/s) that is 1.259 ms of bytes and 1.599 ms of arithmetic, so the fused
    kernel at 2.673 ms sits at 60 % of its COMPUTE roof -- the one place in this fold where the
    compute roof binds before the byte roof. If that holds, the remaining headroom on this op is
    at most 1.67x and realistically ~1.3x, the op is 12.9 % of the block, and the brief's own
    repoint rule fires: a perfect kernel here is 1.03x on the fold.

    FALSIFIER: if the measured compute roof at this fidelity is >2x the bandwidth-implied roof,
    or the fused op measures under 45 % of both roofs, there is real headroom and the brief stands.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import triatt_sdpa as TS

S, H, D = 512, 8, 32
SDPA_BYTES = (4 * S * H * S * D + H * S * S) * 2
SDPA_FLOPS = 2 * 2 * S * H * S * S * D
CKC = (ttnn.MathFidelity.HiFi2, True, False, False)


def timed(fn, dev, warm=3, reps=20):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        if r is not None:
            ttnn.deallocate(r)
    return st.median(ts), ts


def interleaved(a, b, dev, reps=20):
    """Paired A/B in one process, the only perf protocol this campaign accepts on a shared host."""
    for _ in range(3):
        a(); b()
    ta, tb = [], []
    for _ in range(reps):
        for fn, acc in ((a, ta), (b, tb)):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            r = fn()
            ttnn.synchronize_device(dev)
            acc.append(time.perf_counter() - t0)
            if r is not None:
                ttnn.deallocate(r)
    return st.median(ta), st.median(tb), ta, tb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/b2z_sdpa_floor/floor_512.json")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    dev = T.get_device()
    grid = tuple(T.COMPUTE_GRID_MAIN)
    res = {"doc": __doc__, "meta": {
        "card": os.environ.get("TT_VISIBLE_DEVICES"), "grid": list(grid),
        "loadavg": os.getloadavg(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "shape": [S, H, S, D], "sdpa_bytes": SDPA_BYTES, "sdpa_flops": SDPA_FLOPS}}
    print(json.dumps(res["meta"]), flush=True)

    def dram(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # --- roof 1: streaming bytes. ttnn.add moves 3N, ttnn.clone 2N; take the better of the two.
    torch.manual_seed(0)
    n = 8192
    nb = n * n * 2
    x, y = dram(torch.randn(n, n).to(torch.bfloat16)), dram(torch.randn(n, n).to(torch.bfloat16))
    bw = {}
    for name, op, moved in (("add", lambda: ttnn.add(x, y), 3), ("clone", lambda: ttnn.clone(x), 2)):
        try:
            m, _ = timed(op, dev, reps=10)
            bw[name] = {"ms": m * 1e3, "gbps": moved * nb / m / 1e9}
        except Exception as exc:  # noqa: BLE001
            bw[name] = {"error": str(exc)[:200]}
    ttnn.deallocate(x); ttnn.deallocate(y)
    bw_gbps = max((d["gbps"] for d in bw.values() if "gbps" in d), default=0.0)
    res["roof_bw"] = {"arms": bw, "gbps": bw_gbps}
    print("BW roof", json.dumps(res["roof_bw"]), flush=True)

    # --- roof 2: arithmetic, at the SAME fidelity the SDPA kernel runs. A roof taken at LoFi is
    # not this op's roof: fidelity is passes through the FPU, and the fused kernel ships HiFi2.
    mm_ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=CKC[0], math_approx_mode=CKC[1], fp32_dest_acc_en=CKC[2],
        packer_l1_acc=False)
    fl = {}
    for m_ in (2048, 4096):
        a = dram(torch.randn(m_, m_).to(torch.bfloat16))
        b = dram(torch.randn(m_, m_).to(torch.bfloat16))
        try:
            med, _ = timed(lambda: ttnn.matmul(a, b, compute_kernel_config=mm_ckc,
                                               core_grid=T.CORE_GRID_MAIN), dev, reps=10)
            fl[m_] = {"ms": med * 1e3, "tflops": 2 * m_ ** 3 / med / 1e12}
        except Exception as exc:  # noqa: BLE001
            fl[m_] = {"error": str(exc)[:200]}
        ttnn.deallocate(a); ttnn.deallocate(b)
    tflops = max((d["tflops"] for d in fl.values() if "tflops" in d), default=0.0)
    res["roof_flops"] = {"arms": fl, "tflops": tflops, "ckc": "HiFi2,approx,no_fp32_acc"}
    print("FLOP roof", json.dumps(res["roof_flops"]), flush=True)

    # --- the op, exactly as the fold issues it, fused against stock.
    q, k, v = (dram(torch.randn(S, H, S, D).to(torch.bfloat16)) for _ in range(3))
    bias = dram(torch.randn(1, H, S, S).to(torch.bfloat16))
    scale = D ** -0.5

    def fused():
        TS._ENABLED = True
        return T._tri_att_sdpa_at(q, k, v, bias, scale)

    def stock():
        TS._ENABLED = False
        return T._tri_att_sdpa_at(q, k, v, bias, scale)

    of, os_ = fused(), stock()
    tf, tt_ = ttnn.to_torch(of).float(), ttnn.to_torch(os_).float()
    res["parity_fused_vs_stock"] = {
        "equal": bool(torch.equal(tf, tt_)),
        "max_abs": float((tf - tt_).abs().max()),
        "rel_rms": float(((tf - tt_) ** 2).mean().sqrt() / (tt_ ** 2).mean().sqrt())}
    ttnn.deallocate(of); ttnn.deallocate(os_)
    print("parity", json.dumps(res["parity_fused_vs_stock"]), flush=True)

    mf, ms, af, as_ = interleaved(fused, stock, dev, reps=args.reps)
    TS._ENABLED = True
    res["op"] = {
        "fused_ms": mf * 1e3, "stock_ms": ms * 1e3, "fused_over_stock": ms / mf,
        "fused_all_ms": [t * 1e3 for t in af], "stock_all_ms": [t * 1e3 for t in as_],
        "fused_served": TS.STATS[0], "fused_declined": TS.STATS[1],
        "rejects": {str(k_): v for k_, v in TS.REJECTS.items()},
        "picks": {str(k_): v for k_, v in T.SDPA_CHUNK_PICKS.items()},
        "routes": dict(T.SDPA_ROUTE_COUNTS)}

    # --- place it
    floor_bytes_ms = SDPA_BYTES / (bw_gbps * 1e9) * 1e3 if bw_gbps else None
    floor_flops_ms = SDPA_FLOPS / (tflops * 1e12) * 1e3 if tflops else None
    floor = max(x_ for x_ in (floor_bytes_ms, floor_flops_ms) if x_)
    res["floor"] = {
        "bytes_ms": floor_bytes_ms, "flops_ms": floor_flops_ms, "binding_ms": floor,
        "binding": "compute" if floor_flops_ms >= floor_bytes_ms else "bandwidth",
        "fused_over_floor": mf * 1e3 / floor,
        "frac_of_bw_roof": floor_bytes_ms / (mf * 1e3),
        "frac_of_flop_roof": floor_flops_ms / (mf * 1e3)}
    print("floor", json.dumps(res["floor"], indent=1), flush=True)

    res["meta"]["loadavg_end"] = os.getloadavg()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
