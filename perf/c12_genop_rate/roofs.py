#!/usr/bin/env python3
"""The two roofs this row's headroom is measured against, on qb2 card 3, in ONE session.

`c12-genericop-rate` reads its per-site rates in situ out of `c12-profiled-fold`'s ops reports
(`insitu_sites.py`), so no arm here re-measures a site. What is missing is the denominator: the
brief compares generic_op's 26.02 TFLOP/s to a 115.685 TFLOP/s cube taken in a different session,
and `roofline-roof-must-be-measured-not-asserted` says that is not a roof. Every one of the six
sites moves 67-403 MB of DRAM per call, so the roof that can bind them is the STREAMING roof, and
a dense cube cannot tell you anything about an op at 0-254 FLOP/byte.

Arms, interleaved rep by rep in one process on one card, each timed region fenced by
`ttnn.synchronize_device` on both sides, 3 warm reps discarded, statistic = min over reps with the
median printed beside it:

  bw_add8192     the campaign's own streaming-roof arm, verbatim from
                 `perf/roof_shape/shape_roofs.py:177` -- a starved 8192^2 bf16 add, 2R + 1W =
                 402.65 MB, the same footprint scale as the sites themselves.
  bw_add8192_aa  its A/A floor.
  bw_clone       1R + 1W at 100.66 MB each, because reblock_permute_gated is 1R + 1W and the
                 achievable rate is a function of the read:write mix, not just of the bytes.
  bw_clone_aa    its A/A floor.
  cube4096       the dense 4096^3 at HiFi4, this session's compute denominator.
  cube4096_aa    its A/A floor.
  cube4096_lofi  the same cube at LoFi, the highest FPU rate this part can reach at all.
  cube2048       the known-answer control: b2z measured 0.1537 ms and c12-profiled-fold
                 reproduced 0.1520 ms on this card. A session that does not land there is not
                 measuring this part.

Clock is forced and sampled by `clk.py` in a separate process for the whole session.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import ttnn  # noqa: E402

HERE = Path(__file__).resolve().parent
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _f(*dims):
    """2 * M * K * N for a matmul, the campaign's FLOP convention."""
    m, k, n = dims
    return 2.0 * m * k * n


def timed(fn, device, reps, warm):
    out = []
    for i in range(reps + warm):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        if isinstance(r, ttnn.Tensor):
            ttnn.deallocate(r)
        if i >= warm:
            out.append((t1 - t0) * 1e3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--out", type=Path, default=HERE / "roofs_qb2c3.json")
    a = ap.parse_args()

    import torch
    from tt_bio import tenstorrent as T

    device = T.get_device()
    g = device.compute_with_storage_grid_size()
    cores = g.x * g.y
    print("grid %dx%d = %d cores" % (g.x, g.y, cores))

    def dev(t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=DRAM)

    torch.manual_seed(0)
    big = dev(torch.randn(8192, 8192) * 0.05)
    bigb = dev(torch.randn(8192, 8192) * 0.05)
    clo = dev(torch.randn(8192, 6144) * 0.05)
    c4 = dev(torch.randn(4096, 4096) * 0.05)
    c4b = dev(torch.randn(4096, 4096) * 0.05)
    c2 = dev(torch.randn(2048, 2048) * 0.05)
    c2b = dev(torch.randn(2048, 2048) * 0.05)

    # same kernel config the campaign's own cube arm uses (perf/roof_shape/shape_roofs.py:63)
    kcls = (ttnn.types.WormholeComputeKernelConfig if device.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    kc_hifi4 = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                    fp32_dest_acc_en=True, packer_l1_acc=True)
    kc_lofi = kcls(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                   fp32_dest_acc_en=False, packer_l1_acc=True)
    # scripts/profiling/roofline_bh.py:32 -- the config the campaign's published roofs use, and
    # the one the b2z 2048^3 instrument proof ran at. HiFi4 + fp32_dest_acc_en=True is a different
    # arm and lands 1.43x off the proof, which is a config difference, not a sick part.
    def kc_fid(fid):
        return kcls(math_fidelity=fid, math_approx_mode=False,
                    fp32_dest_acc_en=False, packer_l1_acc=False)
    kc_r_hifi4 = kc_fid(ttnn.MathFidelity.HiFi4)
    kc_r_hifi2 = kc_fid(ttnn.MathFidelity.HiFi2)
    kc_r_lofi = kc_fid(ttnn.MathFidelity.LoFi)

    BW_ADD = 3 * 8192 * 8192 * 2          # 2R + 1W, bytes
    BW_CLO = 2 * 8192 * 6144 * 2          # 1R + 1W, bytes

    # name -> (callable, metric_kind, amount)
    arms = [
        ("bw_add8192", lambda: ttnn.add(big, bigb, memory_config=DRAM), "bytes", BW_ADD),
        ("bw_add8192_aa", lambda: ttnn.add(big, bigb, memory_config=DRAM), "bytes", BW_ADD),
        ("bw_clone", lambda: ttnn.clone(clo, memory_config=DRAM), "bytes", BW_CLO),
        ("bw_clone_aa", lambda: ttnn.clone(clo, memory_config=DRAM), "bytes", BW_CLO),
        ("cube4096", lambda: ttnn.matmul(c4, c4b, compute_kernel_config=kc_hifi4,
                                         memory_config=DRAM), "flop", _f(4096, 4096, 4096)),
        ("cube4096_aa", lambda: ttnn.matmul(c4, c4b, compute_kernel_config=kc_hifi4,
                                            memory_config=DRAM), "flop", _f(4096, 4096, 4096)),
        ("cube4096_lofi", lambda: ttnn.matmul(c4, c4b, compute_kernel_config=kc_lofi,
                                              memory_config=DRAM), "flop", _f(4096, 4096, 4096)),
        ("cube2048", lambda: ttnn.matmul(c2, c2b, compute_kernel_config=kc_hifi4,
                                         memory_config=DRAM), "flop", _f(2048, 2048, 2048)),
        # the known-answer control at the config the proof used: no compute_kernel_config at all
        ("cube2048_dflt", lambda: ttnn.matmul(c2, c2b, memory_config=DRAM),
         "flop", _f(2048, 2048, 2048)),
        ("cube2048_dflt_aa", lambda: ttnn.matmul(c2, c2b, memory_config=DRAM),
         "flop", _f(2048, 2048, 2048)),
        # the three fidelities on the published-roof config, so each site is scored against the
        # FPU roof of the fidelity it actually runs at (sdpa and reblock_back run HiFi2)
        ("cube4096_r_hifi4", lambda: ttnn.matmul(c4, c4b, compute_kernel_config=kc_r_hifi4,
                                                 memory_config=DRAM), "flop", _f(4096, 4096, 4096)),
        ("cube4096_r_hifi2", lambda: ttnn.matmul(c4, c4b, compute_kernel_config=kc_r_hifi2,
                                                 memory_config=DRAM), "flop", _f(4096, 4096, 4096)),
        ("cube4096_r_lofi", lambda: ttnn.matmul(c4, c4b, compute_kernel_config=kc_r_lofi,
                                                memory_config=DRAM), "flop", _f(4096, 4096, 4096)),
    ]

    # warm every arm once before any timing, then interleave rep by rep
    for name, fn, _k, _a in arms:
        for _ in range(a.warm):
            ttnn.synchronize_device(device)
            r = fn()
            ttnn.synchronize_device(device)
            if isinstance(r, ttnn.Tensor):
                ttnn.deallocate(r)
    samples = {name: [] for name, _f2, _k, _a in arms}
    for rep in range(a.reps):
        for name, fn, _k, _a in arms:
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            r = fn()
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            if isinstance(r, ttnn.Tensor):
                ttnn.deallocate(r)
            samples[name].append((t1 - t0) * 1e3)

    rows = []
    for name, _fn, kind, amt in arms:
        d = samples[name]
        lo, med = min(d), st.median(d)
        rate = amt / (lo * 1e-3) / 1e9          # GB/s for bytes, GFLOP/s for flop
        rows.append(dict(arm=name, kind=kind, min_ms=lo, median_ms=med,
                         spread_pct=100.0 * (max(d) - lo) / lo, reps=len(d),
                         GBs=rate if kind == "bytes" else None,
                         TFLOPs=rate / 1e3 if kind == "flop" else None))
        print("%-16s min %8.4f ms  med %8.4f ms  spread %5.2f%%  %s"
              % (name, lo, med, 100.0 * (max(d) - lo) / lo,
                 ("%.1f GB/s" % rate) if kind == "bytes" else ("%.2f TFLOP/s" % (rate / 1e3))))

    meta = dict(host=os.uname().nodename, grid=[g.x, g.y], cores=cores,
                card=os.environ.get("TT_VISIBLE_DEVICES"),
                arch=str(device.arch()), reps=a.reps, warm=a.warm,
                loadavg=list(os.getloadavg()))
    json.dump(dict(meta=meta, rows=rows), open(a.out, "w"), indent=1)
    print("loadavg", meta["loadavg"])
    ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
