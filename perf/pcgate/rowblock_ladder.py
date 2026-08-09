#!/usr/bin/env python3
"""E8 -- what the row-block search would pick if its budget were right, and what that is worth.

W5's `_l1_resident_row_block` walks the row block downwards and returns the first one its L1 test
admits. That test is the only place left in tt_bio where a program-config budget PICKS a config
rather than vetoing a fixed one, so it is the only place E6's defect can cost or buy time.

Three budgets give three answers at the real 298 aa tri-attention qkv shape:
  device constant (what W5 ships)                 -> 34 rows
  live largest_contiguous_bytes_free_per_bank     -> 33 rows
  the true requirement                            -> measured here, by building every r and seeing
                                                     which ones the program factory actually accepts

Timing is per row block on an idle device, which is the state the gate probe measured at these call
sites (free per bank was the full 1461760 B at every sample). Whole-tensor time is the block time
times the number of full blocks plus one clamped tail block, the way TriangleAttention runs it.
"""
import argparse, json, time
from pathlib import Path

import torch
import ttnn
import tt_bio.tenstorrent as T

REPS = 20


def build_cfg(r, cols, k_tiles, n_tiles, num_cores, gx, gy):
    col_tiles = -(-cols // 32)
    m_tiles = r * col_tiles
    per_core_M = next(
        (p for p in range(max(1, -(-m_tiles // num_cores)), m_tiles + 1) if m_tiles % p == 0), 0)
    if not per_core_M or -(-m_tiles // per_core_M) > num_cores:
        return None, None, None
    osw = max((w for w in range(min(4, n_tiles), 0, -1) if n_tiles % w == 0), default=1)
    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=k_tiles,
        out_subblock_h=1, out_subblock_w=osw, out_block_h=per_core_M, out_block_w=n_tiles,
        per_core_M=per_core_M, per_core_N=n_tiles, fuse_batch=True, fused_activation=None,
        mcast_in0=False)
    return cfg, per_core_M, -(-m_tiles // per_core_M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=298)
    ap.add_argument("--c-in", type=int, default=256)
    ap.add_argument("--c-out", type=int, default=768)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = T.get_device()
    gx, gy = T.COMPUTE_GRID_MAIN
    num_cores = gx * gy
    device_l1 = int(ttnn.get_max_worker_l1_unreserved_size())
    free0 = int(ttnn.get_memory_view(dev, ttnn.BufferType.L1)
                .largest_contiguous_bytes_free_per_bank)
    k_tiles, n_tiles = -(-a.c_in // 32), -(-a.c_out // 32)

    w = ttnn.from_torch(torch.randn(a.c_in, a.c_out, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)

    rows = []
    for r in [17, 20, 33, 34, 40, 50, 60, 68, 75, 85, 100, 120, 149, 298]:
        if r > a.cols:
            continue
        cfg, pcm, blocks = build_cfg(r, a.cols, k_tiles, n_tiles, num_cores, gx, gy)
        rec = dict(rows=r, per_core_M=pcm, cores_used=blocks,
                   out_bytes_per_bank=-(-(r * -(-a.cols // 32) * n_tiles) // num_cores) * 2048)
        if cfg is None:
            rec.update(status="no legal per_core_M")
            rows.append(rec)
            continue
        x = ttnn.from_torch(torch.randn(r, a.cols, a.c_in, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        try:
            y = ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                            memory_config=ttnn.L1_MEMORY_CONFIG, program_config=cfg)
            ttnn.synchronize_device(dev)
            ttnn.deallocate(y)
            t0 = time.perf_counter()
            for _ in range(REPS):
                y = ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                memory_config=ttnn.L1_MEMORY_CONFIG, program_config=cfg)
                ttnn.deallocate(y)
            ttnn.synchronize_device(dev)
            ms = (time.perf_counter() - t0) / REPS * 1e3
            full, tail = divmod(a.cols, r)
            rec.update(status="ok", block_ms=round(ms, 4),
                       whole_ms=round(ms * (full + (1 if tail else 0)), 4),
                       blocks_per_tensor=full + (1 if tail else 0))
        except Exception as e:
            rec.update(status="throw", error=str(e).strip().splitlines()[0][:200])
        ttnn.deallocate(x)
        rows.append(rec)

    ok = [r for r in rows if r["status"] == "ok"]
    out = dict(cols=a.cols, c_in=a.c_in, c_out=a.c_out, grid=[gx, gy],
               device_l1_unreserved=device_l1, free_per_bank_idle=free0,
               largest_r_that_builds=max((r["rows"] for r in ok), default=0),
               best_r=min(ok, key=lambda r: r["whole_ms"])["rows"] if ok else None,
               rows=rows)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    T.cleanup()


if __name__ == "__main__":
    main()
