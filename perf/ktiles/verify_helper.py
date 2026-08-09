#!/usr/bin/env python3
"""Verify tt_bio.tenstorrent.batched_matmul on the real 298 aa diffusion attention shapes:
which classes it takes, that the config it picks is legal, that the result is bit-exact against
today's plain ttnn.matmul, and what it is worth per call. Synced both sides, warm, median of 5.
"""
import json, statistics, sys, time
import torch, ttnn
from tt_bio import tenstorrent as T

F32, BF16 = ttnn.float32, ttnn.bfloat16
CASES = [  # site (pre-splice line), a, b, dtype, calls/block, qb2 ms/fold
    ("protenix.py:417 atom QK^T pv2", (75, 4, 32, 128), (75, 4, 128, 32), F32, 6, 633.0),
    ("protenix.py:414 atom AV   pv2", (75, 4, 32, 32), (75, 4, 32, 128), F32, 6, 387.8),
    ("tenstorrent.py:1656 DiT AV pv2", (1, 16, 320, 320), (1, 16, 320, 64), F32, 24, 627.4),
    ("tenstorrent.py:1650 DiT QK pv2", (1, 16, 320, 64), (1, 16, 64, 320), F32, 24, 223.9),
    ("protenix.py:417 atom QK^T odde", (75, 4, 32, 128), (75, 4, 128, 32), BF16, 6, 592.6),
    ("protenix.py:414 atom AV   odde", (75, 4, 32, 32), (75, 4, 32, 128), BF16, 6, 357.4),
    ("tenstorrent.py:1678 DiT AV odde", (1, 16, 608, 608), (1, 16, 608, 64), BF16, 24, 1325.7),
    ("tenstorrent.py:1670 DiT QK odde", (1, 16, 608, 64), (1, 16, 64, 608), BF16, 24, 369.6),
]
REPS, ITERS, STEPS = 20, 5, 200


def timeit(fn, dev):
    fn(); ttnn.synchronize_device(dev)
    out = []
    for _ in range(ITERS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(REPS):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / REPS * 1e3)
    return statistics.median(out)


def main(out_path):
    dev = ttnn.open_device(device_id=0)
    T._configure_active_compute_grid(dev)
    print(f"grid {T.COMPUTE_GRID_MAIN} = {T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]} cores")
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    rows, saved = [], {}
    for label, sa, sb, dt, per_block, qb2 in CASES:
        torch.manual_seed(0)
        ta, tb = torch.randn(sa), torch.randn(sb)
        a = ttnn.from_torch(ta, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = ttnn.from_torch(tb, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        batch = 1
        for d in sa[:-2]:
            batch *= d
        cfg = T._batched_matmul_config(batch, sa[-2] // 32, sa[-1] // 32, sb[-1] // 32,
                                      4 if dt == F32 else 2)
        base = timeit(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=ckc)), dev)
        new = timeit(lambda: ttnn.deallocate(T.batched_matmul(a, b, compute_kernel_config=ckc)), dev)
        ref = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc))
        got = ttnn.to_torch(T.batched_matmul(a, b, compute_kernel_config=ckc))
        exact = torch.equal(ref, got)
        gold = ta.float() @ tb.float()
        model = "protenix-v2" if "pv2" in label else "opendde"
        ms_fold = (base - new) * per_block * STEPS
        saved[model] = saved.get(model, 0.0) + ms_fold
        desc = "declined (falls back to ttnn.matmul)" if cfg is None else (
            f"per_core_M={cfg.per_core_M} per_core_N={cfg.per_core_N} "
            f"in0_block_w={cfg.in0_block_w} sub={cfg.out_subblock_h}x{cfg.out_subblock_w}")
        print(f"{label}\n   {desc}\n   ttnn.matmul {base:8.4f} ms | batched_matmul {new:8.4f} ms | "
              f"{base / new:6.2f}x | torch.equal={exact} | {ms_fold:+8.1f} ms/fold "
              f"({per_block}/block x {STEPS} steps)", flush=True)
        rows.append(dict(label=label, model=model, a=list(sa), b=list(sb), dtype=str(dt),
                         config=None if cfg is None else desc, calls_per_block=per_block,
                         steps=STEPS, base_ms=base, new_ms=new, ratio=base / new,
                         torch_equal=exact, ms_per_fold_saved=ms_fold,
                         rmsd_base_vs_torch=(gold - ref.float()).pow(2).mean().sqrt().item(),
                         rmsd_new_vs_torch=(gold - got.float()).pow(2).mean().sqrt().item(),
                         qb2_ms_per_fold=qb2))
        ttnn.deallocate(a); ttnn.deallocate(b)
    print("\nop-isolated projection, ms/fold saved: " +
          ", ".join(f"{m} {v:.0f}" for m, v in saved.items()))
    ttnn.close_device(dev)
    json.dump({"grid": list(T.COMPUTE_GRID_MAIN), "rows": rows, "op_isolated_ms_per_fold": saved},
              open(out_path, "w"), indent=1)
    print(f"wrote {out_path}")
    bad = [r["label"] for r in rows if r["config"] and not r["torch_equal"]]
    assert not bad, f"not bit-exact: {bad}"
    applied = [r["label"] for r in rows if r["config"]]
    assert len(applied) == 6, f"expected 6 applied classes, got {applied}"
    print("PASS: 6 classes applied, all bit-exact")


main(sys.argv[1])
