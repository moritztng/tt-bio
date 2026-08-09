"""E6: the chunked-Transition matmul ladder, measured under the L1 residency the fold actually has.

Every config is timed with the swiglu's own L1-resident activations already allocated, because that
residency is what the program config has to fit around -- a ladder run on an idle device picks
configs that cannot be built inside the block.
"""
import json
import statistics
import sys

import torch
import ttnn

sys.path.insert(0, "/home/ttuser/.coworker/wt/perfwar-chunked-transition-cb/perf/chunked_transition")
from cb_model import cb_need, subblock  # noqa: E402

BF16 = 2


def divisors(n, cap):
    return [d for d in range(1, min(n, cap) + 1) if n % d == 0]


def time_op(dev, fn, reps=5):
    for _ in range(2):
        y = fn()
        ttnn.synchronize_device(dev)
        ttnn.deallocate(y)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = ttnn.__dict__.get("_t", None)
        import time
        t0 = time.perf_counter()
        y = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(y)
    return statistics.median(ts)


def main():
    dev = ttnn.open_device(device_id=0)
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    g = dev.compute_with_storage_grid_size()
    gx, gy = g.x, g.y
    banks = ttnn.get_memory_view(dev, ttnn.BufferType.L1).num_banks
    core_grid = ttnn.CoreGrid(x=gx, y=gy)
    print(f"grid {gx}x{gy} banks {banks}")

    def mk(shape, l1=False):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16), device=dev,
                               layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                               memory_config=ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG)

    # (name, h_rows, W, c, hidden, leg) -- leg 'up' is fc1/fc2 (c->hidden, L1 out),
    # 'down' is fc3 (hidden->c, DRAM out). Residency mirrors swiglu at that point.
    CASES = [
        ("protenix-v2 up   ", 30, 320, 256, 1024, "up"),
        ("protenix-v2 down ", 30, 320, 256, 1024, "down"),
        ("protenix-v2 up28 ", 28, 320, 256, 1024, "up"),
        ("protenix-v2 down28", 28, 320, 256, 1024, "down"),
        ("opendde up       ", 30, 320, 384, 1536, "up"),
        ("opendde down     ", 30, 320, 384, 1536, "down"),
    ]
    out = {}
    for name, h, W, c, hid, leg in CASES:
        mt = h * (W // 32)
        kt, nt = (c // 32, hid // 32) if leg == "up" else (hid // 32, c // 32)
        out_l1 = leg == "up"
        # residency: at fc2 x_norm[c] + x_1[hid] are live; at fc3 x[hid] is live
        res = ([mk((1, h, W, c), l1=True), mk((1, h, W, hid), l1=True)] if leg == "up"
               else [mk((1, h, W, hid), l1=True)])
        x = mk((1, h, W, c) if leg == "up" else (1, h, W, hid))
        w = mk((c, hid) if leg == "up" else (hid, c))
        free = ttnn.get_memory_view(dev, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank
        out_bytes = (-(-(mt * nt) // banks) * 1024 * BF16) if out_l1 else 0
        budget = free - out_bytes
        mcfg = dict(memory_config=ttnn.L1_MEMORY_CONFIG if out_l1 else ttnn.DRAM_MEMORY_CONFIG)

        base_t = time_op(dev, lambda: ttnn.linear(x, w, compute_kernel_config=ckc,
                                                  dtype=ttnn.bfloat16, core_grid=core_grid, **mcfg))
        ref = ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                          core_grid=core_grid, **mcfg)
        ref_t = ttnn.to_torch(ref)
        ttnn.deallocate(ref)
        per_core_M, per_core_N = -(-mt // gy), -(-nt // gx)
        rows = []
        for bw in divisors(kt, 8)[::-1]:
            for obh in divisors(per_core_M, per_core_M)[::-1]:
                obw = per_core_N
                need = cb_need(obh, obw, bw, kt)
                if need > budget:
                    continue
                sh, sw = subblock(obh, obw)
                pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=bw,
                    out_subblock_h=sh, out_subblock_w=sw, out_block_h=obh, out_block_w=obw,
                    per_core_M=per_core_M, per_core_N=per_core_N, transpose_mcast=False,
                    fused_activation=None, fuse_batch=True)
                try:
                    t = time_op(dev, lambda: ttnn.linear(x, w, compute_kernel_config=ckc,
                                                         dtype=ttnn.bfloat16, program_config=pc, **mcfg))
                    y = ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                    program_config=pc, **mcfg)
                    yt = ttnn.to_torch(y)
                    ttnn.deallocate(y)
                    eq = bool(torch.equal(yt, ref_t))
                    rows.append((bw, obh, obw, sh, sw, need, t, base_t / t, eq))
                except Exception as e:
                    rows.append((bw, obh, obw, sh, sw, need, None, None, str(e)[:60]))
        rows.sort(key=lambda r: (r[6] is None, r[6]))
        print(f"\n{name}  mt={mt} kt={kt} nt={nt} pcM={per_core_M} pcN={per_core_N} "
              f"out={'L1' if out_l1 else 'DRAM'}  free={free} out_bytes={out_bytes} budget={budget}")
        print(f"   BASE core_grid {base_t:.4f} ms")
        for bw, obh, obw, sh, sw, need, t, gain, eq in rows[:8]:
            print(f"   bw={bw:<2} obh={obh:<3} obw={obw} sub={sh}x{sw} need={need:>7} "
                  f"{'%.4f' % t if t else 'FAIL':>8} ms  gain={('%.3fx' % gain) if gain else '-':>7}  "
                  f"torch.equal={eq}")
        out[name.strip()] = dict(mt=mt, kt=kt, nt=nt, base_ms=base_t, budget=budget,
                                 rows=[dict(bw=r[0], obh=r[1], obw=r[2], sub=[r[3], r[4]], need=r[5],
                                            ms=r[6], gain=r[7], equal=r[8]) for r in rows])
        for r in res:
            ttnn.deallocate(r)
        ttnn.deallocate(x)
        ttnn.deallocate(w)
    with open("/home/ttuser/.coworker/wt/perfwar-chunked-transition-cb/perf/chunked_transition/ladder_card3.json", "w") as f:
        json.dump(out, f, indent=1)
    ttnn.close_device(dev)


if __name__ == "__main__":
    sys.exit(main())
