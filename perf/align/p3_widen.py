#!/usr/bin/env python3
"""p3-align-widen, deliverable 2: is the widen a metadata change, and what does it buy?

Runs against a SOURCE-BUILT tt-metal (not the 0.68 production wheel) carrying one patch:
`infer_dims_for_reshape` accepts a logical shape that grows into padding the tensor already
owns. Ratios only; this is not a campaign absolute and it is not the production wheel.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import ttnn

GRID = (11, 10)


def ckc():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)


def trimul_pc(seq_len_tiles=10):
    gx, gy = GRID
    per_core_M = -(-seq_len_tiles // gy)
    per_core_N = -(-seq_len_tiles // gx)
    in0_block_w = max(d for d in range(min(10, seq_len_tiles), 0, -1) if seq_len_tiles % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=in0_block_w,
        out_subblock_h=1, out_subblock_w=1, out_block_h=per_core_M, out_block_w=per_core_N,
        per_core_M=per_core_M, per_core_N=per_core_N, transpose_mcast=False,
        fused_activation=None, fuse_batch=False)


def timeit(dev, fn, reps=20, warm=3):
    for _ in range(warm):
        o = fn()
        del o
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [fn() for _ in range(reps)]
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / reps
    del outs
    return dt * 1e6


def addr(t):
    for name in ("buffer_address",):
        f = getattr(t, name, None)
        if f is not None:
            try:
                return int(f())
            except Exception:                                    # noqa: BLE001
                pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    L1 = ttnn.L1_MEMORY_CONFIG
    dev = ttnn.open_device(device_id=0)
    out = {}
    try:
        at = torch.randn(1, 32, 298, 298, dtype=torch.float32)
        bt = torch.randn(1, 32, 298, 298, dtype=torch.float32)
        x = ttnn.from_torch(at, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=L1)
        b = ttnn.from_torch(bt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=L1)
        a0 = addr(x)

        # ---- 1. the widen itself -------------------------------------------------------------
        for tag, fn in (("experimental_view", lambda: ttnn.experimental.view(x, (1, 32, 320, 320))),
                        ("reshape", lambda: ttnn.reshape(x, (1, 32, 320, 320)))):
            try:
                y = fn()
                out[tag] = {
                    "ok": True, "logical": list(y.shape), "padded": list(y.padded_shape),
                    "in_addr": a0, "out_addr": addr(y),
                    "same_buffer": addr(y) == a0 and a0 is not None,
                    "us": timeit(dev, fn, reps=100, warm=10),
                }
            except Exception as e:                               # noqa: BLE001
                out[tag] = {"ok": False, "error": str(e)[:400]}
            print(tag, out[tag] if not out[tag].get("ok") else
                  {k: out[tag][k] for k in ("logical", "same_buffer", "us")})

        if not out.get("experimental_view", {}).get("ok"):
            return out

        xw = ttnn.experimental.view(x, (1, 32, 320, 320))
        bw = ttnn.experimental.view(b, (1, 32, 320, 320))

        # ---- 2. what is actually in the padding the widen exposes? ---------------------------
        wt = ttnn.to_torch(xw)
        out["padding_contents_from_torch"] = {
            "tail_rows_max_abs": float(wt[:, :, 298:, :].abs().max()),
            "tail_cols_max_abs": float(wt[:, :, :, 298:].abs().max()),
            "valid_equal": bool(torch.equal(wt[:, :, :298, :298], ttnn.to_torch(x))),
        }
        print("padding after from_torch:", out["padding_contents_from_torch"])

        # ---- 3. does ttnn.permute leave the padding it creates zero? -------------------------
        # this is the amortisation question: if permute zero-fills, one fill upstream serves
        # every contraction in the trimul; if it does not, the fill is per contraction.
        pre = ttnn.from_torch(torch.randn(1, 298, 298, 32), dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
        perm = ttnn.permute(pre, (0, 3, 1, 2), memory_config=L1)
        pw = ttnn.experimental.view(perm, (1, 32, 320, 320))
        pwt = ttnn.to_torch(pw)
        out["permute_padding"] = {
            "logical_after_permute": list(perm.shape),
            "padded_after_permute": list(perm.padded_shape),
            "tail_rows_max_abs": float(pwt[:, :, 298:, :].abs().max()),
            "tail_cols_max_abs": float(pwt[:, :, :, 298:].abs().max()),
        }
        print("permute padding:", out["permute_padding"])

        # same question for the matmul's own output, and for layer_norm
        mm = ttnn.matmul(x, b, compute_kernel_config=ckc(), memory_config=L1,
                         program_config=trimul_pc(), dtype=ttnn.bfloat16)
        mmw = ttnn.to_torch(ttnn.experimental.view(mm, (1, 32, 320, 320)))
        out["matmul_output_padding"] = {
            "tail_rows_max_abs": float(mmw[:, :, 298:, :].abs().max()),
            "tail_cols_max_abs": float(mmw[:, :, :, 298:].abs().max())}
        print("matmul out padding:", out["matmul_output_padding"])

        for t in (pre, perm, mm):
            try:
                ttnn.deallocate(t)
            except Exception:                                    # noqa: BLE001
                pass

        # ---- 4. the payoff: the contraction through the widen --------------------------------
        pc = trimul_pc()
        try:
            base = timeit(dev, lambda: ttnn.matmul(x, b, compute_kernel_config=ckc(),
                                               memory_config=L1, program_config=pc,
                                               dtype=ttnn.bfloat16), reps=20, warm=5)
            # the widen makes the contracted axis logically 320; zero the tail first so the
            # arithmetic is unchanged
            xz = ttnn.fill_implicit_tile_padding(x, 0.0)
            xzw = ttnn.experimental.view(xz, (1, 32, 320, 320))
            wide = timeit(dev, lambda: ttnn.matmul(xzw, bw, compute_kernel_config=ckc(),
                                                   memory_config=L1, program_config=pc,
                                                   dtype=ttnn.bfloat16), reps=20, warm=5)
            out["contraction"] = {"logical_298_us": base, "widened_320_us": wide,
                                  "ratio": base / wide, "saved_us_per_call": base - wide}
        except Exception as e:                                   # noqa: BLE001
            out["contraction"] = {"error": str(e)[:600]}
        print("contraction:", out["contraction"])

        # ---- 5. parity of the widened arm over the valid region ------------------------------
        try:
            r298 = ttnn.to_torch(ttnn.matmul(x, b, compute_kernel_config=ckc(), memory_config=L1,
                                            program_config=pc, dtype=ttnn.bfloat16))
            r320 = ttnn.to_torch(ttnn.matmul(xzw, bw, compute_kernel_config=ckc(), memory_config=L1,
                                         program_config=pc, dtype=ttnn.bfloat16))
            v = r320[:, :, :298, :298]
            out["parity"] = {"torch_equal": bool(torch.equal(r298, v)),
                             "max_abs": float((r298 - v).abs().max()),
                             "n_differ": int((r298 != v).sum()), "n_total": int(r298.numel())}
        except Exception as e:                                   # noqa: BLE001
            out["parity"] = {"error": str(e)[:600]}
        print("parity:", out["parity"])

        # ---- 6. the fill, priced on this build too -------------------------------------------
        try:
            out["fill_implicit_tile_padding_us"] = timeit(
                dev, lambda: ttnn.fill_implicit_tile_padding(x, 0.0), reps=50, warm=5)
            out["clone_us"] = timeit(dev, lambda: ttnn.clone(x, memory_config=L1), reps=20,
                                     warm=3)
        except Exception as e:                                   # noqa: BLE001
            out["fill_error"] = str(e)[:400]
        print("fill", out.get("fill_implicit_tile_padding_us"), "clone", out.get("clone_us"))
    finally:
        if a.out:
            Path(a.out).write_text(json.dumps(out, indent=1))
            print("wrote", a.out)
        ttnn.close_device(dev)
    return out


if __name__ == "__main__":
    main()
