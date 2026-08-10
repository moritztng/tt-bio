#!/usr/bin/env python3
"""p2-alignment, part 2: what an aligned fill costs, and the kernel-cache kill test."""
import argparse, json, sys, time
from pathlib import Path
import torch, ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import tt_bio.tenstorrent as T  # noqa: E402

from a_probe import ckc, L1, DRAM, mk, timeit, _run_arm, _pc  # noqa: E402


def cmd_fill2(dev, a):
    out = {}
    pc = T._triangle_mul_program_config(10)
    # A. ttnn.pad on one contraction operand, L1, [1,32,298,298] -> [1,32,320,320]
    x = mk(dev, (1, 32, 298, 298), L1())
    try:
        us = timeit(dev, lambda: ttnn.pad(x, [(0, 0), (0, 0), (0, 22), (0, 22)], value=0.0),
                    reps=20, warm=2)
        y = ttnn.pad(x, [(0, 0), (0, 0), (0, 22), (0, 22)], value=0.0)
        out["pad_operand_L1"] = {"us": us, "logical": list(y.shape), "padded": list(y.padded_shape)}
        print(f"  ttnn.pad one operand (L1 6.55MB) -> {us:8.2f} us, logical now {list(y.shape)}")
        ttnn.deallocate(y)
    except Exception as e:                                       # noqa: BLE001
        out["pad_operand_L1"] = {"error": str(e)[:200]}
        print("  pad operand:", str(e)[:160])
    ttnn.deallocate(x)
    # B. the same fill once on the pair tensor z, [1,298,298,256] DRAM -> [1,320,320,256]
    z = mk(dev, (1, 298, 298, 256), DRAM())
    try:
        us = timeit(dev, lambda: ttnn.pad(z, [(0, 0), (0, 22), (0, 22), (0, 0)], value=0.0),
                    reps=6, warm=2)
        y = ttnn.pad(z, [(0, 0), (0, 22), (0, 22), (0, 0)], value=0.0)
        out["pad_z_DRAM"] = {"us": us, "logical": list(y.shape), "padded": list(y.padded_shape),
                             "MB_in": 298 * 320 * 256 * 2 / 1e6, "MB_out": 320 * 320 * 256 * 2 / 1e6}
        print(f"  ttnn.pad z once (DRAM 48.8MB) -> {us:8.2f} us, logical now {list(y.shape)}")
        ttnn.deallocate(y)
    except Exception as e:                                       # noqa: BLE001
        out["pad_z_DRAM"] = {"error": str(e)[:200]}
        print("  pad z:", str(e)[:160])
    ttnn.deallocate(z)
    # C. does the trunk's other work get more expensive once z is logically 320? Same padded shape
    #    on the row axis is impossible -- dim1 is not a tiled axis, so 320 rows are 320 real rows.
    for tag, rows in (("proj_rows298_real", 298), ("proj_rows320_real", 320)):
        zz = mk(dev, (1, rows, 298, 256), DRAM())
        w = mk(dev, (256, 256), DRAM(), seed=1)
        cfg = T._pair_proj_config(zz, w)
        us = timeit(dev, lambda: ttnn.linear(zz, w, compute_kernel_config=ckc(),
                                             program_config=cfg, dtype=ttnn.bfloat16),
                    reps=8, warm=2)
        out[tag] = {"us": us, "padded": list(zz.padded_shape)}
        print(f"  {tag:20s} padded {list(zz.padded_shape)} -> {us:8.2f} us")
        ttnn.deallocate(zz); ttnn.deallocate(w)
    # D. the end-to-end fixed arm: pad both operands, then contract, against the production arm
    ap = mk(dev, (1, 32, 298, 298), L1())
    bp = mk(dev, (1, 32, 298, 298), L1(), seed=1)
    us_prod = timeit(dev, lambda: ttnn.matmul(ap, bp, compute_kernel_config=ckc(),
                                              memory_config=L1(), program_config=pc,
                                              dtype=ttnn.bfloat16), reps=20)

    def fixed():
        aa = ttnn.pad(ap, [(0, 0), (0, 0), (0, 22), (0, 22)], value=0.0)
        bb = ttnn.pad(bp, [(0, 0), (0, 0), (0, 22), (0, 22)], value=0.0)
        o = ttnn.matmul(aa, bb, compute_kernel_config=ckc(), memory_config=L1(),
                        program_config=pc, dtype=ttnn.bfloat16)
        ttnn.deallocate(aa); ttnn.deallocate(bb)
        return o
    try:
        us_fixed = timeit(dev, fixed, reps=20)
        out["end_to_end"] = {"production_us": us_prod, "pad_then_contract_us": us_fixed,
                             "ratio": us_fixed / us_prod}
        print(f"  production 298-logical contraction     -> {us_prod:8.2f} us")
        print(f"  pad both operands + aligned contraction -> {us_fixed:8.2f} us "
              f"({us_fixed / us_prod:.3f}x)")
    except Exception as e:                                       # noqa: BLE001
        out["end_to_end"] = {"error": str(e)[:200]}
    ttnn.deallocate(ap); ttnn.deallocate(bp)
    return out


def cmd_arm(dev, a):
    """One arm, one call. Used with an empty kernel cache so the generated tree can be diffed."""
    pc = T._triangle_mul_program_config(10)
    sh = (298, 298) if a.arm == "unaligned" else (320, 320)
    bs = (298, 320) if a.arm == "unaligned" else (320, 320)
    x = mk(dev, (1, 32) + sh, L1())
    y = mk(dev, (1, 32) + bs, L1(), seed=1)
    o = ttnn.matmul(x, y, compute_kernel_config=ckc(), memory_config=L1(), program_config=pc,
                    dtype=ttnn.bfloat16)
    ttnn.synchronize_device(dev)
    print(f"  arm={a.arm} logical a {list(x.shape)} b {list(y.shape)} -> out {list(o.shape)}")
    return {"arm": a.arm, "a": list(x.shape), "b": list(y.shape), "out": list(o.shape)}


CMDS = {"fill2": cmd_fill2, "arm": cmd_arm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=sorted(CMDS))
    ap.add_argument("--arm", default="aligned")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = T.get_device()
    res = CMDS[a.cmd](dev, a)
    res["_grid"] = list(T.COMPUTE_GRID_MAIN)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    T.cleanup()


if __name__ == "__main__":
    main()
