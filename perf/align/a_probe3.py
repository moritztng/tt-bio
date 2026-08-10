#!/usr/bin/env python3
"""p2-alignment, part 3: can the widen be made free, and does the reader really zero the tail?

Three questions Phase 3 needs answered before it designs anything:
  1. does the logical-298 contraction's RESULT depend on what sits in the padded tail? If not, the
     NCRISC reader is zero-filling it on every call -- which is exactly what it is being paid 1.585x
     to do.
  2. is there an API in ttnn 0.68 that widens a logical shape into padding the tensor already owns,
     at no cost? `ttnn.experimental.view` advertises a 0-cost view; `ttnn.reshape` takes a pad_value.
  3. if there is, what does the whole fix cost end to end -- zero the tail once, relabel, contract --
     against the 40.79 us/call the alignment saves, and is it still bit-exact?
"""
import argparse, json, sys, time
from pathlib import Path
import torch, ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import tt_bio.tenstorrent as T  # noqa: E402
from a_probe import ckc, L1, DRAM, mk, timeit  # noqa: E402

B, S, SP = 32, 298, 320


def _mask(dev, shape):
    m = torch.ones(*shape, dtype=torch.float32)
    return ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=L1())


def cmd_tail(dev, a):
    """Does the logical-298 result depend on the padded tail? Pollute it with a layer_norm bias."""
    pc = T._triangle_mul_program_config(10)
    g = torch.Generator().manual_seed(11)
    at = torch.randn(1, B, S, S, generator=g, dtype=torch.float32)
    bt = torch.randn(1, B, S, S, generator=g, dtype=torch.float32)
    out = {}

    def up(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=L1())

    # A: clean tail (from_torch zero-pads)
    ac, bc = up(at), up(bt)
    o_clean = ttnn.matmul(ac, bc, compute_kernel_config=ckc(), memory_config=L1(),
                          program_config=pc, dtype=ttnn.bfloat16)
    r_clean = ttnn.to_torch(o_clean)

    # B: tail polluted the way the fold pollutes it -- a layer_norm with a non-zero bias writes
    #    beta into every padded row, because the norm of an all-zero row is beta.
    w = ttnn.from_torch(torch.ones(S), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=L1())
    bias = ttnn.from_torch(torch.full((S,), 7.0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=L1())
    ap = ttnn.layer_norm(up(at), weight=w, bias=bias, epsilon=1e-5,
                         compute_kernel_config=ckc(), memory_config=L1())
    bp = ttnn.layer_norm(up(bt), weight=w, bias=bias, epsilon=1e-5,
                         compute_kernel_config=ckc(), memory_config=L1())
    o_poll = ttnn.matmul(ap, bp, compute_kernel_config=ckc(), memory_config=L1(),
                         program_config=pc, dtype=ttnn.bfloat16)
    r_poll = ttnn.to_torch(o_poll)

    # C: the same polluted operands with the tail zeroed by a mask multiply -- an eltwise op works
    #    on whole tiles, so multiplying by a mask whose own padding is 0 zeroes the operand's tail.
    m = _mask(dev, (1, 1, S, S))
    az = ttnn.multiply(ap, m, memory_config=L1())
    bz = ttnn.multiply(bp, m, memory_config=L1())
    o_zero = ttnn.matmul(az, bz, compute_kernel_config=ckc(), memory_config=L1(),
                         program_config=pc, dtype=ttnn.bfloat16)
    r_zero = ttnn.to_torch(o_zero)

    eq = torch.equal(r_poll, r_zero)
    d = (r_poll.float() - r_zero.float())
    out["polluted_vs_zeroed"] = {"torch_equal": bool(eq), "max_abs": float(d.abs().max()),
                                 "n_diff": int((d != 0).sum()), "n_total": int(d.numel())}
    print(f"  logical-298 result, polluted tail vs zeroed tail: torch.equal={eq} "
          f"max_abs={float(d.abs().max()):.3e}  {int((d != 0).sum())}/{d.numel()} differ")
    out["_note"] = "beta=7.0 written into every padded row by layer_norm, as the fold does"
    for t in (ac, bc, o_clean, ap, bp, o_poll, az, bz, o_zero, m, w, bias):
        try:
            ttnn.deallocate(t)
        except Exception:                                        # noqa: BLE001
            pass
    return out


def cmd_widen(dev, a):
    """Is there a zero-cost widen from logical 298 into padding the tensor already owns?"""
    out = {}
    x = mk(dev, (1, B, S, S), L1())
    print(f"  operand logical {list(x.shape)} padded {list(x.padded_shape)}")
    for tag, fn in (
        ("experimental_view", lambda: ttnn.experimental.view(x, (1, B, SP, SP))),
        ("reshape_pad_value", lambda: ttnn.reshape(x, (1, B, SP, SP), pad_value=0.0)),
        ("reshape_plain", lambda: ttnn.reshape(x, (1, B, SP, SP))),
    ):
        try:
            y = fn()
            us = timeit(dev, fn, reps=50, warm=3)
            out[tag] = {"us": us, "logical": list(y.shape), "padded": list(y.padded_shape),
                        "same_buffer": None}
            print(f"  {tag:20s} -> {us:8.3f} us  logical {list(y.shape)} padded "
                  f"{list(y.padded_shape)}")
        except Exception as e:                                   # noqa: BLE001
            out[tag] = {"error": str(e).split(chr(10))[0][:180]}
            print(f"  {tag:20s} -> FAILS: {str(e).split(chr(10))[0][:150]}")
    ttnn.deallocate(x)
    return out


def cmd_fixed(dev, a):
    """The whole candidate fix, end to end, against production: zero the tail, widen, contract."""
    pc = T._triangle_mul_program_config(10)
    g = torch.Generator().manual_seed(13)
    at = torch.randn(1, B, S, S, generator=g, dtype=torch.float32)
    bt = torch.randn(1, B, S, S, generator=g, dtype=torch.float32)

    def up(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=L1())
    ap, bp = up(at), up(bt)
    m = _mask(dev, (1, 1, S, S))
    out = {}

    def prod():
        return ttnn.matmul(ap, bp, compute_kernel_config=ckc(), memory_config=L1(),
                           program_config=pc, dtype=ttnn.bfloat16)
    out["production_us"] = timeit(dev, prod, reps=20)
    print(f"  production (logical 298)                   -> {out['production_us']:8.2f} us")

    # variants of the fix, cheapest first
    def mask_only():
        az = ttnn.multiply(ap, m, memory_config=L1())
        ttnn.deallocate(az)
    out["mask_multiply_one_operand_us"] = timeit(dev, mask_only, reps=20)
    print(f"  cost: mask multiply, one operand, L1       -> "
          f"{out['mask_multiply_one_operand_us']:8.2f} us")

    # production's own slot at tenstorrent.py:1347 is an IN-PLACE multiply, which reads and writes
    # the operand once instead of allocating a second 6.55 MB output. Price that, and a rank-1 mask.
    m1 = _mask(dev, (1, 1, 1, S))
    for tag, mm in (("inplace_2d_mask", m), ("inplace_1d_mask", m1)):
        tmp = up(at)

        def one():
            ttnn.multiply_(tmp, mm)
        try:
            us = timeit(dev, one, reps=20)
            out[f"mask_{tag}_us"] = us
            print(f"  cost: {tag:22s}          -> {us:8.2f} us")
        except Exception as e:                                   # noqa: BLE001
            out[f"mask_{tag}_us"] = None
            print(f"  cost: {tag}: {str(e).split(chr(10))[0][:120]}")
        ttnn.deallocate(tmp)

    for tag, widen in (("view", lambda t: ttnn.experimental.view(t, (1, B, SP, SP))),
                       ("reshape_pad", lambda t: ttnn.reshape(t, (1, B, SP, SP), pad_value=0.0))):
        try:
            def fixed():
                az = ttnn.multiply(ap, m, memory_config=L1())
                aw, bw = widen(az), widen(bp)
                o = ttnn.matmul(aw, bw, compute_kernel_config=ckc(), memory_config=L1(),
                                program_config=pc, dtype=ttnn.bfloat16)
                ttnn.deallocate(az)
                return o
            us = timeit(dev, fixed, reps=20)
            o_fix = fixed()
            o_prod = prod()
            rf = ttnn.to_torch(o_fix)[..., :S, :S]
            rp = ttnn.to_torch(o_prod)[..., :S, :S]
            eq = torch.equal(rf, rp)
            d = (rf.float() - rp.float())
            out[f"fixed_{tag}"] = {"us": us, "ratio_vs_production": us / out["production_us"],
                                   "torch_equal": bool(eq), "max_abs": float(d.abs().max()),
                                   "rmsd": float(d.pow(2).mean().sqrt()),
                                   "n_diff": int((d != 0).sum()), "n_total": int(d.numel())}
            print(f"  fix via {tag:12s} -> {us:8.2f} us ({us / out['production_us']:.3f}x) "
                  f"torch.equal={eq} max_abs={float(d.abs().max()):.3e}")
            ttnn.deallocate(o_fix); ttnn.deallocate(o_prod)
        except Exception as e:                                   # noqa: BLE001
            out[f"fixed_{tag}"] = {"error": str(e).split(chr(10))[0][:180]}
            print(f"  fix via {tag}: FAILS {str(e).split(chr(10))[0][:150]}")
    return out


CMDS = {"tail": cmd_tail, "widen": cmd_widen, "fixed": cmd_fixed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=sorted(CMDS))
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
