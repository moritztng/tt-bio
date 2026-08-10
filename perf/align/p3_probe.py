#!/usr/bin/env python3
"""p3-align-widen probes.

Deliverable 1: what the tail-zeroing fill actually costs, in the trimul that would carry it,
and whether ttnn.fill_implicit_tile_padding (which ships in the 0.68 wheel and which P2 never
priced) beats the 22.51 us/call multiply_ floor P2 measured.

Deliverable 3: where the alignment dies inside one real trimul call.

    TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=... python3 perf/align/p3_probe.py <cmd> --out x.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import tt_bio.tenstorrent as T  # noqa: E402

CKC = None


def ckc():
    global CKC
    if CKC is None:
        CKC = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)
    return CKC


def L1():
    return ttnn.L1_MEMORY_CONFIG


def DRAM():
    return ttnn.DRAM_MEMORY_CONFIG


def mk(dev, shape, mc, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(*shape, generator=g, dtype=torch.float32)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=mc)


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
    return dt * 1e6            # us/call


# --------------------------------------------------------------------------------------------- #
# D1a — the fill floor, re-opened: fill_implicit_tile_padding vs P2's multiply_
# --------------------------------------------------------------------------------------------- #
def cmd_fillpad(dev, a):
    """Price every tail-zeroing route on the contraction's own operand shape.

    The operand the contraction sees is post-permute [1, C, L, L] with C the channel chunk on
    the batch axis, so BOTH inner axes carry tile padding there. That is the shape priced here.
    """
    out = {}
    shapes = {"operand_1x32x298x298": (1, 32, 298, 298),
              "operand_1x64x298x298": (1, 64, 298, 298),
              "operand_1x8x298x298": (1, 8, 298, 298),
              "xnorm_1x298x298x256": (1, 298, 298, 256)}
    for tag, shp in shapes.items():
        row = {"logical": list(shp)}
        mc = L1() if shp[1] * 320 * 320 * 2 <= 20e6 else DRAM()
        x = mk(dev, shp, mc)
        row["padded"] = list(x.padded_shape)
        row["MB"] = x.padded_shape[-1] * x.padded_shape[-2] * shp[1] * 2 / 1e6
        # A. fill_implicit_tile_padding — writes only the padding lanes
        for name, fn in (
            ("fill_implicit_tile_padding", lambda: ttnn.fill_implicit_tile_padding(x, 0.0)),
        ):
            try:
                us = timeit(dev, fn, reps=50, warm=5)
                y = fn()
                row[name] = {"us": us, "logical": list(y.shape), "padded": list(y.padded_shape),
                             "same_buffer": bool(y.buffer_address() == x.buffer_address())
                             if hasattr(y, "buffer_address") else None}
            except Exception as e:                                # noqa: BLE001
                row[name] = {"error": str(e)[:250]}
        # B. P2's floor: a full-tensor in-place multiply by a 2-D mask
        try:
            m = torch.ones(1, 1, shp[-2], shp[-1], dtype=torch.float32)
            mask = ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=mc)
            xc = ttnn.clone(x, memory_config=mc)
            us = timeit(dev, lambda: ttnn.multiply_(xc, mask), reps=20, warm=3)
            row["multiply__mask2d"] = {"us": us}
            ttnn.deallocate(xc)
            ttnn.deallocate(mask)
        except Exception as e:                                    # noqa: BLE001
            row["multiply__mask2d"] = {"error": str(e)[:250]}
        # C. scale reference: an L1->L1 clone of the same bytes (a pure write of the whole thing)
        try:
            us = timeit(dev, lambda: ttnn.clone(x, memory_config=mc), reps=20, warm=3)
            row["clone_L1"] = {"us": us}
        except Exception as e:                                    # noqa: BLE001
            row["clone_L1"] = {"error": str(e)[:250]}
        ttnn.deallocate(x)
        out[tag] = row
        print(f"  {tag}: " + "  ".join(
            f"{k}={v['us']:.2f}us" for k, v in row.items()
            if isinstance(v, dict) and "us" in v))
    # D. the pre-permute site, where production's mask multiply actually sits: [1,L,L,C]
    try:
        x = mk(dev, (1, 298, 298, 32), L1())
        out["pre_permute_1x298x298x32"] = {
            "logical": list(x.shape), "padded": list(x.padded_shape),
            "fill_implicit_tile_padding_us": timeit(
                dev, lambda: ttnn.fill_implicit_tile_padding(x, 0.0), reps=50, warm=5)}
        ttnn.deallocate(x)
    except Exception as e:                                        # noqa: BLE001
        out["pre_permute_1x298x298x32"] = {"error": str(e)[:250]}
    return out


# --------------------------------------------------------------------------------------------- #
# D1b — does the fill hold through the contraction, and is the contraction still bit-exact?
# --------------------------------------------------------------------------------------------- #
def cmd_fillexact(dev, a):
    """A logical-298 contraction is insensitive to its tail (P2 E7). Confirm that the fill does
    not change the valid region, i.e. that inserting it into production is arithmetically inert
    on its own -- the widen is what turns it into a win."""
    out = {}
    at = torch.randn(1, 32, 298, 298, dtype=torch.float32)
    bt = torch.randn(1, 32, 298, 298, dtype=torch.float32)
    pc = T._triangle_mul_program_config(10)
    aa = ttnn.from_torch(at, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=L1())
    bb = ttnn.from_torch(bt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=L1())
    ref = ttnn.to_torch(ttnn.matmul(aa, bb, compute_kernel_config=ckc(), memory_config=L1(),
                                    program_config=pc, dtype=ttnn.bfloat16))
    aa2 = ttnn.fill_implicit_tile_padding(aa, 0.0)
    got = ttnn.to_torch(ttnn.matmul(aa2, bb, compute_kernel_config=ckc(), memory_config=L1(),
                                    program_config=pc, dtype=ttnn.bfloat16))
    out["filled_vs_unfilled"] = {
        "torch_equal": bool(torch.equal(ref, got)),
        "max_abs": float((ref - got).abs().max()),
        "n_differ": int((ref != got).sum()),
        "n_total": int(ref.numel()),
    }
    print("  fill vs no fill:", out["filled_vs_unfilled"])
    return out


# --------------------------------------------------------------------------------------------- #
# D1c — the real trimul: what the mask route costs in the module the trunk actually runs
# --------------------------------------------------------------------------------------------- #
def _trimul_weights(c_z=256, hidden=256, seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32) * 0.05
    return {
        "norm_in.weight": torch.ones(c_z), "norm_in.bias": r(c_z),
        "norm_out.weight": torch.ones(hidden), "norm_out.bias": r(hidden),
        "g_in.weight": r(2 * hidden, c_z), "p_in.weight": r(2 * hidden, c_z),
        "g_out.weight": r(c_z, c_z), "p_out.weight": r(c_z, hidden),
    }


def cmd_trimul(dev, a):
    """Real `TriangleMultiplication.__call__` at 298 aa, mask on vs mask off.

    Production (`protenix.py:2223`) passes no mask, so the tail-zeroing multiply at
    tenstorrent.py:1347 never runs. This is the A/B that prices turning it on, and the parity
    check on the trimul's own output.
    """
    out = {}
    L, c_z = a.tokens, 256
    mod = T.TriangleMultiplication(False, _trimul_weights(c_z), ckc())
    z = mk(dev, (1, L, L, c_z), DRAM(), seed=7)
    C = T._trimul_chunk_size(L, mod._hidden)
    n_pairs = 2 * mod._hidden // C // 2
    out["shape"] = {"L": L, "c_z": c_z, "chunk": C, "n_pairs": n_pairs,
                    "hidden": mod._hidden,
                    "mem": str(T._triangle_mul_memory_config(L))[:60],
                    "z_padded": list(z.padded_shape)}
    print(f"  L={L} chunk={C} n_pairs={n_pairs} (contractions per trimul call)")

    mt = torch.ones(1, L, L, dtype=torch.float32)
    mask = ttnn.from_torch(mt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=T._triangle_mul_memory_config(L))
    for tag, m in (("mask_off", None), ("mask_on", mask)):
        us = timeit(dev, lambda m=m: mod(z, m), reps=a.reps, warm=2)
        out[tag] = {"us": us, "ms": us / 1e3}
        print(f"  trimul {tag:9s} -> {us / 1e3:8.3f} ms/call")
    if "us" in out["mask_off"] and "us" in out["mask_on"]:
        d = out["mask_on"]["us"] - out["mask_off"]["us"]
        out["mask_cost"] = {"us_per_trimul": d, "us_per_contraction": d / n_pairs,
                            "ratio": out["mask_on"]["us"] / out["mask_off"]["us"]}
        print(f"  mask costs {d:.2f} us/trimul = {d / n_pairs:.2f} us per contraction")

    # parity: does the mask change the valid region of the trimul output?
    r0 = ttnn.to_torch(mod(z, None))
    r1 = ttnn.to_torch(mod(z, mask))
    out["parity_mask_on_vs_off"] = {
        "torch_equal": bool(torch.equal(r0, r1)),
        "max_abs": float((r0 - r1).abs().max()),
        "n_differ": int((r0 != r1).sum()), "n_total": int(r0.numel()),
        "rmsd": float(((r0 - r1) ** 2).mean().sqrt()),
    }
    print("  parity mask on vs off:", out["parity_mask_on_vs_off"])
    return out


# --------------------------------------------------------------------------------------------- #
# D1d — the fill placed where it has to go: immediately before the contraction
# --------------------------------------------------------------------------------------------- #
def cmd_fillinplace(dev, a):
    """Patch the trimul to zero the contraction operand's tail with fill_implicit_tile_padding
    right before ttnn.matmul, and price that against the untouched module."""
    out = {}
    L, c_z = a.tokens, 256
    mod = T.TriangleMultiplication(False, _trimul_weights(c_z), ckc())
    z = mk(dev, (1, L, L, c_z), DRAM(), seed=7)
    C = T._trimul_chunk_size(L, mod._hidden)
    n_pairs = 2 * mod._hidden // C // 2
    base = timeit(dev, lambda: mod(z, None), reps=a.reps, warm=2)
    r0 = ttnn.to_torch(mod(z, None))

    orig = ttnn.matmul

    def patched(x, y, **kw):
        x = ttnn.fill_implicit_tile_padding(x, 0.0)
        return orig(x, y, **kw)

    ttnn.matmul = patched
    try:
        filled = timeit(dev, lambda: mod(z, None), reps=a.reps, warm=2)
        r1 = ttnn.to_torch(mod(z, None))
    finally:
        ttnn.matmul = orig
    out = {"n_pairs": n_pairs, "chunk": C,
           "base_us": base, "filled_us": filled,
           "delta_us_per_trimul": filled - base,
           "delta_us_per_contraction": (filled - base) / n_pairs,
           "parity": {"torch_equal": bool(torch.equal(r0, r1)),
                      "max_abs": float((r0 - r1).abs().max()),
                      "n_differ": int((r0 != r1).sum()), "n_total": int(r0.numel())}}
    print(f"  trimul base {base / 1e3:.3f} ms, with fill {filled / 1e3:.3f} ms, "
          f"{(filled - base) / n_pairs:.2f} us per contraction")
    print("  parity:", out["parity"])
    return out


# --------------------------------------------------------------------------------------------- #
# D3 — where does the alignment die? trace the logical shapes through one real trimul
# --------------------------------------------------------------------------------------------- #
def cmd_trace(dev, a):
    """Record the logical and padded shape of every ttnn op output inside one real trimul call,
    so the point at which a 320-logical tensor would be re-sliced back to 298 is visible."""
    recs = []
    wrapped = {}
    for name in ("matmul", "linear", "permute", "transpose", "layer_norm", "multiply_",
                 "multiply", "concat", "clone", "reallocate", "chunk"):
        fn = getattr(ttnn, name, None)
        if fn is None:
            continue
        wrapped[name] = fn

        def mkw(name=name, fn=fn):
            def w(*args, **kw):
                r = fn(*args, **kw)
                try:
                    ins = [(list(t.shape), list(t.padded_shape)) for t in args
                           if isinstance(t, ttnn.Tensor)]
                    outs = r if isinstance(r, (list, tuple)) else [r]
                    o = [(list(t.shape), list(t.padded_shape)) for t in outs
                         if isinstance(t, ttnn.Tensor)]
                    recs.append({"op": name, "in": ins, "out": o})
                except Exception:                                 # noqa: BLE001
                    pass
                return r
            return w
        setattr(ttnn, name, mkw())
    try:
        mod = T.TriangleMultiplication(False, _trimul_weights(256), ckc())
        z = mk(dev, (1, a.tokens, a.tokens, 256), DRAM(), seed=7)
        recs.clear()
        mod(z, None)
    finally:
        for name, fn in wrapped.items():
            setattr(ttnn, name, fn)
    # collapse the chunk loop: keep the first iteration's ops plus everything after it
    return {"n_records": len(recs), "records": recs}


def cmd_grid(dev, a):
    return {"grid": list(T.COMPUTE_GRID_MAIN)}


CMDS = {"fillpad": cmd_fillpad, "fillexact": cmd_fillexact, "trimul": cmd_trimul,
        "fillinplace": cmd_fillinplace, "trace": cmd_trace, "grid": cmd_grid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=sorted(CMDS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokens", type=int, default=298)
    ap.add_argument("--reps", type=int, default=6)
    a = ap.parse_args()
    dev = T.get_device()
    print(f"grid {T.COMPUTE_GRID_MAIN}")
    t0 = time.time()
    res = CMDS[a.cmd](dev, a)
    res["_cmd"] = a.cmd
    res["_grid"] = list(T.COMPUTE_GRID_MAIN)
    res["_wall_s"] = time.time() - t0
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
        print("wrote", a.out)
    T.cleanup()


if __name__ == "__main__":
    main()
