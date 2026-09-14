#!/usr/bin/env python3
"""`_PAIR_FFN_FC1_BLOCK_W` on Wormhole: is 16 the right rung, and does 32 clash here too?

The constant was swept on a p300c (11x10) with a clash documented one rung up, and it reaches
ESMFold2/ESMC's pair FFN on every part with no gate. The arm is the real `SwiGLUFFN.__call__`,
not a hand-rolled chain, so every fallback the module owns -- the row-block shrink, the L1 slice,
the L1 layer-norm, the per-shape refusal memo -- is in the measurement. A hand-rolled chain pins
`memory_config=L1` on the SiLU product where the module would not, and that alone OOMs here.

Interleaved, fixed arm order, the shipped width run twice as its own A/A. `L1_FC1_STATS` is
[served, declined] on the constant's own call, so a width that quietly takes the DRAM path is
reported as declined rather than as a slow pass.
"""
from __future__ import annotations
import argparse, json, platform, time
from pathlib import Path

import torch, ttnn
import tt_bio.tenstorrent as T
import tt_bio.esmc as E

D = ttnn.DRAM_MEMORY_CONFIG


def build(ck, cz, dff):
    """A SwiGLUFFN with ESMFold2's pair-transition layout: fc1 [2*dff, cz], fc2 [cz, dff]."""
    torch.manual_seed(0)
    sd = {"0.weight": torch.randn(cz, dtype=torch.float32) * 0.02 + 1.0,
          "0.bias": torch.zeros(cz, dtype=torch.float32),
          "1.weight": torch.randn(2 * dff, cz, dtype=torch.float32) * 0.02,
          "3.weight": torch.randn(cz, dff, dtype=torch.float32) * 0.02}
    return E.SwiGLUFFN(sd, ck, fuse_swiglu=True)


def clear(shape_key):
    for name in ("_L1_LN_REFUSED", "_L1_SLICE_REFUSED", "_FUSED_RESID_REFUSED",
                 "_FILL_ASSEMBLY_REFUSED", "_UNBLOCKED_REFUSED", "_ROW_BLOCK_SHRUNK"):
        d = getattr(E, name, None)
        if d is not None:
            d.clear()
    T._L1_OUT_REFUSED.clear()
    T._L1_OUT_RUNG.clear()
    T.LATCH_STATS.pop("l1_out", None)
    E.L1_FC1_STATS[0] = E.L1_FC1_STATS[1] = 0
    T._pair_proj_program_config.cache_clear()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--l", type=int, default=512)
    ap.add_argument("--cz", type=int, default=256)
    ap.add_argument("--dff", type=int, default=1024)
    ap.add_argument("--widths", default="16,16,8,24,32,48,64")
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()

    dev = T.get_device()
    cc = dev.compute_with_storage_grid_size()
    kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    ck = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    ffn = build(ck, a.cz, a.dff)
    torch.manual_seed(1)
    x = ttnn.from_torch(torch.randn(1, a.l, a.l, a.cz, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)
    res = {"host": platform.node(), "arch": str(dev.arch()), "grid": [cc.x, cc.y],
           "shipped_block_w": T._PAIR_FFN_FC1_BLOCK_W, "bw": T._PAIR_FFN_FC1_BW,
           "split_swiglu": ffn.split_swiglu, "is_small_grid": T._IS_SMALL_GRID,
           "l1_fc1_on": E._PAIR_FFN_L1_FC1, "row_block": E._PAIR_FFN_ROW_BLOCK,
           "shape": {"L": a.l, "cz": a.cz, "dff": a.dff}, "rows": []}
    ref = None
    for i, w in enumerate(a.widths.split(",")):
        obw = int(w)
        old = T._PAIR_FFN_FC1_BLOCK_W
        T._PAIR_FFN_FC1_BLOCK_W = obw
        clear(tuple(x.padded_shape))
        rec = {"arm": "obw%d_%d" % (obw, i), "obw": obw}
        try:
            ttnn.deallocate(ffn(x))                       # one warm call, then the census
            clear(tuple(x.padded_shape))
            best = float("inf")
            for _ in range(a.blocks):
                t0 = time.perf_counter()
                for _ in range(a.reps):
                    ttnn.deallocate(ffn(x))
                ttnn.synchronize_device(dev)
                best = min(best, (time.perf_counter() - t0) / a.reps)
            rec["ms"] = best * 1e3
            rec["l1_fc1"] = {"served": E.L1_FC1_STATS[0], "declined": E.L1_FC1_STATS[1]}
            rec["latch"] = {k: v for k, v in T.LATCH_STATS.get("l1_out", {}).items() if k != "why"}
            rec["why"] = T.LATCH_STATS.get("l1_out", {}).get("why", [])[:1]
            host = ttnn.to_torch(ffn(x))
            if ref is None:
                ref = host
                rec["equal_to_shipped"] = True
            else:
                rec["equal_to_shipped"] = bool(torch.equal(host, ref))
                rec["max_abs"] = float((host.float() - ref.float()).abs().max())
        except Exception as e:                                              # noqa: BLE001
            rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:160].replace("\n", " "))
        finally:
            T._PAIR_FFN_FC1_BLOCK_W = old
        res["rows"].append(rec)
        print("  obw=%-3d %s  fc1=%s latch=%s eq=%s" % (
            obw, ("%8.4f ms" % rec["ms"]) if "ms" in rec else rec.get("error"),
            rec.get("l1_fc1"), rec.get("latch"), rec.get("equal_to_shipped")), flush=True)
    T.cleanup()
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    print("wrote", p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
