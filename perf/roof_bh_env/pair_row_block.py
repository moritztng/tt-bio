#!/usr/bin/env python3
"""`PAIR_ROW_BLOCK = 128`: an envelope nobody measured, or an arbitrary tile multiple?

The comment claims only "a tile multiple". `_pair_bias_from_z` is the one of its three use sites
that can be driven without model weights, and it is a faithful instance of the shape: a row-local
layer_norm plus a per-row projection, blocked over the leading dim, with the whole-tensor result
as the bit-exact reference. Sweeping the block height there answers both halves of the question at
once -- whether any height refuses (an envelope) and whether the shipped one is on a flat curve
(arbitrary) or a peak (fitted by accident).

Fixed arm order, the shipped height run twice as its own A/A, chunk=None as the reference.
"""
from __future__ import annotations
import argparse, json, platform, time
from pathlib import Path

import torch, ttnn
import tt_bio.tenstorrent as T

D = ttnn.DRAM_MEMORY_CONFIG


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="512,640,768")
    ap.add_argument("--cz", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--chunks", default="32,64,128,128,192,256,320,512")
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()

    dev = T.get_device()
    cc = dev.compute_with_storage_grid_size()
    kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    ck = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"host": platform.node(), "arch": str(dev.arch()), "grid": [cc.x, cc.y],
           "shipped": T.PAIR_ROW_BLOCK, "seq_len_more_chunking": T.SEQ_LEN_MORE_CHUNKING,
           "is_small_grid": T._IS_SMALL_GRID, "sweeps": []}
    for S in [int(v) for v in a.sizes.split(",")]:
        torch.manual_seed(0)
        mk = lambda *s, dt=torch.bfloat16: ttnn.from_torch(
            torch.randn(*s, dtype=dt), layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)
        z = mk(S, S, a.cz)
        # Same construction `Module.torch_to_tt` uses for a norm weight: TILE layout, bf16.
        lw = ttnn.from_torch(torch.ones(a.cz, dtype=torch.float32), layout=ttnn.TILE_LAYOUT,
                             device=dev, dtype=ttnn.bfloat16)
        lb = ttnn.from_torch(torch.zeros(a.cz, dtype=torch.float32), layout=ttnn.TILE_LAYOUT,
                             device=dev, dtype=ttnn.bfloat16)
        bw = mk(a.cz, a.heads)   # torch [heads, cz] transposed, as AF2 loads it
        rows = []
        ref = None
        for label, chunk in [("whole", None)] + [("c%s_%d" % (c, i), int(c))
                                                 for i, c in enumerate(a.chunks.split(","))]:
            if chunk is not None and chunk > S:
                continue
            rec = {"arm": label, "chunk": chunk, "S": S, "blocks": None if chunk is None
                   else -(-S // chunk)}
            try:
                call = lambda: T._pair_bias_from_z(z, lw, lb, bw, ck, chunk)
                ttnn.deallocate(call())
                best = float("inf")
                for _ in range(a.blocks):
                    t0 = time.perf_counter()
                    for _ in range(a.reps):
                        ttnn.deallocate(call())
                    ttnn.synchronize_device(dev)
                    best = min(best, (time.perf_counter() - t0) / a.reps)
                rec["ms"] = best * 1e3
                host = ttnn.to_torch(call())
                if ref is None:
                    ref = host
                    rec["equal_to_whole"] = True
                else:
                    rec["equal_to_whole"] = bool(torch.equal(host, ref))
                    rec["max_abs"] = float((host.float() - ref.float()).abs().max())
            except Exception as e:                                          # noqa: BLE001
                rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:160].replace("\n", " "))
            rows.append(rec)
            print("  S=%-4d %-10s chunk=%-5s blocks=%-4s %s eq=%s" % (
                S, label, chunk, rec["blocks"],
                ("%8.4f ms" % rec["ms"]) if "ms" in rec else rec.get("error"),
                rec.get("equal_to_whole")), flush=True)
        res["sweeps"].append({"S": S, "rows": rows})
        for t in (z, lw, lb, bw):
            ttnn.deallocate(t)
    T.cleanup()
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    print("wrote", p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
