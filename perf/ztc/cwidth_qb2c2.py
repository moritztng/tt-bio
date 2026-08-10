#!/usr/bin/env python3
"""z-transition-chunk -- the correction probe: is the h=32 clash about in-block L1 pressure, or
about the fc1's own channel width?

Why this exists. The pass-1 record (§4.2, §4.4 of the state doc) concluded that h=32 "runs to
completion and returns a bit-exact result" at "the fold's own [1,512,512,256] shape" outside a
fold, and read that as proof the mechanism is in-block L1 pressure. The raw JSON it was written
from records `parity.shape = [1,512,512,64]`. The harness's `forced()` predicate accepts any
`x.shape[-1] <= 256`, so `GRAB` captured the FIRST forced call, which is the c=64 TEMPLATE
Transition, not the c=256 pair Transition the leg owns. Every parity number in §4.4 is a c=64
number and the isolation claim in §4.2 is a c=64 claim.

So two things need re-taking at the right channel width:

  A. the clash boundary   bare fc1 [1,h,512,c] x [c,4c] -> L1, over h x c. If the clash tracks c
                          at fixed h, in-block pressure is not the mechanism and the matmul's own
                          static CB allocation is.
  B. QC, properly         a real `Transition` at [1,512,512,c] for c = 64 AND c = 256, torch.equal
                          across heights against production h=16, including ragged heights.

Weights are synthetic and fixed per c, which is all torch.equal across heights needs -- the
comparison is height-vs-height on identical weights, not against a checkpoint.

Both sides of every timed region synchronise. NOTHING IN tt_bio/ IS CHANGED.
qb2 card 2 / ttnn 0.68.0 -- RATIO inputs, owe a qb1/0.67.4 re-take.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import ttnn  # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def free(*ts):
    for t in ts:
        if t is not None:
            try:
                ttnn.deallocate(t)
            except Exception:                                                   # noqa: BLE001
                pass


def short(e):
    s = str(e).replace("\n", " ")
    return s[:230]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--heights", default="16,20,24,28,32")
    ap.add_argument("--widths", default="64,128,256")
    a = ap.parse_args()

    import importlib.metadata as im
    import tt_bio.tenstorrent as T

    dev = T.get_device()
    gx, gy = T.COMPUTE_GRID_MAIN
    HEIGHTS = [int(s) for s in a.heights.split(",")]
    WIDTHS = [int(s) for s in a.widths.split(",")]
    res = {"host": "qb2", "card": "physical 2", "ttnn": im.version("ttnn"),
           "note": "qb2 / ttnn 0.68.0 -- RATIO inputs, owe a qb1/0.67.4 re-take",
           "core_grid_main": f"{gx}x{gy}", "cores_main": gx * gy,
           "prod_chunk_default": T.TRANSITION_H_CHUNK_SIZE,
           "why": "pass-1 parity/isolation was taken at c=64 (template), not c=256 (pair)"}
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

    # --- A. the clash boundary over (h, c), bare fc1 only ------------------------------------
    grid = []
    for c in WIDTHS:
        for h in HEIGHTS:
            M, N = h * 512, 4 * c
            rec = {"c": c, "h": h, "x_shape": [1, h, 512, c], "M": M, "N": N,
                   "Mt": M // 32, "nt": N // 32, "out_MB": round(M * N * 2 / 1e6, 3)}
            x = w = None
            try:
                x = ttnn.from_torch(torch.randn(1, h, 512, c), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
                w = ttnn.from_torch(torch.randn(c, N), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
                for _ in range(2):
                    free(ttnn.linear(x, w, activation="silu", compute_kernel_config=ckc,
                                     memory_config=L1, dtype=ttnn.bfloat16,
                                     core_grid=T.CORE_GRID_MAIN))
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(4):
                    free(ttnn.linear(x, w, activation="silu", compute_kernel_config=ckc,
                                     memory_config=L1, dtype=ttnn.bfloat16,
                                     core_grid=T.CORE_GRID_MAIN))
                ttnn.synchronize_device(dev)
                rec["ok"] = True
                rec["ms"] = round((time.perf_counter() - t0) / 4 * 1e3, 4)
            except Exception as e:                                              # noqa: BLE001
                rec["ok"] = False
                rec["err"] = short(e)
            finally:
                free(x, w)
            grid.append(rec)
            print("  fc1 " + json.dumps(rec), flush=True)
    res["clash_grid"] = grid

    # --- B. QC at the real pair shape, via a real Transition ---------------------------------
    par = {}
    for c in WIDTHS:
        n = 4 * c
        sd = {"norm.weight": torch.ones(c), "norm.bias": torch.zeros(c),
              "fc1.weight": torch.randn(n, c) * 0.05,
              "fc2.weight": torch.randn(n, c) * 0.05,
              "fc3.weight": torch.randn(c, n) * 0.05}
        block = {"c": c, "shape": [1, 512, 512, c]}
        try:
            inst = T.Transition(sd, ckc)
        except Exception as e:                                                  # noqa: BLE001
            block["build_err"] = short(e)
            par[f"c{c}"] = block
            print("  tr " + json.dumps(block), flush=True)
            continue
        xt = (torch.randn(1, 512, 512, c) * 0.5)
        outs, prod = {}, T.TRANSITION_H_CHUNK_SIZE
        for h in HEIGHTS:
            xz = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                 memory_config=DRAM)
            y = None
            try:
                T.TRANSITION_H_CHUNK_SIZE = h
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                y = inst(xz)
                ttnn.synchronize_device(dev)
                block[f"h{h}_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
                outs[h] = ttnn.to_torch(y).clone()
            except Exception as e:                                              # noqa: BLE001
                block[f"h{h}_THROW"] = short(e)
            finally:
                T.TRANSITION_H_CHUNK_SIZE = prod
                free(y, xz)
        ref = outs.get(prod)
        for h, y in outs.items():
            nch = -(-512 // h)
            eff = -(-512 // nch)
            last = 512 - eff * (nch - 1)
            r = {"n_chunks": nch, "eff_h": eff, "last_chunk": last, "ragged": last != eff}
            if ref is not None:
                d = y.float() - ref.float()
                r["torch_equal_vs_h%d" % prod] = bool(torch.equal(y, ref))
                r["max_abs"] = float(d.abs().max())
                r["rel_rmsd"] = float(d.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt())
            block[f"h{h}"] = r
            print(f"  tr c={c} h={h} " + json.dumps(r), flush=True)
        block["heights_that_allocate"] = sorted(outs)
        par[f"c{c}"] = block
    res["transition_parity"] = par

    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote " + a.out, flush=True)


if __name__ == "__main__":
    main()
