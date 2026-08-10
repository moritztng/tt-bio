#!/usr/bin/env python3
"""Q-B — price the k-chunked out-projection region at [1,512,512,256].

The hand-off under test (z-progcfg-h5 §"ms per fold"): at 512 aa the pair track's destination term
is ZERO because `p_out` and `g_out` need 268.4 MB of live L1 against 160.8 MB on the chip, and P7
proved no program config recovers it. Split the region into k row blocks and the live set drops to
268.4/k. The byte model calibrated at 82-92 % of predicted on three shapes puts the destination term
at ~0.72 ms/region, i.e. ~760 ms/fold over 1048 regions.

That is a projection from a model. This measures it. Predictions K1-K5 are in
perf/progcfg/PREDICTIONS_INFOLD.md, committed before this file opened a device; the short version is
that ~0.72 is expected to be about 2.2x too high because it never charged the assembly, and that the
best k is expected to be roughly break-even.

THE ASSEMBLY IS INCLUDED IN EVERY CELL. `z-rowblock` measured the same method on the pair transpose
and found the sibling probe's 1.777 ms/call became 1.338 once the blocks were concatenated back into
the single tensor production actually returns -- a probe that appends blocks to a list and frees them
is not measuring a production-legal form. A `noassembly` variant is reported alongside purely to
quantify how big that error is here.

Cells, per k:
  l1          k blocks -> tuned L1-output config, per-block sigmoid multiply_, one concat to DRAM
  l1_spill    same, but each product is written straight to DRAM as it is produced and its two L1
              operands freed, so live L1 is ONE block and not the whole tensor
  dram        k blocks -> tuned DRAM-output config, one concat to DRAM         <- K4's control
  l1_noass    l1 without the concat                                            <- the sibling's form
k=1 dram is production today and is the control every other cell is scored against.

`l1` and `l1_noass` both hold all k product blocks in L1 until the end, so their live set is the
WHOLE 134.22 MB tensor at every k -- they do not reduce L1 at all, and at 512:256 they both die on a
circular-buffer clash partway through the block loop for exactly that reason. `l1_spill` is the only
one of the three that is the thing the hand-off proposes. Same distinction z-rowblock draws between
its variants C and D, reached here from the other direction and the hard way.

Usage (qb2 chip 0):

  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-h5-infold \\
  TT_MESH_GRAPH_DESC_PATH=$TTNN/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \\
  python3 perf/progcfg/h5_kchunk.py --shapes 512:256,298:256 --ks 1,2,4,8,16 \\
      --out perf/progcfg/h5_kchunk_qb2c0.json

qb2 runs ttnn 0.68.0: every absolute here is a RATIO owing a qb1/0.67.4 re-take (charter 4.8).
"""
import argparse, json, statistics as st, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch    # noqa: E402
import ttnn     # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def ceil32(v):
    return -(-v // 32) * 32


def timed(dev, fn, warm=2, pipe=2, reps=5):
    """Median seconds per call. Synchronise on both sides of every timed region."""
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / pipe)
    return st.median(out)


def cfg_fields(pc):
    if pc is None:
        return None
    return {f: getattr(pc, f) for f in ("in0_block_w", "out_subblock_h", "out_subblock_w",
                                        "out_block_h", "out_block_w", "per_core_M", "per_core_N")
            if hasattr(pc, f)}


def roofs_at_shape(dev, x, tbytes):
    """The copy roof AT THIS SHAPE, both destinations, re-taken on this card this pass.
    Charter 4.1: roofs are per-card and are never inherited."""
    out = {}
    for tag, mc in (("dram", DRAM), ("l1", L1)):
        try:
            f = lambda mc=mc: ttnn.deallocate(ttnn.clone(x, memory_config=mc))   # noqa: E731
            s = timed(dev, f, warm=2, pipe=2, reps=5)
            out[f"clone_{tag}_ms"] = round(s * 1e3, 4)
            out[f"clone_{tag}_GBps"] = round(2 * tbytes / s / 1e9, 1)
        except Exception as e:                                                  # noqa: BLE001
            out[f"clone_{tag}_error"] = str(e)[:160]
    if "clone_dram_GBps" in out and "clone_l1_GBps" in out:
        out["l1_over_dram"] = round(out["clone_l1_GBps"] / out["clone_dram_GBps"], 3)
    return out


def run_shape(dev, N, c, ks, ckc):
    import tt_bio.tenstorrent as tt

    gx, gy = tt.COMPUTE_GRID_MAIN
    ncores = gx * gy
    rows_total = N * ceil32(N)
    tbytes = rows_total * ceil32(c) * 2
    res = {"N": N, "c": c, "tensor_bytes": tbytes, "grid_cores": ncores,
           "l1_bank_bytes": tt._l1_bank_bytes(),
           "proj_flops": 2 * rows_total * ceil32(c) * ceil32(c),
           "cells": {}}
    # in + out; the [c,c] weight is 0.13 MB and is negligible against a 134.2 MB activation
    res["arithmetic_intensity_flop_per_byte"] = res["proj_flops"] / (2 * tbytes)

    xt = torch.randn(1, N, N, c, dtype=torch.bfloat16)
    x2t = torch.randn(1, N, N, c, dtype=torch.bfloat16)
    x = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    x2 = ttnn.from_torch(x2t, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    wp = ttnn.from_torch(torch.randn(c, c, dtype=torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    wg = ttnn.from_torch(torch.randn(c, c, dtype=torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    res["roofs_at_this_shape"] = roofs_at_shape(dev, x, tbytes)

    ref = None
    for k in ks:
        if N % k:
            res["cells"][f"k{k}"] = {"skipped": f"N={N} not divisible by k={k}"}
            continue
        R = N // k
        # Pre-slice: production's blocked tail recomputes the row-local layer_norm per block, so it
        # never pays a slice either. The slice is charged separately below rather than hidden.
        blocks_x = [x[:, s:s + R] for s in range(0, N, R)]
        blocks_x2 = [x2[:, s:s + R] for s in range(0, N, R)]

        m_tiles = R * (ceil32(N) // 32)
        per_core_M = -(-(-(-m_tiles // ncores)) // 5) * 5
        cores_engaged = -(-m_tiles // per_core_M)
        pc_l1 = tt._pair_proj_config(blocks_x[0], wp, bw_cap=tt._PAIR_PROJ_L1_BW, out_l1=True)
        pc_dram = tt._pair_proj_config(blocks_x[0], wp, bw_cap=tt._PAIR_PROJ_BW, out_l1=False)
        ent = {"k": k, "rows_per_block": R, "m_tiles_per_block": m_tiles,
               "per_core_M": per_core_M, "cores_engaged": cores_engaged,
               "cores_engaged_of": ncores,
               "cfg_l1": cfg_fields(pc_l1), "cfg_dram": cfg_fields(pc_dram),
               "cfg_fields_identical": cfg_fields(pc_l1) == cfg_fields(pc_dram),
               "program_launches_per_region": 3 * k + (1 if k > 1 else 0)}

        def region(pc, mc, assemble=True, spill=False):
            outs = []
            for bx, bg in zip(blocks_x, blocks_x2):
                p = ttnn.linear(bx, wp, memory_config=mc, dtype=ttnn.bfloat16,
                                compute_kernel_config=ckc, program_config=pc)
                g = ttnn.linear(bg, wg, memory_config=mc, dtype=ttnn.bfloat16,
                                compute_kernel_config=ckc, program_config=pc)
                if spill:
                    # The product lands in DRAM directly: the two L1 operands are read once and
                    # freed, so live L1 never exceeds one block's pair.
                    r = ttnn.multiply(p, g, memory_config=DRAM,
                                      input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
                    ttnn.deallocate(p)
                else:
                    r = ttnn.multiply_(p, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
                ttnn.deallocate(g)
                outs.append(r)
            if assemble:
                z = ttnn.concat(outs, dim=1, memory_config=DRAM) if len(outs) > 1 else outs[0]
                for o in outs:
                    if o is not z:
                        ttnn.deallocate(o)
                return z
            return outs

        for name, pc, mc, assemble, spill in (("l1", pc_l1, L1, True, False),
                                              ("l1_spill", pc_l1, L1, True, True),
                                              ("dram", pc_dram, DRAM, True, False),
                                              ("l1_noass", pc_l1, L1, False, False)):
            cell = {"program_config": cfg_fields(pc), "output": "L1" if mc is L1 else "DRAM",
                    "assembled": assemble, "spill_each_block_to_dram": spill,
                    "live_l1_bytes": (2 * (tbytes // k) if spill else
                                      tbytes + 2 * (tbytes // k)) if mc is L1 else 0}
            if pc is None:
                cell["error"] = "no program config for this shape"
                ent[name] = cell
                continue
            try:
                def one(pc=pc, mc=mc, assemble=assemble, spill=spill):
                    o = region(pc, mc, assemble, spill)
                    if assemble:
                        ttnn.deallocate(o)
                    else:
                        for t in o:
                            ttnn.deallocate(t)
                cell["region_ms"] = round(timed(dev, one) * 1e3, 4)
                if assemble:                                   # parity against the k=1 DRAM control
                    z = region(pc, mc, True, spill)
                    got = ttnn.to_torch(z).float()
                    ttnn.deallocate(z)
                    if ref is None and name == "dram" and k == ks[0]:
                        ref = got
                        cell["torch_equal_vs_k1_dram"] = "is the reference"
                    elif ref is not None:
                        cell["torch_equal_vs_k1_dram"] = bool(torch.equal(got, ref))
                        cell["max_abs_vs_k1_dram"] = float((got - ref).abs().max())
            except Exception as e:                                              # noqa: BLE001
                cell["error"] = f"{type(e).__name__}: {str(e)[:220]}"
            ent[name] = cell

        # what the slice itself costs, charged separately rather than hidden in the cell
        try:
            def slicing():
                for s in range(0, N, R):
                    ttnn.deallocate(ttnn.clone(x[:, s:s + R], memory_config=DRAM))
            ent["slice_clone_ms"] = round(timed(dev, slicing) * 1e3, 4)
        except Exception as e:                                                  # noqa: BLE001
            ent["slice_clone_error"] = str(e)[:160]

        res["cells"][f"k{k}"] = ent
        print(json.dumps(ent, indent=1)[:2200], flush=True)

    # ---- scored against the k=1 tuned-DRAM control, which is production today --------------------
    base = res["cells"].get(f"k{ks[0]}", {}).get("dram", {}).get("region_ms")
    if base:
        res["control_k1_dram_ms_per_region"] = base
        sc = {}
        for kk, ent in res["cells"].items():
            for name in ("l1", "l1_spill", "dram", "l1_noass"):
                ms = ent.get(name, {}).get("region_ms")
                if ms:
                    sc[f"{kk}|{name}"] = {
                        "ms_per_region": ms,
                        "saving_ms_per_region": round(base - ms, 4),
                        "ms_per_fold_over_1048_regions": round((base - ms) * 1048, 1),
                        "x_vs_control": round(base / ms, 4)}
        res["scored_vs_k1_dram_control"] = sc
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default="512:256,298:256")
    ap.add_argument("--ks", default="1,2,4,8,16")
    a = ap.parse_args()

    import importlib.metadata as im
    from tt_bio.tenstorrent import get_device, COMPUTE_GRID_MAIN
    import tt_bio.tenstorrent as tt

    dev = get_device()
    gx, gy = COMPUTE_GRID_MAIN
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    ks = [int(v) for v in a.ks.split(",")]

    out = {"host": "qb2", "chip": 0, "ttnn": im.version("ttnn"),
           "compute_grid_main": [gx, gy],
           "device_grid": [dev.compute_with_storage_grid_size().x,
                           dev.compute_with_storage_grid_size().y],
           "max_worker_l1_unreserved": int(ttnn.get_max_worker_l1_unreserved_size()),
           "l1_bank_bytes": tt._l1_bank_bytes(),
           "l1_total_MB": round(tt._l1_bank_bytes() * gx * gy / 1e6, 1),
           "flags": {"_PAIR_PROJ_BW": tt._PAIR_PROJ_BW, "_PAIR_PROJ_L1_BW": tt._PAIR_PROJ_L1_BW},
           "note": "qb2 at ttnn 0.68.0 -- every figure is a RATIO owing a qb1/0.67.4 re-take",
           "shapes": {}}
    for spec in a.shapes.split(","):
        N, c = (int(v) for v in spec.split(":"))
        print(f"--- shape N={N} c={c} ---", flush=True)
        try:
            out["shapes"][spec] = run_shape(dev, N, c, ks, ckc)
        except Exception as e:                                                  # noqa: BLE001
            out["shapes"][spec] = {"error": str(e)[:400],
                                   "traceback": traceback.format_exc()[-900:]}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
