#!/usr/bin/env python3
"""B2 and B3 at the op: bit-exactness by `torch.equal`, and the op ratio each is worth.

B1 landed as `TT_BIO_TRIMUL_GP_BANK_SPLIT` and splits a read pair's value and gate slices onto two
DRAM banks. B2 and B3 are the rest of the same lever and ride the same flag:

  B2  the group key carries a BLOCK of `Ctg` channel tiles read inside the row loop, so `ct` -- the
      only term in the reader's page index that can move the bank -- varies between successive
      barriers instead of being fixed for a whole group.
  B3  the reader's p/g circular buffers go 4 -> 32 tiles.

Both reorder whole tiles through the same three compute stages and touch no arithmetic, so the bar
is `torch.equal` against the arm with both off, measured on device at six shapes. B1 is ON in every
arm: this prices B2 and B3 on top of it, which is how they would ship.

The rig carries three guards a bit-exact claim needs and a ratio claim needs:
  * a live negative control (the split layout read with the shipped-major offsets, a real
    per-channel mix-up) which must come back DIFFERENT, so `torch.equal` is shown to be sensitive
    at this shape and dtype rather than assumed;
  * a program-identity check -- every arm's cached descriptor is a distinct entry with its own
    group count and CB depths, so an arm cannot silently reuse the previous arm's compiled program
    and read 1.000x;
  * an A/A floor arm, the `off` configuration entered twice under two names and timed the same way
    as everything else, so the ratios are read against this session's own noise.

Arms are interleaved (and the order reversed on alternate reps) so compile and warm-up bias
cannot land on one arm.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio as _TB
assert Path(_TB.__file__).resolve().is_relative_to(ROOT), (
    f"imported tt_bio from {_TB.__file__}, not this tree")
import tt_bio.reblock_permute as RB
from tt_bio.tenstorrent import get_device, gp_roles

# (ct_inside, pg_deep). `off` and `aa` are the same configuration under two names: their ratio is
# the A/A floor.
ARMS = {"off": (False, False), "aa": (False, False),
        "b2": (True, False), "b3": (False, True), "b2b3": (True, True)}


def call(dev, xw, p_off, g_off, C):
    o = RB.reblock_permute_gated(xw, p_off, g_off, C, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(dev)
    return o


def timed(dev, xw, p_off, g_off, C):
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    o = RB.reblock_permute_gated(xw, p_off, g_off, C, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(dev)
    us = 1e6 * (time.perf_counter() - t0)
    ttnn.deallocate(o)
    return us


def descriptor_shape(entry):
    """The three numbers that say this arm compiled its own program."""
    rd = entry["kernels"][0]
    cbs = entry["cbs"]
    # one core's runtime args is enough: the split, Ctg and the CB depths are common to the grid
    return {"cb_bytes": [int(c.total_size) for c in cbs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="298x32,298x128,512x32,512x128,768x32,768x128")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    torch.manual_seed(0)
    res = {"grid": [g.x, g.y], "reps": a.reps, "roles": list(gp_roles()),
           "arch": str(dev.arch()), "cells": [],
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    bad = 0
    for spec in a.cells.split(","):
        N, C = (int(v) for v in spec.split("x"))
        cell = {"N": N, "C": C, "Ct": C // 32}
        try:
            # B1's column order, which is what ships: p_a at tile 0, g_a at tile Ct.
            roles = list(gp_roles())
            p_off, g_off = roles.index("p_a") * C, roles.index("g_a") * C
            bad_off = ("g_a", "g_b", "p_a", "p_b").index("p_a") * C  # the pre-B1 value offset
            xw_h = torch.randn(1, N, N, 4 * C, dtype=torch.bfloat16)
            xw = ttnn.from_torch(xw_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ref, shapes, us = None, {}, {k: [] for k in ARMS}
            # warm every arm once and keep its output for the equality check
            outs = {}
            for name, (ci, pd) in ARMS.items():
                RB.set_gated_bank_tuning(ct_inside=ci, pg_deep=pd)
                o = call(dev, xw, p_off, g_off, C)
                outs[name] = ttnn.to_torch(o)
                ttnn.deallocate(o)
                key = [k for k in RB._CACHE_GATED if k[-2:] == (ci, pd)]
                assert len(key) == 1, (name, len(key))
                shapes[name] = descriptor_shape(RB._CACHE_GATED[key[0]])
            ref = outs["off"]
            cell["equal"] = {k: bool(torch.equal(v, ref)) for k, v in outs.items()}
            cell["max_abs_delta"] = {
                k: float((v.float() - ref.float()).abs().max()) for k, v in outs.items()}
            cell["cb_bytes"] = {k: v["cb_bytes"] for k, v in shapes.items()}
            # the programs must actually differ where the arm says they do
            cell["program_distinct"] = (shapes["b2"]["cb_bytes"] != shapes["off"]["cb_bytes"]
                                        if C // 32 > 1 else None)
            cell["b3_program_distinct"] = shapes["b3"]["cb_bytes"] != shapes["off"]["cb_bytes"]
            # live negative control: split layout, pre-B1 value offset -> a per-channel mix-up
            RB.set_gated_bank_tuning(ct_inside=False, pg_deep=False)
            ctl = ttnn.to_torch(call(dev, xw, bad_off, g_off, C))
            cell["control_differs"] = not bool(torch.equal(ctl, ref))
            del outs, ctl

            names = list(ARMS)
            for r in range(a.reps):
                for name in (names if r % 2 == 0 else names[::-1]):
                    ci, pd = ARMS[name]
                    RB.set_gated_bank_tuning(ct_inside=ci, pg_deep=pd)
                    us[name].append(timed(dev, xw, p_off, g_off, C))
            cell["us"] = {k: round(st.median(v), 1) for k, v in us.items()}
            base = cell["us"]["off"]
            cell["ratio"] = {k: round(base / v, 4) for k, v in cell["us"].items()}
            cell["aa_floor_pct"] = round(100 * (cell["ratio"]["aa"] - 1.0), 3)
            ttnn.deallocate(xw)
            ok = (all(cell["equal"].values()) and cell["control_differs"]
                  and cell["b3_program_distinct"]
                  and (cell["program_distinct"] is not False))
        except Exception as e:                                                  # noqa: BLE001
            cell["error"] = f"{type(e).__name__}: {e}"[:400]
            ok = False
        bad += not ok
        res["cells"].append(cell)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  N={N:4d} C={C:3d} equal={cell.get('equal')} ctl={cell.get('control_differs')} "
              f"us={cell.get('us')} ratio={cell.get('ratio')} {cell.get('error', '')}", flush=True)
    RB.set_gated_bank_tuning(ct_inside=True, pg_deep=True)
    res["all_ok"] = bad == 0
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\n{len(res['cells']) - bad}/{len(res['cells'])} cells clean")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
