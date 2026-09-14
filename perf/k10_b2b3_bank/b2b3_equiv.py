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

THE CODE THIS DRIVES IS NOT IN THE TREE ANY MORE. B2 and B3 both measured NEGATIVE (0.7239x and
0.9977x at the production 512 aa shape, against an A/A floor of -0.19 % to +0.04 %), so they were
reverted and the gated kernel is byte-for-byte the B1 one again. Their numbers are
`b2b3_equiv_qb2c2.json`, beside this file. To run this again, put the code back first:

    git checkout c7b737bcf -- tt_bio/reblock_permute.py tt_bio/tenstorrent.py \
        tt_bio/kernels/reblock_permute_gated/

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
# (ct_inside, pg_deep, ctg_max). `off` and `aa` are the same configuration under two names.
PRESETS = {
    "base": {"off": (False, False, None), "aa": (False, False, None),
             "b2": (True, False, None), "b3": (False, True, None), "b2b3": (True, True, None)},
    # B2's block width, one value at a time. B2 as built loses; this says whether the loss is the
    # writer's L1 gather stride and window (which scale with Ctg) or the coarser work split (which
    # does not scale the same way), and whether any width is positive.
    "ctgsweep": {"off": (False, False, None), "aa": (False, False, None),
                 "ctg2": (True, False, 2), "ctg4": (True, False, 4)},
    "all": {"off": (False, False, None), "aa": (False, False, None),
            "b3": (False, True, None), "ctg2": (True, False, 2), "ctg4": (True, False, 4)},
}
ARMS = PRESETS["base"]


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
    ap.add_argument("--arms", default="base", choices=sorted(PRESETS))
    a = ap.parse_args()
    global ARMS
    ARMS = PRESETS[a.arms]

    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    torch.manual_seed(0)
    res = {"grid": [g.x, g.y], "reps": a.reps, "roles": list(gp_roles()), "arms": a.arms,
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
            for name, (ci, pd, cm) in ARMS.items():
                RB.set_gated_bank_tuning(ct_inside=ci, pg_deep=pd, ctg_max=cm)
                o = call(dev, xw, p_off, g_off, C)
                outs[name] = ttnn.to_torch(o)
                ttnn.deallocate(o)
                # The cache holds every shape this run has touched, so qualify on the shape as
                # well as on the arm: `_cache_key_gated` is
                # (dev, N, 4C, C, N, ..., GRAN, ct_inside, pg_deep).
                key = [k for k in RB._CACHE_GATED
                       if k[1] == N and k[3] == C and k[-3:] == (ci, pd, cm)]
                assert len(key) == 1, (name, len(key))
                shapes[name] = descriptor_shape(RB._CACHE_GATED[key[0]])
                shapes[name]["ctg"] = RB._gated_ctg(C // 32, RB._gated_pg_depth())
            ref = outs["off"]
            cell["equal"] = {k: bool(torch.equal(v, ref)) for k, v in outs.items()}
            cell["max_abs_delta"] = {
                k: float((v.float() - ref.float()).abs().max()) for k, v in outs.items()}
            cell["cb_bytes"] = {k: v["cb_bytes"] for k, v in shapes.items()}
            cell["ctg"] = {k: v["ctg"] for k, v in shapes.items()}
            # the programs must actually differ where the arm says they do
            probe = "b2" if "b2" in shapes else "ctg4"
            cell["program_distinct"] = (shapes[probe]["cb_bytes"] != shapes["off"]["cb_bytes"]
                                        if C // 32 > 1 else None)
            cell["b3_program_distinct"] = (
                shapes["b3"]["cb_bytes"] != shapes["off"]["cb_bytes"] if "b3" in shapes else True)
            # live negative control: split layout, pre-B1 value offset -> a per-channel mix-up
            RB.set_gated_bank_tuning(ct_inside=False, pg_deep=False, ctg_max=None)
            ctl = ttnn.to_torch(call(dev, xw, bad_off, g_off, C))
            cell["control_differs"] = not bool(torch.equal(ctl, ref))
            del outs, ctl

            # `off` first on every rep so each arm's pair partner is one call away in time, and
            # the rest reversed on alternate reps so no arm keeps a fixed distance from it.
            names = [n for n in ARMS if n != "off"]
            for r in range(a.reps):
                for name in ["off"] + (names if r % 2 == 0 else names[::-1]):
                    ci, pd, cm = ARMS[name]
                    RB.set_gated_bank_tuning(ct_inside=ci, pg_deep=pd, ctg_max=cm)
                    us[name].append(timed(dev, xw, p_off, g_off, C))
            cell["us"] = {k: round(st.median(v), 1) for k, v in us.items()}
            cell["us_spread_pct"] = {k: round(100 * (max(v) - min(v)) / st.median(v), 1)
                                     for k, v in us.items()}
            # PAIRED per rep, then the median of the ratios. A co-tenant that slows the box for
            # part of a session shifts both arms of a rep together and drops out of the ratio; a
            # ratio of medians does not have that property, and on the first ctgsweep attempt it
            # read an A/A floor of -22.5 % where the paired form reads the real one.
            cell["ratio_paired"] = {k: round(st.median([o / v for o, v in zip(us["off"], vals)]), 4)
                                    for k, vals in us.items()}
            base = cell["us"]["off"]
            cell["ratio_unpaired"] = {k: round(base / v, 4) for k, v in cell["us"].items()}
            cell["ratio"] = cell["ratio_paired"]
            cell["aa_floor_pct"] = round(100 * (cell["ratio_paired"]["aa"] - 1.0), 3)
            cell["aa_floor_unpaired_pct"] = round(
                100 * (cell["ratio_unpaired"]["aa"] - 1.0), 3)
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
    RB.set_gated_bank_tuning(ct_inside=True, pg_deep=True, ctg_max=None)
    res["all_ok"] = bad == 0
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\n{len(res['cells']) - bad}/{len(res['cells'])} cells clean")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
