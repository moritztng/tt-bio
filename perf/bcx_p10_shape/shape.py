#!/usr/bin/env python3
"""bcx-p10-shape: `bcx-p10-devmap`s device attribution, taken at the shape the card runs.

The campaigns device table was measured at n = 275. 275 is the shape the HOST hands across
the seam, not the shape the card executes: `bindcraft2.EvoformerOnDevice._pad` rounds the
token axis up to a multiple of 32 before `trunk.up`, and `_pad32(275) = 288`. Where the 275
comes from is worth saying because it is two buckets and not one -- BindCraft 2 pads the
DESIGN chain alone to its own 32 (`bindcraft/af2.py::padded_prediction_complex`), so a
146-residue binder becomes 160 and the 115-residue hPDL1 target is left alone: 160 + 115 =
275, which is not a multiple of 32 and gets padded again here.

Three cells, one process, one device open, so the only thing that moves between them is the
thing under test:

  A  n = 275, all-ones MSA mask, no checkpointing   `bcx-p10-devmap`s arm, the control
  B  n = 288, all-ones MSA mask, no checkpointing   the token axis, alone
  C  n = 288, the folds masks, checkpointing        the program BindCraft 2 actually runs

A -> B is the padding. B -> C is the rest of the folds machinery on the same axis. Both are
needed: a table taken straight at 288 with the harnesss own defaults would confound them.

The instrument is `perf/bcx_p10_devmap/devmap.py` unchanged -- its label context, its
`OpTimer` and its measured sync floor -- so the numbers are comparable to the table they
replace by construction rather than by assertion.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_p10_devmap import devmap as D          # noqa: E402
from perf.bcx_stack import stack as S                # noqa: E402

OUT = ROOT / "perf" / "bcx_p10_shape" / "out"

#: The public configuration `perf/bcx_p10_hostmap` stamped: binder 146 + hPDL1 target 115.
#: 146 -> 160 on BindCraft 2s own 32, + 115 = 275 across the seam, -> 288 on the card.
N_HOST, N_REAL = 275, 261


def pad_like_fold(m0, z0, n32):
    """`EvoformerOnDevice._pad`, applied to the harnesss host tensors.

    The same three `torch.nn.functional.pad` calls in the same order, so the device shape is
    the folds and the added tokens hold the folds zeros rather than fresh noise.
    """
    n = z0.shape[0]
    pad = n32 - n
    if pad == 0:
        return m0, z0
    m = torch.nn.functional.pad(m0, (0, 0, 0, pad))
    z = torch.nn.functional.pad(z0, (0, 0, 0, pad, 0, pad))
    return m, z


def cell_masks(dev, n32, n_real, depth, mode):
    """What a block is handed: the MSA mask, and `af2_pair_masks` or nothing.

    `mode="ones"` is the harness arm every PERF10 block number was taken on -- an all-ones MSA
    mask and no pair mask at all. `mode="fold"` is what a padded fold passes: `n_real` is the
    residue count rather than the host axis, because the 14 tokens BindCraft 2 added to reach
    160 are already masked before tt-bio sees them, so the card runs 288 tokens of which 261
    are real. `AF2EvoformerBlock._mask_biases` wants one mask row per MSA row, so the mask is
    `[depth, n32]` and not `[1, n32]`.
    """
    from tt_bio.af2 import af2_pair_masks
    if mode == "ones":
        return dev.up(torch.ones(depth, n32)), (None, None)
    seq = torch.zeros(n32)
    seq[:n_real] = 1.0
    pair_masks = af2_pair_masks(seq[:, None] * seq[None, :], dev.device)
    assert pair_masks[0] is not None, "the pair mask collapsed to None: n_real == n32"
    return dev.up(seq.expand(depth, n32).contiguous()), pair_masks


#: Each cell changes ONE thing from the one before it, so the chain decomposes the whole gap
#: between `bcx-p10-devmap`s harness and the program BindCraft 2 runs:
#:   A -> B  the token axis          B -> C  the folds masks and its checkpointing
#:   C -> D  the multimer folds second MSA row
#:   D -> E  the composed arm's triangle attention: `bcx-p10-tapegen`'s tape entry for
#:           `tri_att_sdpa_hifi`, which is what `bcx-p10-stack` leg 5 ran
CELLS = {
    "A": {"n": N_HOST, "pad": 0, "masks": "ones", "ckpt": False, "depth": 1, "ks": (2,)},
    "B": {"n": N_HOST, "pad": 288, "masks": "ones", "ckpt": False, "depth": 1, "ks": (2,)},
    "C": {"n": N_HOST, "pad": 288, "masks": "fold", "ckpt": True, "depth": 1, "ks": (2,)},
    "D": {"n": N_HOST, "pad": 288, "masks": "fold", "ckpt": True, "depth": 2, "ks": (2,)},
    "E": {"n": N_HOST, "pad": 288, "masks": "fold", "ckpt": True, "depth": 2, "ks": (2,),
          "hifi": True},
}


def set_hifi(on):
    """Arm or disarm the `hifi` route, the way `perf/bcx_round/run_round.py` does it.

    Three switches, and opening one without the others measures the others: the tape entry
    makes the fused kernel reachable under a tape at all, `TT_BIO_TRIATT_FUSED_HIFI` picks it
    over the materialised score path, and dividing-k is what makes 288 servable. The first and
    third are live reads so the environment carries them; `_TRIATT_FUSED_HIFI` is resolved at
    import, so it is set on the module.
    """
    from tt_bio import tenstorrent as _tn
    os.environ["TT_BIO_TAPED_KERNELS"] = "tri_att_sdpa_hifi" if on else ""
    os.environ["TT_BIO_TRIATT_DIVIDING_K"] = "1" if on else "0"
    _tn._TRIATT_FUSED_HIFI = bool(on)


def reach():
    """What the hifi route actually did, so a cell that did not fire cannot read as a result.

    `served` is the fused kernel running under a tape. `taped` counts the calls that declined
    BECAUSE they were taped, which is exactly what the tape entry removes: a hifi cell with
    taped > 0 and served == 0 measured the materialised path and its seconds mean nothing.
    """
    from tt_bio import taped_ttnn, tenstorrent
    return {"kernel_entry": {k: list(v) for k, v in taped_ttnn.KERNEL_STATS.items()},
            "fused_hifi": dict(tenstorrent.TRIATT_FUSED_HIFI_STATS)}


def cmd_cells(args):
    import ttnn
    lv, dev, ref = S.open_all(args)
    D.install_labels()
    timer = D.OpTimer(ttnn, dev.device).install()
    clock = S.Clock()

    cells = args.cells.split(",")
    blob = {"stamp": S.stamp(args, clock), "n_host": N_HOST, "n_real": N_REAL,
            "seed": args.seed, "reps": args.reps, "cells": {}, "dims": D.DIMS,
            "loadavg_start": os.getloadavg(), "started_utc": time.strftime("%FT%TZ", time.gmtime())}

    # 275 on purpose: cell A is the ragged control and every other cell is padded to
    # 288 by `pad_like_fold` below, which is the fold's own pad and not this one.
    raw_m, raw_z, _, _ = S.inputs(ref, N_HOST, args.seed, ragged=True)
    prepared = {}
    for name in cells:
        spec = CELLS[name]
        n32 = spec["pad"] or spec["n"]
        m0, z0 = pad_like_fold(raw_m, raw_z, n32)
        depth = spec.get("depth", 1)
        if depth > 1:
            m0 = m0.repeat(depth, 1, 1)          # the multimer folds extra MSA row
        torch.manual_seed(args.seed + 1)
        wm = torch.randn(m0.shape) / m0.numel() ** 0.5
        wz = torch.randn(z0.shape) / z0.numel() ** 0.5
        masks = cell_masks(dev, n32, N_REAL, depth, spec["masks"])
        prepared[name] = (m0, z0, wm, wz, masks, n32, spec)
        print(json.dumps({"cell": name, "device_axis": n32, "msa": list(m0.shape),
                          "pair": list(z0.shape), "masks": spec["masks"],
                          "ckpt": spec["ckpt"], "depth": depth}), flush=True)

    for name in cells:                 # warm every cell: JIT + program cache, untimed
        m0, z0, wm, wz, masks, n32, spec = prepared[name]
        set_hifi(spec.get("hifi", False))
        for stack in args.stacks.split(","):
            for k in spec["ks"]:
                S.block_step(dev, lv, m0, z0, wm, wz, stack, k=k, ckpt=spec["ckpt"],
                             masks=masks)
        print(json.dumps({"cell": name, "warm_reach": reach()}), flush=True)
    blob["sync_floor_s"] = D.sync_floor(ttnn, dev.device)
    print(json.dumps({"sync_floor_median_s": blob["sync_floor_s"]["median"]}), flush=True)

    # Cells are INTERLEAVED inside each rep, not run as four consecutive blocks. Run 2 ran them
    # blockwise and paid for it: cell C got a window at loadavg 14.2 while cell B had 20.7, so
    # the B -> C step carried a load difference it could not separate from its subject. Sampling
    # every cell inside each rep makes drift common-mode, which is the only form of it a median
    # over reps can remove.
    verbs = {name: [collections.Counter(), collections.Counter()] for name in cells}
    for name in cells:
        m0, z0, wm, wz, masks, n32, spec = prepared[name]
        depth = spec.get("depth", 1)
        blob["cells"][name] = {
            "n": spec["n"], "device_axis": n32, "masks": spec["masks"], "ckpt": spec["ckpt"],
            "ks": list(spec["ks"]), "reps": args.reps, "depth": depth,
            "n_real": N_REAL if spec["masks"] == "fold" else n32,
            "hifi": bool(spec.get("hifi", False)),
            "flops_fwd_analytic": D.analytic_flops(spec["n"], depth),
            "flops_fwd_analytic_padded": D.analytic_flops(n32, depth),
            "records": [], "verb": {}, "reach": {}}

    for rep in range(args.reps):
        for name in cells:
            m0, z0, wm, wz, masks, n32, spec = prepared[name]
            cell = blob["cells"][name]
            set_hifi(spec.get("hifi", False))
            verb_wall, verb_calls = verbs[name]
            modes = ["free", "sync"] if rep % 2 == 0 else ["sync", "free"]
            for mode in modes:
                timer.sync = (mode == "sync")
                for stack in args.stacks.split(","):
                    for k in spec["ks"]:
                        timer.on = True
                        r, _ = S.block_step(dev, lv, m0, z0, wm, wz, stack, k=k,
                                            ckpt=spec["ckpt"], masks=masks)
                        timer.on = False
                        snap = timer.take()
                        rec = {"rep": rep, "mode": mode, "stack": stack, "K": k,
                               "wall_fwd": round(r["fwd"], 6), "wall_bwd": round(r["bwd"], 6),
                               "bwd_cpu": round(r["bwd_cpu"], 6),
                               "loadavg": os.getloadavg()[0],
                               "aiclk": clock.window(r["spans"]),
                               "wall": D._round(snap["wall"]), "calls": snap["calls"],
                               "read": D._round(snap["read"], 1),
                               "written": D._round(snap["written"], 1),
                               "read_dtype": D._round(snap["read_dtype"], 1)}
                        if args.verb_records:
                            # `bcx-p10-calls`: the same tag with the ttnn op name appended as a
                            # second key, in BOTH modes, so `device = synced - free - lambda *
                            # calls` is available per op name and not only per family.
                            rec["verb_wall"] = D._round(snap["verb_wall"])
                            rec["verb_calls"] = snap["verb_calls"]
                            rec["verb_read"] = D._round(snap["verb_read"], 1)
                            rec["verb_written"] = D._round(snap["verb_written"], 1)
                        cell["records"].append(rec)
                        if mode == "sync" and k == max(spec["ks"]):
                            for key, v in snap["verb_wall"].items():
                                verb_wall[key] += v
                            for key, v in snap["verb_calls"].items():
                                verb_calls[key] += v
                        print(json.dumps({"cell": name, "rep": rep, "mode": mode,
                                          "stack": stack, "K": k,
                                          "fwd": round(r["fwd"], 3), "bwd": round(r["bwd"], 3),
                                          "op_wall": round(sum(snap["wall"].values()), 3),
                                          "ops": sum(snap["calls"].values()),
                                          "loadavg": round(rec["loadavg"], 1),
                                          "aiclk": rec["aiclk"]}), flush=True)
            cell["verb"] = {"wall": D._round(verb_wall), "calls": dict(verb_calls),
                            "reps": args.reps, "note": "sync mode, K=max, summed over reps"}
            cell["reach"] = reach()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
        print(f"wrote {OUT / args.out} through rep {rep}", flush=True)

    blob["loadavg_end"] = os.getloadavg()
    blob["finished_utc"] = time.strftime("%FT%TZ", time.gmtime())
    clock.stop()
    timer.uninstall()
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print(f"wrote {OUT / args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cells"])
    ap.add_argument("--params", default=None)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--cells", default="A,B,C,D")   # E is D + the hifi route
    ap.add_argument("--stacks", default="evo,extra")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="cells.json")
    ap.add_argument("--verb-records", action="store_true",
                    help="also record per-ttnn-op-name wall/calls/bytes in every record")
    args = ap.parse_args()
    if args.params is None:
        from perf.bcx_afgrad import afgrad as A
        args.params = A.DEFAULT_PARAMS
    torch.set_num_threads(args.threads)
    {"cells": cmd_cells}[args.cmd](args)


if __name__ == "__main__":
    main()
