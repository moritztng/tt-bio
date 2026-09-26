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


def fold_masks(dev, n32, n_real):
    """The two things a padded fold hands every block: the MSA mask and `af2_pair_masks`.

    `n_real` is the residue count, not the host axis: the 14 tokens BindCraft 2 added to reach
    160 are masked out before tt-bio ever sees them, so the card runs 288 tokens of which 261
    are real. The device program is the same either way -- the mask is a tensor of a fixed
    shape -- but writing the real count down is free and keeps the arm honest.
    """
    from tt_bio.af2 import af2_pair_masks
    seq = torch.zeros(n32)
    seq[:n_real] = 1.0
    msa_mask = dev.up(seq.reshape(1, n32))
    pair_masks = af2_pair_masks(seq[:, None] * seq[None, :], dev.device)
    assert pair_masks[0] is not None, "the pair mask collapsed to None: n_real == n32"
    return msa_mask, pair_masks


CELLS = {
    "A": {"n": N_HOST, "pad": 0, "masks": "ones", "ckpt": False, "ks": (2,)},
    "B": {"n": N_HOST, "pad": 288, "masks": "ones", "ckpt": False, "ks": (2,)},
    "C": {"n": N_HOST, "pad": 288, "masks": "fold", "ckpt": True, "ks": (2,)},
}

# ------------------------------------------------------------------ the route arms

# `bcx-p10-bytes` adds one dimension to the same cells: WHICH triangle-attention program runs.
# Cells A-C hold the route fixed at the shipped `_fp32_softmax_attention` and move the shape;
# these hold the shape fixed at 288 and move the route, so the two tables subtract. Every arm
# names every switch, so a cell can never inherit the arm that ran before it -- an arm that
# silently kept the previous route is the failure a byte census cannot tell from a null result.
MAT = {"TT_BIO_TAPED_KERNELS": "", "TT_BIO_TRIATT_FUSED_HIFI": "0",
       "TT_BIO_TRIATT_DIVIDING_K": "0", "TT_BIO_TRIATT_TAPED_SDPA": "0",
       "TT_BIO_SDPA_OWN_FORWARD": "0"}
#: `bcx-p10-triatt`s shippable arm: the taped call takes the stock fused SDPA verb and
#: `autograd.triangle_attention` computes its own chunked forward and the backward.
AGTRI = dict(MAT, TT_BIO_TRIATT_TAPED_SDPA="1", TT_BIO_SDPA_OWN_FORWARD="1")
#: `bcx-p10-tapegen`s arm: the persistent-mask fused kernel serves the forward through its tape
#: entry. Both gates, because opening one without the other measures the other.
HIFI = dict(MAT, TT_BIO_TAPED_KERNELS="tri_att_sdpa_hifi", TT_BIO_TRIATT_FUSED_HIFI="1",
            TT_BIO_TRIATT_DIVIDING_K="1")

for _name, _env in (("mat", MAT), ("agtri", AGTRI), ("hifi", HIFI)):
    CELLS[_name] = {"n": N_HOST, "pad": 288, "masks": "ones", "ckpt": False, "ks": (2,),
                    "env": _env}


def route(spec):
    """Put the process on this cell's triangle-attention route.

    Every switch below is a LIVE read -- `_triatt_taped_sdpa_on`, `_sdpa_own_forward`,
    `enabled_kernels` at tape open, `_triatt_hifi_dividing_k` -- except the fused-HiFi flag,
    which `tenstorrent` resolves at import, so that one is set on the module the way
    `perf/bcx_p10_tapegen/round_ab.py` sets it.
    """
    env = spec.get("env")
    if not env:
        return
    os.environ.update(env)
    from tt_bio import tenstorrent as tn
    tn._TRIATT_FUSED_HIFI = env["TT_BIO_TRIATT_FUSED_HIFI"] == "1"


def reach():
    """Every counter that says which route a cell actually took."""
    from tt_bio import tenstorrent as tn
    from tt_bio import taped_ttnn as TT
    return {"fp32_softmax_calls": tn.FP32_SOFTMAX_STATS["calls"],
            "fused_hifi": dict(tn.TRIATT_FUSED_HIFI_STATS),
            "taped_sdpa": dict(tn.TRIATT_TAPED_SDPA_STATS),
            "own_forward": dict(TT.SDPA_OWN_FORWARD_STATS),
            "entries": {k: list(v) for k, v in TT.KERNEL_STATS.items()},
            "picks": {str(k): v for k, v in tn.TRIATT_FUSED_HIFI_PICKS.items()}}


def reach_delta(before, after):
    """`after - before`, so a cell reports what IT served rather than the process total.

    `picks` is a map of shape to plan, not a count, so it is taken as it stands.
    """
    def sub(a, b):
        if isinstance(a, list):
            return [x - y for x, y in zip(a, b or [0] * len(a))]
        if isinstance(a, dict):
            return {k: sub(v, (b or {}).get(k, 0)) for k, v in a.items()}
        return a - (b or 0)
    return {k: v if k == "picks" else sub(v, before.get(k)) for k, v in after.items()}


def reach_add(total, delta):
    """Accumulate one turn's reach delta onto a cell's running total."""
    if total is None:
        return delta

    def add(a, b):
        if isinstance(a, list):
            return [x + y for x, y in zip(a, b)]
        if isinstance(a, dict):
            return {k: add(v, b.get(k, 0 if not isinstance(v, (list, dict)) else type(v)()))
                    for k, v in a.items()}
        return a + b
    return {k: v if k == "picks" else add(v, total.get(k, 0)) for k, v in delta.items()}


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

    raw_m, raw_z, _, _ = S.inputs(ref, N_HOST, args.seed)
    prepared = {}
    for name in cells:
        spec = CELLS[name]
        n32 = spec["pad"] or spec["n"]
        m0, z0 = pad_like_fold(raw_m, raw_z, n32)
        torch.manual_seed(args.seed + 1)
        wm = torch.randn(m0.shape) / m0.numel() ** 0.5
        wz = torch.randn(z0.shape) / z0.numel() ** 0.5
        masks = fold_masks(dev, n32, N_REAL) if spec["masks"] == "fold" else None
        prepared[name] = (m0, z0, wm, wz, masks, n32, spec)
        print(json.dumps({"cell": name, "device_axis": n32, "msa": list(m0.shape),
                          "pair": list(z0.shape), "masks": spec["masks"],
                          "ckpt": spec["ckpt"]}), flush=True)

    lv.mask = True                     # the design path always hands an MSA mask
    for name in cells:                 # warm every cell: JIT + program cache, untimed
        m0, z0, wm, wz, masks, n32, spec = prepared[name]
        route(spec)
        for stack in args.stacks.split(","):
            for k in spec["ks"]:
                S.block_step(dev, lv, m0, z0, wm, wz, stack, k=k, ckpt=spec["ckpt"],
                             masks=masks)
    blob["sync_floor_s"] = D.sync_floor(ttnn, dev.device)
    print(json.dumps({"sync_floor_median_s": blob["sync_floor_s"]["median"]}), flush=True)

    # One cell dict per cell, then the REPS outermost: the cells take turns instead of
    # running one after the other, so a drift in the box lands on every arm equally. Cells
    # that differ only in shape do not need this; arms that differ in route do, because the
    # whole result is a ratio between them.
    tally = {}
    for name in cells:
        spec = prepared[name][6]
        tally[name] = [collections.Counter(), collections.Counter()]
        blob["cells"][name] = {
            "n": spec["n"], "device_axis": prepared[name][5], "masks": spec["masks"],
            "ckpt": spec["ckpt"], "ks": list(spec["ks"]), "reps": args.reps,
            "n_real": N_REAL if spec["masks"] == "fold" else prepared[name][5],
            "flops_fwd_analytic": D.analytic_flops(spec["n"], args.depth),
            "flops_fwd_analytic_padded": D.analytic_flops(prepared[name][5], args.depth),
            "env": dict(spec.get("env") or {}), "records": [], "verb": {}}

    for rep in range(args.reps):
        for name in cells:
            m0, z0, wm, wz, masks, n32, spec = prepared[name]
            cell = blob["cells"][name]
            verb_wall, verb_calls = tally[name]
            route(spec)
            # Snapped around THIS cell's turn. The arms interleave, so a baseline taken once
            # per cell would charge it with every serve the other arms made in between.
            reach_before = reach()
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
                                          "aiclk": rec["aiclk"]}), flush=True)
            cell["reach"] = reach_add(cell.get("reach"),
                                      reach_delta(reach_before, reach()))
            cell["verb"] = {"wall": D._round(verb_wall), "calls": dict(verb_calls),
                            "reps": rep + 1, "note": "sync mode, K=max, summed over reps"}
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
    ap.add_argument("--cells", default="A,B,C")
    ap.add_argument("--depth", type=int, default=1, help="MSA rows; see the doc on depth 2")
    ap.add_argument("--stacks", default="evo,extra")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="cells.json")
    args = ap.parse_args()
    if args.params is None:
        from perf.bcx_afgrad import afgrad as A
        args.params = A.DEFAULT_PARAMS
    torch.set_num_threads(args.threads)
    {"cells": cmd_cells}[args.cmd](args)


if __name__ == "__main__":
    main()
