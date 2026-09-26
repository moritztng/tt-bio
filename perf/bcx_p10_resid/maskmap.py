#!/usr/bin/env python3
"""bcx-p10-resid leg 1: what `bcx-p10-devmap`'s block harness left out of a block.

devmap attributed ONE block on `perf/bcx_stack` and multiplied by 52. Its harness differs from
the shipped fold in two ways that change how much work a block does, and both are inside its
2.603 s residual by construction:

  pair masks   devmap ran `pair_masks=(None, None)`. The shipped fold at binder 146 pads 275
               residues into a 288 axis and hands both mask tensors to every pair block, which
               the triangle multiplication and both triangle attentions read.
  MSA depth    devmap ran one MSA row. The shipped multimer fold hands the Evoformer TWO
               (`msa [2, 275, 256]`, `msa_mask [2, 275]`, off the round's own event log), so
               the MSA track -- row attention, column attention, transition, outer product
               mean -- runs at twice the rows devmap timed.

This runs the same instrument over a 2x2 of those two, interleaved in one process on one card,
and differences the arms. The extra-MSA stack takes `pair` alone and has no MSA track, so it is
run under the mask arm only.

Timing, tagging and the device/enqueue subtraction are devmap's, imported rather than copied:
`device_i = synced_i - free_i - lambda * calls_i`.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "perf" / "bcx_stack")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from perf.bcx_stack import stack as S            # noqa: E402
from perf.bcx_afgrad import afgrad as A          # noqa: E402
from perf.bcx_p10_devmap import devmap as D      # noqa: E402

OUT = ROOT / "perf" / "bcx_p10_resid"


def widen_extra():
    """`Dev.extra` ignores the pair masks; the shipped stack passes them.

    `tt_bio/bindcraft2.py::_Trunk.extra_msa` runs `blk(blk._residual(z, const), *pair_masks)`,
    so the harness's mask-free call is a different amount of work. Patched BEFORE
    `install_labels`, which captures whatever `Dev.extra` names at that moment.
    """
    def extra(self, i, z, pair_masks=(None, None)):
        blk = self.dm.device_extra_msa[i]
        const = self.dm._up(self.dm.opm_constant[i].reshape(1, 1, -1))
        return blk(blk._residual(z, const), *pair_masks)
    A.Dev.extra = extra


def masks_for(dev, n, real):
    """The two pair tensors and the MSA mask the shipped fold passes at `real` of `n` tokens."""
    from tt_bio.af2 import af2_pair_masks
    seq = torch.zeros(n)
    seq[:real] = 1.0
    pair = af2_pair_masks(seq[:, None] * seq[None, :], dev.device)
    return pair, seq


def step(dev, m0, z0, wm, wz, stack, k, pair_masks, msa_mask):
    """One taped forward + one backward of K blocks, through the tagged entry points."""
    ag = dev.ag
    gc.collect()
    ml, zl = dev.leaf(m0), dev.leaf(z0)
    dev.sync()
    t0 = time.time()
    mask = dev.up(msa_mask) if msa_mask is not None else None
    with dev.tt.tape():
        if stack == "extra":
            z = zl
            for i in range(k):
                z = dev.extra(i, z, pair_masks)
            mo, zo = ml, z
        else:
            m, z = ml, zl
            for i in range(k):
                m, z = dev.evo(i, m, z, mask, pair_masks)
            mo, zo = m, z
    dev.sync()
    t1 = time.time()
    roots = [zo] if stack == "extra" else [mo, zo]
    seeds = ([dev.seed(wz, zo)] if stack == "extra"
             else [dev.seed(wm, mo), dev.seed(wz, zo)])
    dev.sync()
    t2 = time.time()
    ag.backward(roots, seeds)
    dev.sync()
    t3 = time.time()
    del mo, zo, ml, zl, roots, seeds
    return {"fwd": t1 - t0, "bwd": t3 - t2, "spans": [(t0, t1), (t2, t3)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=A.DEFAULT_PARAMS)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--n", type=int, default=288, help="the padded device axis of the fold")
    ap.add_argument("--real", type=int, default=275, help="real tokens inside that axis")
    ap.add_argument("--ks", default="1,2")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="maskmap_n288.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    import ttnn
    widen_extra()
    lv, dev, ref = S.open_all(args)
    D.install_labels()
    timer = D.OpTimer(ttnn, dev.device).install()
    clock = S.Clock()
    m1, z0, wm, wz = S.inputs(ref, args.n, args.seed)
    m2 = m1.repeat(2, 1, 1)                    # the multimer fold's two MSA rows
    wm1, wm2 = wm, wm.repeat(2, 1, 1)
    pair_on, seq = masks_for(dev, args.n, args.real)
    ks = [int(k) for k in args.ks.split(",")]

    #: (name, msa rows, pair masks on). devmap's arm is `d1_nomask`; the shipped fold is
    #: `d2_mask`. The two singles are what splits the difference between them.
    ARMS = [("d1_nomask", 1, False), ("d2_nomask", 2, False),
            ("d1_mask", 1, True), ("d2_mask", 2, True)]

    def inputs_for(depth, on):
        m = m1 if depth == 1 else m2
        w = wm1 if depth == 1 else wm2
        pm = pair_on if on else (None, None)
        # `AF2EvoformerBlock._mask_biases` takes a [rows, n] mask, one row per MSA row.
        mask = (seq.expand(depth, args.n).contiguous() if on
                else torch.ones(depth, args.n))
        return m, w, pm, mask

    blob = {"stamp": S.stamp(args, clock), "n": args.n, "real": args.real, "ks": ks,
            "reps": args.reps, "arms": [a[0] for a in ARMS],
            "loadavg_start": os.getloadavg(), "records": []}

    for name, depth, on in ARMS:                              # warm: JIT + program cache
        m, w, pm, mask = inputs_for(depth, on)
        for k in ks:
            step(dev, m, z0, w, wz, "evo", k, pm, mask)
            if on:
                step(dev, m, z0, w, wz, "extra", k, pm, mask)
    blob["sync_floor_s"] = D.sync_floor(ttnn, dev.device)
    print(json.dumps({"sync_floor_median_s": blob["sync_floor_s"]["median"]}), flush=True)

    for rep in range(args.reps):
        modes = ["free", "sync"] if rep % 2 == 0 else ["sync", "free"]
        for mode in modes:
            timer.sync = (mode == "sync")
            for name, depth, on in ARMS:
                m, w, pm, mask = inputs_for(depth, on)
                stacks = ["evo", "extra"] if on else ["evo"]
                for stack in stacks:
                    for k in ks:
                        timer.on = True
                        r = step(dev, m, z0, w, wz, stack, k, pm, mask)
                        timer.on = False
                        snap = timer.take()
                        rec = {"rep": rep, "mode": mode, "arm": name, "depth": depth,
                               "masked": on, "stack": stack, "K": k,
                               "wall_fwd": round(r["fwd"], 6), "wall_bwd": round(r["bwd"], 6),
                               "loadavg": os.getloadavg()[0],
                               "aiclk": clock.window(r["spans"]),
                               "wall": D._round(snap["wall"]), "calls": snap["calls"],
                               "read": D._round(snap["read"], 1),
                               "written": D._round(snap["written"], 1)}
                        blob["records"].append(rec)
                        print(json.dumps({"rep": rep, "mode": mode, "arm": name,
                                          "stack": stack, "K": k,
                                          "fwd": round(r["fwd"], 3),
                                          "bwd": round(r["bwd"], 3),
                                          "op_wall": round(sum(snap["wall"].values()), 3),
                                          "ops": sum(snap["calls"].values()),
                                          "aiclk": rec["aiclk"]}), flush=True)

    blob["loadavg_end"] = os.getloadavg()
    clock.stop()
    timer.uninstall()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print(f"wrote {OUT / args.out}", flush=True)


if __name__ == "__main__":
    main()
