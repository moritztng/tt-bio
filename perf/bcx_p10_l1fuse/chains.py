#!/usr/bin/env python3
"""bcx-p10-l1fuse leg 1: the composed round's producer -> consumer edges, per block.

Cell E is the composed round's own cell: 288 token axis, the fold's masks, checkpointing on,
extra-MSA depth 2, the hifi route, and the three wave-10/11 levers armed the way
`perf/bcx_round/run_round.py` arms them. `perf/bcx_p10_mmroof/cells.py` established that
arming and this reuses it rather than restating it.

The measurement is the campaign's own K difference. One block's edges are `edges(K=2) -
edges(K=1)`, so everything outside the blocks cancels, and the same difference is taken on the
per-verb wall in both timer modes so device seconds come out of

    device = synced - free - lambda * calls

on exactly the records that produced the edges. Modes are interleaved inside one process and
the order is flipped between reps: a per-verb number compared ACROSS runs is void on this box
(`bcx-p10-trilay` saw a 6x swing on one op at loadavg 10-19), and the edge table is per-verb.
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
from perf.bcx_stack import stack as S                # noqa: E402
from perf.bcx_p10_devmap import devmap as D          # noqa: E402
from perf.bcx_p10_shape import shape as SH           # noqa: E402
from perf.bcx_p10_mmroof import cells as CE          # noqa: E402
from perf.bcx_p10_l1fuse import chainkey as CK       # noqa: E402

OUT = ROOT / "perf" / "bcx_p10_l1fuse" / "out"


def _snap_counters(timer):
    """The four edge counters as plain dicts, and then cleared."""
    out = {"edges": dict(timer.edges), "edge_bytes": dict(timer.edge_bytes),
           "edge_gap": dict(timer.edge_gap), "edge_adj": dict(timer.edge_adj),
           "consumers": dict(timer.consumers),
           "verb_wall": dict(timer.verb_wall), "verb_calls": dict(timer.verb_calls),
           "verb_read": dict(timer.verb_read), "verb_written": dict(timer.verb_written)}
    for c in (timer.edges, timer.edge_bytes, timer.edge_gap, timer.edge_adj, timer.consumers,
              timer.wall, timer.calls, timer.read, timer.written, timer.read_dtype,
              timer.verb_wall, timer.verb_calls, timer.verb_read, timer.verb_written):
        c.clear()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--stacks", default="evo,extra")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="chains_E.json")
    ap.add_argument("--triatt-bw", dest="triatt_bw", type=int, default=1)
    ap.add_argument("--rne-kernel", dest="rne_kernel", type=int, default=1)
    ap.add_argument("--hifi", type=int, default=1)
    ap.add_argument("--mm-layout", dest="mm_layout", type=int, default=1)
    args = ap.parse_args()
    if args.params is None:
        from perf.bcx_afgrad import afgrad as A
        args.params = A.DEFAULT_PARAMS
    torch.set_num_threads(args.threads)

    armed = CE.arm(args.triatt_bw, args.rne_kernel, args.hifi, args.mm_layout)
    print("ARMED %s" % json.dumps(armed), flush=True)

    import ttnn
    lv, dev, ref = S.open_all(args)
    D.install_labels()
    timer = CK.ChainOpTimer(ttnn, dev.device).install()
    clock = S.Clock()

    spec = SH.CELLS["E"]
    n32 = spec["pad"]
    raw_m, raw_z, _, _ = S.inputs(ref, SH.N_HOST, args.seed, ragged=True)
    m0, z0 = SH.pad_like_fold(raw_m, raw_z, n32)
    depth = spec["depth"]
    m0 = m0.repeat(depth, 1, 1)
    torch.manual_seed(args.seed + 1)
    wm = torch.randn(m0.shape) / m0.numel() ** 0.5
    wz = torch.randn(z0.shape) / z0.numel() ** 0.5
    masks = SH.cell_masks(dev, n32, SH.N_REAL, depth, spec["masks"])
    SH.set_hifi(True)

    blob = {"stamp": S.stamp(args, clock), "cell": "E", "device_axis": n32, "depth": depth,
            "armed": armed, "seed": args.seed, "reps": args.reps, "ks": [1, 2],
            "stacks": args.stacks.split(","), "records": [],
            "loadavg_start": os.getloadavg(),
            "started_utc": time.strftime("%FT%TZ", time.gmtime())}

    for stack in args.stacks.split(","):               # warm: JIT + program cache, untimed
        for k in (1, 2):
            S.block_step(dev, lv, m0, z0, wm, wz, stack, k=k, ckpt=spec["ckpt"], masks=masks)
    blob["sync_floor_s"] = D.sync_floor(ttnn, dev.device)
    print(json.dumps({"sync_floor_median_s": blob["sync_floor_s"]["median"],
                      "calls_warm": timer.seq}), flush=True)
    _snap_counters(timer)

    for rep in range(args.reps):
        for mode in (["free", "sync"] if rep % 2 == 0 else ["sync", "free"]):
            timer.sync = (mode == "sync")
            for stack in args.stacks.split(","):
                for k in (1, 2):
                    timer.on = True
                    r, _ = S.block_step(dev, lv, m0, z0, wm, wz, stack, k=k,
                                        ckpt=spec["ckpt"], masks=masks)
                    timer.on = False
                    snap = _snap_counters(timer)
                    rec = {"rep": rep, "mode": mode, "stack": stack, "K": k,
                           "wall_fwd": round(r["fwd"], 6), "wall_bwd": round(r["bwd"], 6),
                           "loadavg": os.getloadavg()[0], "aiclk": clock.window(r["spans"])}
                    rec.update({key: {"||".join(kk): vv for kk, vv in val.items()}
                                for key, val in snap.items()})
                    blob["records"].append(rec)
                    print(json.dumps({"rep": rep, "mode": mode, "stack": stack, "K": k,
                                      "fwd": round(r["fwd"], 3), "bwd": round(r["bwd"], 3),
                                      "edges": len(snap["edges"]),
                                      "edge_calls": sum(snap["edges"].values()),
                                      "loadavg": round(rec["loadavg"], 1),
                                      "aiclk": rec["aiclk"]}), flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
        print(f"wrote {OUT / args.out} through rep {rep}", flush=True)

    from tt_bio import mm_layout as _mm
    from tt_bio import triatt_bw as _tbw
    blob["reach"] = {"mm_layout": _mm.reach(), "triatt_bw": dict(_tbw.STATS)}
    blob["loadavg_end"] = os.getloadavg()
    blob["finished_utc"] = time.strftime("%FT%TZ", time.gmtime())
    clock.stop()
    timer.uninstall()
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print(f"wrote {OUT / args.out}", flush=True)
    print("REACH %s" % json.dumps(blob["reach"], default=str), flush=True)


if __name__ == "__main__":
    main()
