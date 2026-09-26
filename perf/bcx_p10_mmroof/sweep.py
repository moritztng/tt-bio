#!/usr/bin/env python3
"""bcx-p10-mmroof leg 3: sweep the PLAN, per class, on the model's own captured calls.

`perf/bcx_p10_mmlay/micro.py` swept placement and the core grid and stopped where a call already
had a `program_config`, because its lever stopped there. This sweeps the config itself:
`per_core_M`, `per_core_N`, `in0_block_w`, `out_subblock_h/w`, `fuse_batch`, the compute grid,
and the factory. Everything else is `micro.py` unchanged and deliberately so -- the capture, the
replay, the byte model, the grading and the warm-rep median are the same code, so a number here
is comparable to a number there.

The baseline is the call the model makes, program config and all. A sweep point is that call
with ONE field of its config replaced, which means the comparison is a plan against a plan and
not a plan against a reconstruction.

`in0_block_w` is the one field that is not a pure reassociation of the same work: it is the
K-blocking, and the K-blocking decides how the fp32 partial sums group. Every point is graded
against the baseline's own answer (PCC and max absolute deviation), and a point that moves the
answer is reported with the deviation rather than dropped -- the campaign's bar is accuracy
against a 0.60 A kill bar, not bit-exactness, and a number without its deviation cannot be
checked against that bar.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_p10_devmap import devmap as D          # noqa: E402
from perf.bcx_p10_shape import shape as SH           # noqa: E402
from perf.bcx_stack import stack as S                # noqa: E402
from perf.bcx_p10_mmlay import micro as MI           # noqa: E402
from perf.bcx_p10_mmroof import plankey as PK        # noqa: E402
from perf.bcx_p10_mmroof import cells as CE          # noqa: E402

OUT = ROOT / "perf" / "bcx_p10_mmroof" / "out"
TILE = 32


class CaptureTimer(MI.CaptureTimer, PK.PlanOpTimer):
    """`micro.CaptureTimer`'s banking with `PlanOpTimer`'s key.

    The MRO matters and is the point of writing it out: `CaptureTimer.verb_key` calls `super()`,
    which lands on `PlanOpTimer.verb_key`, which lands on `ShapeOpTimer`'s. So the banked call
    is keyed by shape AND plan, and two calls of one shape running two different configs are
    two separate baselines instead of whichever one ran first.
    """


def fields_of(pc) -> dict:
    """Every constructor field a live program config carries, by its real attribute name."""
    out = {}
    for attr, _ in PK._FIELDS:
        if hasattr(pc, attr):
            try:
                out[attr] = getattr(pc, attr)
            except Exception:                                                 # noqa: BLE001
                pass
    return out


def rebuild(pc, **override):
    """`pc` with some fields replaced, or None if this factory will not take them.

    Constructed by keyword off the live object's own fields, so a factory that does not carry
    `out_block_h` simply never sees it and a factory that does keeps whatever it had.
    """
    f = fields_of(pc)
    f.update(override)
    return type(pc)(**f)


def _divisors(n, lo=1, hi=None):
    hi = n if hi is None else hi
    return [d for d in range(lo, hi + 1) if n % d == 0]


def _tiles(x):
    return -(-int(x) // TILE)


def points(ttnn, dev, p, pc, dest_tiles):
    """(label, transform) pairs for one class, derived from its OWN plan and shape.

    Nothing here is a fixed table: the legal values of `per_core_M` are the divisors of this
    class's `Mt`, the legal `out_subblock_h * out_subblock_w` is what fits the dest register
    file this class's compute kernel config leaves it, and both are read off the captured call.
    """
    P = [("baseline: the call the model makes", lambda a, k: (a, k))]
    if pc is None:
        # No plan at all. bcx-p10-mmlay owns these and already measured them; the only thing
        # worth carrying here is the grid it chose, as the reference point for the class.
        for y, x in ((8, 8), (11, 10)):
            def g(a, k, _y=y, _x=x):
                k = {kk: v for kk, v in k.items() if kk not in ("program_config", "config")}
                k["core_grid"] = ttnn.CoreGrid(y=_y, x=_x)
                return a, k
            P.append(("core_grid %dx%d (mmlay's lever)" % (y, x), g))
        return P

    f = fields_of(pc)
    Mt, Nt, Kt = _tiles(p["M"]), _tiles(p["N"]), _tiles(p["K"])
    if f.get("fuse_batch"):
        Mt *= max(1, p["batch"])
    slot = "program_config" if "program_config" in () else None               # resolved by caller

    def mk(label, **ov):
        def f2(a, k, _ov=ov):
            k = dict(k)
            key = "program_config" if k.get("program_config") is not None else "config"
            new = rebuild(k[key], **_ov)
            k[key] = new
            return a, k
        P.append((label, f2))

    # per_core_M: how many M-tiles one core owns. Smaller spreads the work over more cores and
    # re-reads in1 more often; larger amortises the write barrier. The shipped rule in
    # tenstorrent.py takes the SMALLEST legal split and its own docstring says a per-shape
    # sweep beats it by 13-18 % at two of four classes. This is that sweep.
    cur_m = f.get("per_core_M")
    if isinstance(cur_m, int) and cur_m > 0:
        for m in _divisors(Mt):
            if m == cur_m:
                continue
            sh = max(h for h in range(min(dest_tiles, m), 0, -1) if m % h == 0)
            sw = max(w for w in range(min(dest_tiles // sh, f.get("out_subblock_w", 1) or 1), 0, -1))
            ov = {"per_core_M": m, "out_subblock_h": sh}
            if "out_block_h" in f:
                ov["out_block_h"] = m
            mk("per_core_M %d -> %d (subblock h %d)" % (cur_m, m, sh), **ov)

    # out_subblock_h/w: the output block one pass of the packer writes. 1x1 pays a write
    # barrier per output tile. The product is capped by the dest register file, which
    # fp32_dest_acc_en halves to 4 tiles -- that cap is READ off the class's own kernel config.
    cur_h, cur_w = f.get("out_subblock_h"), f.get("out_subblock_w")
    if isinstance(cur_h, int) and isinstance(cur_w, int):
        blk_h = f.get("out_block_h", cur_m) or cur_m or 1
        blk_w = f.get("out_block_w", f.get("per_core_N", Nt)) or Nt
        seen = {(cur_h, cur_w)}
        for h in _divisors(int(blk_h)):
            for w in _divisors(int(blk_w)):
                if h * w > dest_tiles or (h, w) in seen:
                    continue
                seen.add((h, w))
                mk("out_subblock %dx%d -> %dx%d" % (cur_h, cur_w, h, w),
                   out_subblock_h=h, out_subblock_w=w)

    # in0_block_w: the K-blocking. The ONE field that regroups the fp32 accumulation, so every
    # point here is expected to move the answer and is graded for how far.
    cur_w0 = f.get("in0_block_w")
    if isinstance(cur_w0, int) and cur_w0 > 0:
        for w in _divisors(Kt):
            if w != cur_w0 and w <= 16:
                mk("in0_block_w %d -> %d (regroups the K accumulation)" % (cur_w0, w),
                   in0_block_w=w)

    # fuse_batch: fold the batch into M so the whole batch goes on the grid at once instead of
    # running batch sequential passes inside it. Only meaningful when in1 is not batched.
    if "fuse_batch" in f:
        mk("fuse_batch %s -> %s" % (bool(f["fuse_batch"]), not f["fuse_batch"]),
           fuse_batch=not f["fuse_batch"])

    # the compute grid itself
    cur_g = f.get("compute_with_storage_grid_size")
    if cur_g is not None:
        for gx, gy in ((11, 10), (8, 8), (11, 5), (6, 6)):
            try:
                if (int(getattr(cur_g, "x", cur_g[0])), int(getattr(cur_g, "y", cur_g[1]))) == (gx, gy):
                    continue
            except Exception:                                                 # noqa: BLE001
                pass
            mk("grid -> %dx%d" % (gx, gy), compute_with_storage_grid_size=(gx, gy))

    # and the control: no plan at all, so the class can be compared against what ttnn picks.
    def nop(a, k):
        k = {kk: v for kk, v in k.items() if kk not in ("program_config", "config")}
        return a, k
    P.append(("plan dropped: ttnn's own planner", nop))
    return P


def dest_tiles_of(ck) -> int:
    """8 tiles of dest, or 4 when `fp32_dest_acc_en` halves it. Read, not assumed."""
    try:
        return 4 if bool(getattr(ck, "fp32_dest_acc_en", False)) else 8
    except Exception:                                                         # noqa: BLE001
        return 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", required=True, help="roof.py's json, for the class ranking")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--only", default=None)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--dram", type=float, default=442.3)
    ap.add_argument("--tflops", type=float, default=85.90)
    ap.add_argument("--params", default=None)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--triatt-bw", dest="triatt_bw", type=int, default=1)
    ap.add_argument("--rne-kernel", dest="rne_kernel", type=int, default=1)
    ap.add_argument("--mm-layout", dest="mm_layout", type=int, default=1)
    ap.add_argument("--out", default="sweep_E.json")
    args = ap.parse_args()
    if args.params is None:
        from perf.bcx_afgrad import afgrad as A
        args.params = A.DEFAULT_PARAMS
    torch.set_num_threads(args.threads)
    args.stacks, args.reps_cells = "evo,extra", 1

    armed = CE.arm(args.triatt_bw, args.rne_kernel, 1, args.mm_layout)
    print("ARMED %s" % json.dumps(armed), flush=True)

    import ttnn
    rank = json.load(open(args.rank))
    want = [r["key"] for r in rank["classes"]]
    if args.only:
        import re
        want = [k for k in want if re.search(args.only, k)]
    want = want[:args.top]
    by_key = {r["key"]: r for r in rank["classes"]}

    lv, dev, ref = S.open_all(args)
    D.install_labels()
    timer = CaptureTimer(ttnn, dev.device).install()
    clock = S.Clock()

    spec = SH.CELLS["E"]
    raw_m, raw_z, _, _ = S.inputs(ref, SH.N_HOST, args.seed, ragged=True)
    m0, z0 = SH.pad_like_fold(raw_m, raw_z, 288)
    m0 = m0.repeat(spec["depth"], 1, 1)
    torch.manual_seed(args.seed + 1)
    wm = torch.randn(m0.shape) / m0.numel() ** 0.5
    wz = torch.randn(z0.shape) / z0.numel() ** 0.5
    masks = SH.cell_masks(dev, 288, SH.N_REAL, spec["depth"], spec["masks"])
    SH.set_hifi(True)
    timer.sync = False
    for stack in ("evo", "extra"):
        timer.on = True
        S.block_step(dev, lv, m0, z0, wm, wz, stack, k=2, ckpt=spec["ckpt"], masks=masks)
        timer.on = False
    timer.take()
    timer.uninstall()
    print("captured %d classes from a warm block step" % len(timer.repro), flush=True)

    blob = {"stamp": S.stamp(args, clock), "armed": armed, "reps": args.reps,
            "roofs": {"dram": args.dram, "tflops": args.tflops},
            "started_utc": time.strftime("%FT%TZ", time.gmtime()), "classes": {}}
    missing = [k for k in want if k not in timer.repro]
    if missing:
        print("NOT CAPTURED (skipped): %d of %d" % (len(missing), len(want)), flush=True)
    for key in want:
        if key not in timer.repro:
            continue
        p = PK.parse(key)
        row = by_key[key]
        print("\n== %s %s %s | round %.3f s over %d calls | %.1f %% dram, %.1f %% compute, %s cores, binds %s"
              % (p["op"], p["shape"], p["flags"], row["device_s"], round(row["calls"]),
                 row["pct_dram"], row["pct_compute"], row["cores"], row["binds"]), flush=True)
        print("   plan %s | ck %s" % (p["plan"], p["ck"]), flush=True)
        print("%-44s %10s %8s %6s %8s %7s  %s"
              % ("point", "ms", "GB/s", "%dram", "TFLOP/s", "vs base", "against the baseline's answer"),
              flush=True)

        rec = timer.repro[key]
        live_pc = live_ck = None
        for kind, slot, v in rec:
            if kind == "kw" and v[0] == "raw":
                if slot in ("program_config", "config"):
                    live_pc = v[1]
                elif slot == "compute_kernel_config":
                    live_ck = v[1]
        pts = points(ttnn, dev.device, p, live_pc, dest_tiles_of(live_ck))

        results, base, refans = [], None, None
        for label, tf in pts:
            try:
                a, k = MI.build(ttnn, dev.device, rec, seed=args.seed)
                a, k = tf(a, k)
                sec, nb, host = MI.timed(ttnn, dev.device, p["op"], a, k, args.reps)
            except Exception as e:                                            # noqa: BLE001
                msg = str(e).split("\n")[0][:64]
                print("%-44s %10s  %s" % (label, "REFUSED", msg), flush=True)
                results.append({"point": label, "error": msg})
                continue
            fl = p["flop"]
            gbs, tfs = nb / sec / 1e9, fl / sec / 1e12
            if base is None:
                base, refans = sec, host
            g = MI.grade(refans, host)
            r = {"point": label, "s": sec, "bytes": nb, "GBs": gbs, "TFLOPs": tfs,
                 "pct_dram": 100 * gbs / args.dram, "pct_compute": 100 * tfs / args.tflops,
                 "vs_base": base / sec, "grade": g,
                 "aiclk": clock.window([(time.time() - 1.0, time.time())]),
                 "loadavg": os.getloadavg()[0]}
            results.append(r)
            print("%-44s %10.4f %8.1f %5.1f%% %8.2f %6.4fx  %s"
                  % (label, sec * 1e3, gbs, r["pct_dram"], tfs, r["vs_base"],
                     "" if g is None else "pcc %.6f maxdev %.2e" % (g["pcc"], g["max_abs_dev"])),
                  flush=True)
        ok = [r for r in results if "s" in r]
        best = max(ok, key=lambda r: r["vs_base"], default=None)
        if best:
            print("   BEST %s at %.4fx -> the round row %.3f s becomes %.3f s, saving %.3f s"
                  % (best["point"], best["vs_base"], row["device_s"],
                     row["device_s"] / best["vs_base"],
                     row["device_s"] - row["device_s"] / best["vs_base"]), flush=True)
        blob["classes"][key] = {"round_row": row, "points": results}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))

    blob["finished_utc"] = time.strftime("%FT%TZ", time.gmtime())
    blob["loadavg_end"] = os.getloadavg()
    clock.stop()
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print("\nwrote", OUT / args.out, flush=True)


if __name__ == "__main__":
    main()
