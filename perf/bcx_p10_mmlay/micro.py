#!/usr/bin/env python3
"""bcx-p10-mmlay leg 2: sweep what is configurable on the round's top matmul shapes.

The trap this instrument is built to avoid: a micro-benchmark that reconstructs the model's
matmul from a shape in a table is not measuring the model's matmul. `ttnn.linear` at
288x288x128x128 with the program config `tt_bio` passes and `ttnn.linear` at the same shape
with ttnn's default are different programs, and only the first is the baseline a lever has to
beat. So the baseline is CAPTURED, not reconstructed: `CaptureTimer` records the first call of
every shape key during a real warm block step -- the operand metadata AND the live
`program_config` / `compute_kernel_config` objects -- and every sweep point replays that exact
call with one thing changed.

Bytes are the census byte model (operand + result padded bytes), so a GB/s here is the same
quantity as a GB/s in leg 1's table and the two can be compared. A point that moves an operand
to L1 still pays its census bytes: the number then answers "did the card go faster", not "did
DRAM traffic fall", and those are different questions.
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
from perf.bcx_p10_mmlay import mmkey as MK           # noqa: E402

OUT = ROOT / "perf" / "bcx_p10_mmlay" / "out"


class CaptureTimer(MK.ShapeOpTimer):
    """`ShapeOpTimer` that also banks one replayable call per shape key."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.repro = {}

    def verb_key(self, path, args, kwargs, out):
        key = super().verb_key(path, args, kwargs, out)
        if "#" in key and key not in self.repro:
            T = self.ttnn.Tensor
            meta = []
            for i, x in enumerate(args):
                meta.append(("pos", i, _meta(x, T)))
            for k, v in kwargs.items():
                meta.append(("kw", k, _meta(v, T)))
            self.repro[key] = meta
        return key


def _meta(x, T):
    """A tensor becomes a rebuild recipe; anything else is passed through as the live object."""
    if not isinstance(x, T):
        return ("raw", x)
    return ("tensor", {"shape": [int(d) for d in x.padded_shape],
                       "dtype": x.dtype, "layout": x.layout,
                       "memory_config": x.memory_config()})


def build(ttnn, dev, rec, seed=0):
    """Rebuild a captured call's arguments: fresh tensors with the captured placement."""
    torch.manual_seed(seed)
    args, kwargs = {}, {}
    for kind, slot, v in rec:
        if v[0] == "raw":
            val = v[1]
        else:
            m = v[1]
            t = torch.randn(m["shape"]) * 0.05
            val = ttnn.from_torch(t, dtype=m["dtype"], layout=m["layout"],
                                  device=dev, memory_config=m["memory_config"])
        (args if kind == "pos" else kwargs)[slot] = val
    return [args[i] for i in sorted(args)], kwargs


def census_bytes(ttnn, args, kwargs, out):
    tot = 0.0
    for x in list(args) + list(kwargs.values()):
        if isinstance(x, ttnn.Tensor):
            tot += D._nbytes(x)
    o = out
    if isinstance(o, (list, tuple)):
        tot += sum(D._nbytes(t) for t in o if isinstance(t, ttnn.Tensor))
    elif isinstance(o, ttnn.Tensor):
        tot += D._nbytes(o)
    return tot


def call(ttnn, path, args, kwargs):
    fn = ttnn
    for part in path.split("."):
        fn = getattr(fn, part)
    return fn(*args, **kwargs)


def timed(ttnn, dev, path, args, kwargs, reps):
    """Median synced wall over `reps`, warm. Returns (seconds, bytes, result) or raises.

    The result comes back as host tensors so every sweep point can be graded against the
    baseline's answer. A core grid changes how the matmul is parallelised, which changes the
    order the partial products accumulate in, so "it went faster" is not a result until the
    number it produced is the same number.
    """
    out = call(ttnn, path, args, kwargs)
    ttnn.synchronize_device(dev)
    nb = census_bytes(ttnn, args, kwargs, out)
    host = [ttnn.to_torch(t) for t in (out if isinstance(out, (list, tuple)) else [out])
            if isinstance(t, ttnn.Tensor)]
    _dealloc(ttnn, out)
    xs = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = call(ttnn, path, args, kwargs)
        ttnn.synchronize_device(dev)
        xs.append(time.perf_counter() - t0)
        _dealloc(ttnn, out)
    xs.sort()
    return xs[len(xs) // 2], nb, host


def grade(ref, got):
    """Max absolute deviation and PCC of a sweep point's answer against the baseline's."""
    if ref is None or len(ref) != len(got):
        return None
    worst, pcc = 0.0, 1.0
    for a, b in zip(ref, got):
        a, b = a.float().flatten(), b.float().flatten()
        worst = max(worst, float((a - b).abs().max()))
        va, vb = a - a.mean(), b - b.mean()
        den = float(va.norm() * vb.norm())
        if den > 0:
            pcc = min(pcc, float((va * vb).sum()) / den)
    return {"max_abs_dev": worst, "pcc": pcc}


def _dealloc(ttnn, out):
    for t in (out if isinstance(out, (list, tuple)) else [out]):
        if isinstance(t, ttnn.Tensor):
            try:
                ttnn.deallocate(t)
            except Exception:
                pass


# --------------------------------------------------------------- the sweep points

def _mc(ttnn, name):
    return {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}[name]


def points(ttnn, dev):
    """(label, transform) pairs. A transform gets (args, kwargs) and mutates placement."""
    P = [("baseline: the call the model makes", lambda a, k: (a, k))]

    def move(which, where):
        def f(a, k):
            a = list(a)
            tgt = _mc(ttnn, where)
            idx = [i for i, x in enumerate(a) if isinstance(x, ttnn.Tensor)]
            kw_t = [key for key, x in k.items() if isinstance(x, ttnn.Tensor)]
            order = [("pos", i) for i in idx] + [("kw", key) for key in kw_t]
            pick = {"a": 0, "b": 1}[which]
            if pick >= len(order):
                raise ValueError("no operand %s" % which)
            kind, slot = order[pick]
            if kind == "pos":
                a[slot] = ttnn.to_memory_config(a[slot], tgt)
            else:
                k = dict(k)
                k[slot] = ttnn.to_memory_config(k[slot], tgt)
            return a, k
        return f

    P.append(("operand a -> L1 interleaved", move("a", "l1")))
    P.append(("operand b -> L1 interleaved", move("b", "l1")))

    def both_l1(a, k):
        a, k = move("a", "l1")(a, k)
        return move("b", "l1")(a, k)
    P.append(("both operands -> L1 interleaved", both_l1))

    def out_l1(a, k):
        k = dict(k)
        k["memory_config"] = ttnn.L1_MEMORY_CONFIG
        return a, k
    P.append(("output -> L1 interleaved", out_l1))

    def all_l1(a, k):
        a, k = both_l1(a, k)
        return out_l1(a, k)
    P.append(("a, b and output -> L1", all_l1))

    def drop_pc(a, k):
        k = {key: v for key, v in k.items() if key not in ("program_config", "config")}
        if "program_config" not in k and "config" not in k:
            return a, k
        raise ValueError("no program config to drop")
    P.append(("program config dropped: ttnn picks", drop_pc))

    for y, x in ((10, 13), (10, 8), (8, 13), (8, 8), (7, 7), (6, 6), (5, 5), (4, 4), (2, 2)):
        def grid(a, k, _y=y, _x=x):
            k = {key: v for key, v in k.items() if key not in ("program_config", "config")}
            k["core_grid"] = ttnn.CoreGrid(y=_y, x=_x)
            return a, k
        P.append(("core_grid %dx%d, no program config" % (y, x), grid))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", required=True, help="leg 1 report json, for the shape ranking")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--only", default=None,
                    help="regex on the shape key: sweep just these shapes")
    ap.add_argument("--points", default=None,
                    help="substring filter on the sweep point labels")
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--dram", type=float, default=442.3)
    ap.add_argument("--tflops", type=float, default=85.90)
    ap.add_argument("--params", default=None)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="micro_E.json")
    args = ap.parse_args()
    if args.params is None:
        from perf.bcx_afgrad import afgrad as A
        args.params = A.DEFAULT_PARAMS
    torch.set_num_threads(args.threads)
    args.stacks, args.reps_cells = "evo,extra", 1

    import ttnn
    rank = json.load(open(args.rank))
    want = [r["key"] for r in rank["shapes"] if "#" in r["key"]]
    if args.only:
        import re
        want = [k for k in want if re.search(args.only, k)]
    want = want[:args.top]
    by_key = {r["key"]: r for r in rank["shapes"]}

    lv, dev, ref = S.open_all(args)
    D.install_labels()
    timer = CaptureTimer(ttnn, dev.device).install()
    clock = S.Clock()

    # One warm block step per stack with the capture on: this is where the baselines come from.
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
    print("captured %d shape keys from a warm block step" % len(timer.repro), flush=True)

    blob = {"stamp": S.stamp(args, clock), "roofs": {"dram": args.dram, "tflops": args.tflops},
            "reps": args.reps, "started_utc": time.strftime("%FT%TZ", time.gmtime()),
            "shapes": {}}
    missing = [k for k in want if k not in timer.repro]
    if missing:
        print("NOT CAPTURED (skipped): %s" % missing, flush=True)
    for key in want:
        if key not in timer.repro:
            continue
        p = MK.parse(key)
        row = by_key[key]
        print("\n== %s  %s %s | round: %.3f s over %d calls, %.1f %% of DRAM roof, %.1f %% of compute =="
              % (p["op"], p["shape"], p["flags"], row["device_s"], round(row["calls"]),
                 row["pct_dram"], row["pct_compute"]), flush=True)
        print("   a %s | b %s | o %s" % (row["a"], row["b"], row["o"]), flush=True)
        print("%-40s %10s %9s %8s %7s %9s %6s %8s  %s"
              % ("point", "ms", "GB", "GB/s", "%dram", "TFLOP/s", "%cmp", "vs base",
                 "against the baseline's answer"), flush=True)
        results, base, ref = [], None, None
        pts = points(ttnn, dev.device)
        if args.points:
            pts = [(l, f) for l, f in pts if args.points in l or l.startswith("baseline")]
        for label, tf in pts:
            try:
                a, k = build(ttnn, dev.device, timer.repro[key], seed=args.seed)
                a, k = tf(a, k)
                sec, nb, host = timed(ttnn, dev.device, p["op"], a, k, args.reps)
            except Exception as e:
                msg = str(e).split("\n")[0][:70]
                print("%-40s %10s  %s" % (label, "UNSUPPORTED", msg), flush=True)
                results.append({"point": label, "error": msg})
                continue
            fl = p["flop"]
            gbs, tf_s = nb / sec / 1e9, fl / sec / 1e12
            if base is None:
                base, ref = sec, host
            g = grade(ref, host)
            aiclk = clock.window([(time.time() - 1.0, time.time())])
            r = {"point": label, "s": sec, "bytes": nb, "GBs": gbs, "TFLOPs": tf_s,
                 "pct_dram": 100 * gbs / args.dram, "pct_compute": 100 * tf_s / args.tflops,
                 "vs_base": base / sec, "aiclk": aiclk, "grade": g}
            results.append(r)
            print("%-40s %10.4f %9.4f %8.1f %6.1f%% %9.2f %5.1f%% %7.4fx  %s"
                  % (label, sec * 1e3, nb / 1e9, gbs, r["pct_dram"], tf_s,
                     r["pct_compute"], r["vs_base"],
                     "" if g is None else "pcc %.6f maxdev %.2e" % (g["pcc"], g["max_abs_dev"])),
                  flush=True)
        best = max((r for r in results if "s" in r), key=lambda r: r["vs_base"], default=None)
        if best:
            print("   BEST: %s at %.4fx  (the round row is %.3f s -> %.3f s if it transferred)"
                  % (best["point"], best["vs_base"], row["device_s"],
                     row["device_s"] / best["vs_base"]), flush=True)
        blob["shapes"][key] = {"round_row": row, "points": results,
                               "aiclk": clock.window([(time.time() - 2.0, time.time())])}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))

    blob["finished_utc"] = time.strftime("%FT%TZ", time.gmtime())
    blob["loadavg_end"] = os.getloadavg()
    clock.stop()
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print("\nwrote", OUT / args.out, flush=True)


if __name__ == "__main__":
    main()
