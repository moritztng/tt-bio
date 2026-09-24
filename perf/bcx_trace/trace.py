#!/usr/bin/env python3
"""bcx-trace: how much host enqueue can trace capture take out of the taped AF2 step?

One process, one device opened WITH a trace region, the `perf/bcx_afgrad` blocks. Every unit is
run the same way in each arm: its inputs are cloned from persistent device buffers inside the
unit, so a captured trace and an eager call do identical work and the trace can be replayed
without its inputs being consumed.

Units, per block kind (`evo` = one Evoformer block, `extra` = one extra-MSA block):

  fwd    the block's untaped forward
  taped  the taped, checkpointed forward and its backward, cotangents given, grads of both
         inputs returned: exactly what `stack.py whole` does per block

Subcommands:

  census  every host-touching ttnn call a unit makes after warm-up (host-built tensors,
          readbacks), by call site, with count and wall
  floor   per unit, interleaved in one process: eager host enqueue (return without sync),
          eager synced wall, eager process CPU, and the captured trace's replay wall, which is
          the device time alone. Outputs of eager and replay compared bit for bit
  whole   the whole 4+48 checkpointed step, fwd and bwd host enqueue against synced wall

AICLK is sampled from the card's own sysfs node inside every timed window; loadavg per rep.
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
import traceback

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_afgrad import afgrad as A  # noqa: E402
from perf.bcx_stack.stack import Clock, dist, inputs, sysfs_node  # noqa: E402

OUT = ROOT / "perf" / "bcx_trace"

# ------------------------------------------------------------------------------ host census

HOSTY = ("from_torch", "to_torch", "from_device", "to_device", "as_tensor",
         "copy_host_to_device_tensor", "copy_device_to_host_tensor", "zeros", "ones", "full")


class Census:
    """Wraps ttnn's host-touching entry points before anything binds them (the tape shim
    caches an attribute on first access, so this must run before the first tape)."""

    def __init__(self, ttnn):
        self.on, self.rec = False, collections.defaultdict(lambda: [0, 0.0])
        for name in HOSTY:
            if hasattr(ttnn, name):
                setattr(ttnn, name, self._wrap(name, getattr(ttnn, name)))

    def _wrap(self, name, fn):
        def call(*a, **k):
            if not self.on:
                return fn(*a, **k)
            t0 = time.perf_counter()
            out = fn(*a, **k)
            dt = time.perf_counter() - t0
            site = "?"
            for fr in reversed(traceback.extract_stack(limit=12)[:-1]):
                if "/tt_bio/" in fr.filename or "/perf/" in fr.filename:
                    site = f"{fr.filename.split(str(ROOT) + '/')[-1]}:{fr.lineno}"
                    break
            r = self.rec[(name, site, "device" if k.get("device") is not None else "")]
            r[0] += 1
            r[1] += dt
            return out
        return call

    def take(self):
        out = [{"verb": v, "site": s, "kw": d, "calls": c, "wall_s": w}
               for (v, s, d), (c, w) in sorted(self.rec.items(), key=lambda kv: -kv[1][1])]
        self.rec.clear()
        return out


# ------------------------------------------------------------------------------ units


class Units:
    def __init__(self, dev, ref, n, seed):
        self.dev, self.ttnn, self.ag, self.tt = dev, dev.ttnn, dev.ag, dev.tt
        self.ref, self.n = ref, n
        m0, z0, wm, wz = inputs(ref, n, seed)
        self.shapes = (m0.shape, z0.shape)
        up = dev.up
        # persistent device buffers, allocated before any capture
        self.m_in, self.z_in = up(m0), up(z0)
        self.sm = self.ttnn.from_torch(wm.reshape([int(d) for d in self.m_in.shape]).to(torch.bfloat16),
                                       layout=self.ttnn.TILE_LAYOUT, device=dev.device,
                                       dtype=self.ttnn.bfloat16)
        self.sz = self.ttnn.from_torch(wz.reshape([int(d) for d in self.z_in.shape]).to(torch.bfloat16),
                                       layout=self.ttnn.TILE_LAYOUT, device=dev.device,
                                       dtype=self.ttnn.bfloat16)
        # Dev.extra uploads the OPM constant on every call (af2.py:1030 does the same in the
        # shipped stack). A trace cannot hold a host write, so the unit hoists it: one upload
        # per block, before capture, the same bytes.
        self.const = [dev.dm._up(dev.dm.opm_constant[i].reshape(1, 1, -1))
                      for i in range(len(dev.dm.device_extra_msa))]

    def feed(self, seed):
        """Write a new input set into the persistent buffers, in place (a trace's inputs)."""
        m0, z0, wm, wz = inputs(self.ref, self.n, seed)
        ttnn = self.ttnn
        for t, dst in ((m0, self.m_in), (z0, self.z_in), (wm, self.sm), (wz, self.sz)):
            h = ttnn.from_torch(t.reshape([int(d) for d in dst.shape]).to(torch.bfloat16),
                                layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            ttnn.copy_host_to_device_tensor(h, dst)

    def _extra(self, i, z):
        blk = self.dev.dm.device_extra_msa[i]
        return blk(blk._residual(z, self.ttnn.clone(self.const[i])))   # _residual owns (frees) its update

    def fwd(self, kind, i=0):
        c = self.ttnn.clone
        if kind == "evo":
            return list(self.dev.evo(i, c(self.m_in), c(self.z_in)))
        return [self._extra(i, c(self.z_in))]

    def taped(self, kind, i=0):
        ttnn, ag = self.ttnn, self.ag
        zl = ag.Tensor(ttnn.clone(self.z_in), requires_grad=True)
        if kind == "evo":
            ml = ag.Tensor(ttnn.clone(self.m_in), requires_grad=True)
            with self.tt.tape():
                mo, zo = ag.checkpoint(lambda a, b: self.dev.evo(i, a, b), ml, zl)
            ag.backward([mo, zo], [ttnn.clone(self.sm), ttnn.clone(self.sz)])
            out = [ml.grad, zl.grad]
        else:
            with self.tt.tape():
                zo = ag.checkpoint(lambda t: self._extra(i, t), zl)
            ag.backward([zo], [ttnn.clone(self.sz)])
            out = [zl.grad]
        ag.release_pins()
        return out

    def run(self, unit, kind):
        return getattr(self, unit)(kind)

    def host(self, outs):
        return [torch.Tensor(self.ttnn.to_torch(t)).clone() for t in outs]

    def free(self, outs):
        for t in outs:
            self.ttnn.deallocate(t)


def open_dev(args):
    from tt_bio import tenstorrent as T
    T.TRACE_REGIONS["bcx_trace"] = {"blackhole": args.region << 20, "wormhole_b0": args.region << 20}
    import ttnn
    census = Census(ttnn)                 # before the first tape binds ttnn's attributes
    T.get_device(trace="bcx_trace")
    dm, ref = A.load_models(args.params)
    return A.Dev(dm), ref, census


def stamp(args, clock=None):
    s = A.stamp(args.card)
    s.update({"pci": sysfs_node()[1], "aiclk_node": clock.path if clock else None,
              "trace_region_mb": args.region, "argv": sys.argv})
    return s


def save(name, blob):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(blob, indent=1, default=str))
    print(f"wrote {OUT / name}", flush=True)


def sync(dev):
    dev.ttnn.synchronize_device(dev.device)


# ------------------------------------------------------------------------------ census


def cmd_census(args):
    dev, ref, census = open_dev(args)
    U = Units(dev, ref, args.n, args.seed)
    out = {"stamp": stamp(args), "n": args.n, "units": {}}
    for kind in args.kinds.split(","):
        for unit in args.units.split(","):
            for _ in range(2):                           # warm: compiles, one-off probes
                U.free(U.run(unit, kind))
            sync(dev)
            census.on = True
            U.free(U.run(unit, kind))
            sync(dev)
            census.on = False
            rows = census.take()
            out["units"][f"{kind}.{unit}"] = rows
            print(kind, unit, json.dumps(rows, indent=None), flush=True)
    save(args.out or f"census_n{args.n}.json", out)


# ------------------------------------------------------------------------------ floor


def timed_eager(dev, U, unit, kind):
    gc.collect()
    sync(dev)
    t0, c0 = time.time(), time.process_time()
    outs = U.run(unit, kind)
    t1, c1 = time.time(), time.process_time()
    sync(dev)
    t2 = time.time()
    return outs, {"enqueue": t1 - t0, "wall": t2 - t0, "tail": t2 - t1, "cpu": c1 - c0,
                  "load1": os.getloadavg()[0], "spans": [(t0, t2)]}


def capture(dev, U, unit, kind):
    ttnn = dev.ttnn
    sync(dev)
    t0 = time.time()
    tid = ttnn.begin_trace_capture(dev.device, cq_id=0)
    try:
        outs = U.run(unit, kind)
    except Exception:
        # A trace left open wedges every later op; close it before reporting.
        try:
            ttnn.end_trace_capture(dev.device, tid, cq_id=0)
            ttnn.release_trace(dev.device, tid)
        except Exception as e2:
            print("end/release after failure raised:", e2, flush=True)
        raise
    ttnn.end_trace_capture(dev.device, tid, cq_id=0)
    sync(dev)
    return tid, outs, time.time() - t0


def timed_replay(dev, tid):
    sync(dev)
    t0, c0 = time.time(), time.process_time()
    dev.ttnn.execute_trace(dev.device, tid, cq_id=0, blocking=False)
    t1, c1 = time.time(), time.process_time()
    sync(dev)
    t2 = time.time()
    return {"enqueue": t1 - t0, "wall": t2 - t0, "tail": t2 - t1, "cpu": c1 - c0,
            "load1": os.getloadavg()[0], "spans": [(t0, t2)]}


def summ(rs, clock):
    return {k: dist([r[k] for r in rs]) for k in ("enqueue", "wall", "tail", "cpu")} | {
        "load1": [round(r["load1"], 1) for r in rs],
        "aiclk": clock.window([s for r in rs for s in r["spans"]])}


def cmd_floor(args):
    dev, ref, _ = open_dev(args)
    clock = Clock(dt=0.05)
    U = Units(dev, ref, args.n, args.seed)
    blob = {"stamp": stamp(args, clock), "n": args.n, "reps": args.reps, "units": {}}
    for kind in args.kinds.split(","):
        for unit in args.units.split(","):
            key = f"{kind}.{unit}"
            rec = blob["units"][key] = {}
            for _ in range(2):
                U.free(U.run(unit, kind))
            sync(dev)
            tt = dev.tt
            tt.DEVICE_ZEROS = False
            U.free(U.run(unit, kind))
            off = U.host(o := U.run(unit, kind))
            U.free(o)
            tt.DEVICE_ZEROS = True               # the only host write in a warm block (census)
            U.free(U.run(unit, kind))
            ref_out = U.host(o := U.run(unit, kind))
            U.free(o)
            rec["device_zeros_bit_identical"] = all(torch.equal(a, b) for a, b in zip(off, ref_out))
            try:
                tid, touts, tcap = capture(dev, U, unit, kind)
            except Exception as e:
                rec["capture"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:2000],
                                  "py_trace": "".join(traceback.format_tb(e.__traceback__))[-6000:]}
                print(key, "CAPTURE FAILED", rec["capture"]["error"][:600], flush=True)
                save(args.out or f"floor_n{args.n}.json", blob)
                if args.stop_on_fail:
                    raise
                continue
            rec["capture"] = {"ok": True, "capture_s": tcap}
            u0 = time.time()
            eager, replay, bits = [], [], []
            for rep in range(args.reps):
                for arm in (("eager", "trace") if rep % 2 == 0 else ("trace", "eager")):
                    if arm == "eager":
                        outs, r = timed_eager(dev, U, unit, kind)
                        h = U.host(outs)
                        U.free(outs)
                        eager.append(r)
                        bits.append(("eager", all(torch.equal(a, b) for a, b in zip(h, ref_out))))
                    else:
                        r = timed_replay(dev, tid)
                        h = U.host(touts)
                        replay.append(r)
                        bits.append(("trace", all(torch.equal(a, b) for a, b in zip(h, ref_out))))
            rec["eager"], rec["trace"] = summ(eager, clock), summ(replay, clock)
            rec["aiclk_unit"] = clock.window([(u0, time.time())])
            rec["bit_identical"] = {a: all(ok for arm, ok in bits if arm == a) for a in ("eager", "trace")}
            rec["vs_eager_first"] = [A.cmp(a, b) for a, b in zip(U.host(touts), ref_out)]
            # New inputs written into the persistent buffers: the trace must follow them, i.e.
            # nothing about the first inputs is baked in.
            U.feed(args.seed + 1)
            e2 = U.host(o := U.run(unit, kind))
            U.free(o)
            dev.ttnn.execute_trace(dev.device, tid, cq_id=0, blocking=True)
            t2 = U.host(touts)
            rec["fresh_input"] = {"bit_identical": all(torch.equal(a, b) for a, b in zip(t2, e2)),
                                  "differs_from_first": not all(torch.equal(a, b) for a, b in zip(t2, ref_out))}
            U.feed(args.seed)
            e, t = rec["eager"], rec["trace"]
            rec["host_removable_s"] = e["wall"]["median"] - t["wall"]["median"]
            rec["x_wall"] = e["wall"]["median"] / t["wall"]["median"]
            print(json.dumps({"unit": key, "capture_s": round(tcap, 2),
                              "eager_enq": round(e["enqueue"]["median"], 4),
                              "eager_wall": round(e["wall"]["median"], 4),
                              "eager_cpu": round(e["cpu"]["median"], 4),
                              "trace_wall": round(t["wall"]["median"], 4),
                              "trace_enq": round(t["enqueue"]["median"], 5),
                              "x": round(rec["x_wall"], 3), "bits": rec["bit_identical"],
                              "aiclk": rec["aiclk_unit"], "load1": e["load1"]}), flush=True)
            U.free(touts)
            dev.ttnn.release_trace(dev.device, tid)
            save(args.out or f"floor_n{args.n}.json", blob)
    clock.stop()


# ------------------------------------------------------------------------------ whole


def cmd_whole(args):
    """The whole 4+48 checkpointed step as `stack.py whole` runs it, from device leaves.

    eager: host enqueue of the taped forward and of the backward, each against its synced wall.
    trace (`--trace`): the same step captured once and replayed, interleaved with eager in this
    process, grads compared bit for bit, then replayed on a fresh input set against eager."""
    dev, ref, _ = open_dev(args)
    dev.tt.DEVICE_ZEROS = args.trace or args.device_zeros
    clock = Clock(dt=0.1)
    ag, ttnn = dev.ag, dev.ttnn
    U = Units(dev, ref, args.n, args.seed)
    ke, kv = args.extra, args.evo

    def fwd():
        ml = ag.Tensor(ttnn.clone(U.m_in), requires_grad=True)
        zl = ag.Tensor(ttnn.clone(U.z_in), requires_grad=True)
        with dev.tt.tape():
            z = zl
            for i in range(ke):
                z = ag.checkpoint(lambda t, i=i: U._extra(i, t), z)
            m = ml
            for i in range(kv):
                m, z = ag.checkpoint(lambda a, b, i=i: dev.evo(i, a, b), m, z)
        return ml, zl, m, z

    def bwd(ml, zl, m, z):
        ag.backward([m, z], [ttnn.clone(U.sm), ttnn.clone(U.sz)])
        out = [ml.grad, zl.grad]
        ag.release_pins()
        return out

    def eager():
        gc.collect()
        sync(dev)
        t0, c0 = time.time(), time.process_time()
        st = fwd()
        t1, c1 = time.time(), time.process_time()
        sync(dev)
        t2, c2 = time.time(), time.process_time()
        outs = bwd(*st)
        t3, c3 = time.time(), time.process_time()
        sync(dev)
        t4 = time.time()
        del st
        g = U.host(outs)
        U.free(outs)
        gc.collect()
        return g, {"fwd_enqueue": t1 - t0, "fwd_wall": t2 - t0, "fwd_cpu": c1 - c0,
                   "bwd_enqueue": t3 - t2, "bwd_wall": t4 - t2, "bwd_cpu": c3 - c2,
                   "step": t4 - t0, "cpu": (c1 - c0) + (c3 - c2),
                   "load1": os.getloadavg()[0], "spans": [(t0, t4)]}

    blob = {"stamp": stamp(args, clock), "n": args.n, "k_extra": ke, "k_evo": kv, "reps": args.reps,
            "device_zeros": dev.tt.DEVICE_ZEROS, "arms": {}}
    eager()
    eager()
    tid = None
    if args.trace:
        sync(dev)
        t0 = time.time()
        tid = ttnn.begin_trace_capture(dev.device, cq_id=0)
        try:
            touts = bwd(*fwd())
        finally:
            ttnn.end_trace_capture(dev.device, tid, cq_id=0)
        sync(dev)
        blob["capture_s"] = time.time() - t0
        print("captured in", round(blob["capture_s"], 2), "s", flush=True)
    runs = {"eager": [], "trace": []}
    grads = {"eager": [], "trace": []}
    for rep in range(args.reps):
        for arm in (["eager", "trace"] if rep % 2 == 0 else ["trace", "eager"]):
            if arm == "trace" and tid is None:
                continue
            if arm == "eager":
                g, r = eager()
            else:
                r = timed_replay(dev, tid)
                r["step"], r["cpu"] = r["wall"], r["cpu"]
                g = U.host(touts)
            runs[arm].append(r)
            grads[arm].append(g)
            print(arm, rep, {k: round(v, 3) for k, v in r.items() if k != "spans"}, flush=True)
    for arm, rs in runs.items():
        if not rs:
            continue
        keys = [k for k in rs[0] if k not in ("spans", "load1")]
        blob["arms"][arm] = {
            "per": {k: dist([r[k] for r in rs]) for k in keys},
            "load1": [round(r["load1"], 1) for r in rs],
            "aiclk": clock.window([s for r in rs for s in r["spans"]]),
            "reps_bit_identical": all(torch.equal(a, b) for g in grads[arm][1:]
                                      for a, b in zip(g, grads[arm][0]))}
    if tid is not None:
        e, t = blob["arms"]["eager"], blob["arms"]["trace"]
        blob["trace_bit_identical_to_eager"] = all(torch.equal(a, b) for a, b in
                                                   zip(grads["trace"][0], grads["eager"][0]))
        blob["x_step"] = e["per"]["step"]["median"] / t["per"]["step"]["median"]
        blob["host_removable_s"] = e["per"]["step"]["median"] - t["per"]["step"]["median"]
        U.feed(args.seed + 1)
        g2, _ = eager()
        ttnn.execute_trace(dev.device, tid, cq_id=0, blocking=True)
        t2 = U.host(touts)
        blob["fresh_input"] = {"bit_identical": all(torch.equal(a, b) for a, b in zip(t2, g2)),
                               "differs_from_first": not all(torch.equal(a, b) for a, b in
                                                             zip(t2, grads["eager"][0]))}
        ttnn.release_trace(dev.device, tid)
    print(json.dumps({arm: {k: round(v["median"], 3) for k, v in a["per"].items()}
                      for arm, a in blob["arms"].items()}),
          {k: blob.get(k) for k in ("x_step", "host_removable_s", "trace_bit_identical_to_eager",
                                    "fresh_input")},
          {arm: a["aiclk"] for arm, a in blob["arms"].items()}, flush=True)
    clock.stop()
    save(args.out or f"whole_n{args.n}_e{ke}_v{kv}.json", blob)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["census", "floor", "whole"])
    p.add_argument("--params", default=os.path.expanduser("~/.boltz/af2/params/params_model_1_ptm.npz"))
    p.add_argument("--card", default=os.environ.get("TT_VISIBLE_DEVICES", "0"))
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kinds", default="evo,extra")
    p.add_argument("--units", default="fwd,taped")
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--extra", type=int, default=4)
    p.add_argument("--evo", type=int, default=48)
    p.add_argument("--region", type=int, default=512, help="trace region, MB")
    p.add_argument("--stop-on-fail", action="store_true")
    p.add_argument("--trace", action="store_true", help="whole: capture the step, replay interleaved")
    p.add_argument("--device-zeros", action="store_true", help="whole: taped_ttnn.DEVICE_ZEROS on")
    p.add_argument("--out")
    args = p.parse_args()
    {"census": cmd_census, "floor": cmd_floor, "whole": cmd_whole}[args.cmd](args)


if __name__ == "__main__":
    main()
