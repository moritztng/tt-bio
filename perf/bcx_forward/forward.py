#!/usr/bin/env python3
"""bcx-forward: what does the AF2 forward cost in a BindCraft 2 gradient step, and where does it go?

`bcx-stack` won 11.9x on the Evoformer backward and `bcx-realcensus` censused that backward.
Neither touched the forward, which a BC2 gradient step pays TWICE: once under the tape and once
as the stop-gradient recycle pass (`bindcraft/af2.py:132-138`, `design_recycles` = 1).

Subcommands, one device open each, on the card TT_VISIBLE_DEVICES names:

  step     the whole 4+48 trunk at one n: untaped forward (the recycle pass), taped forward,
           and the backward, in ONE process, interleaved, AICLK sampled during every window
  blocks   marginal forward seconds per real Evoformer and extra-MSA block from a K sweep,
           untaped and taped, so a_f and b_f are slopes and not one-point reads
  census   every ttnn op one untaped block forward issues, with its shapes: host enqueue time
           (unsynced pass) and serialized time (synced pass), against roofs measured in the
           same process at the same clock
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import pathlib
import sys
import threading
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_afgrad import afgrad as A  # noqa: E402
from perf.bcx_stack.stack import Clock, Levers, dist, sysfs_node  # noqa: E402

OUT = ROOT / "perf" / "bcx_forward"


def stamp(args, clock=None):
    s = A.stamp(args.card)
    s["pci"] = sysfs_node()[1]
    s["aiclk_node"] = clock.path if clock else None
    return s


def save(name, blob):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(blob, indent=1, default=str))
    print(f"wrote {OUT / name}", flush=True)


def inputs(ref, n, seed):
    torch.manual_seed(seed)
    m0, z0 = A.embed(ref["bf16"], torch.randn(n, 20) * 2.0, torch.arange(n))
    m0, z0 = m0.detach(), z0.detach()
    return (m0, z0, torch.randn(m0.shape) / m0.numel() ** 0.5,
            torch.randn(z0.shape) / z0.numel() ** 0.5)


def open_all(args, arm="stack"):
    lv = Levers()
    lv.arm(arm)
    dm, ref = A.load_models(args.params)
    return lv, A.Dev(dm), ref


# ------------------------------------------------------------------------------ step / blocks


def untaped_fwd(dev, m0, z0, ke, kv):
    """The stop-gradient recycle pass: the same blocks, no tape, no checkpoint, no grad."""
    gc.collect()
    ml, zl = dev.up(m0), dev.up(z0)
    dev.sync()
    t0 = time.time()
    m, z = dev.stack(ml, zl, ke, kv)              # raw ttnn tensors: no tape, no wrapper
    dev.sync()
    t1 = time.time()
    del m, z, ml, zl
    return t0, t1


def taped_step(dev, lv, m0, z0, wm, wz, ke, kv, ckpt=True, backward=True):
    gc.collect()
    ag = dev.ag
    ml, zl = dev.leaf(m0), dev.leaf(z0)
    dev.sync()
    lv.phase = "fwd"
    t0 = time.time()
    with dev.tt.tape():
        mo, zo = dev.stack(ml, zl, ke, kv, ckpt=ckpt)
    dev.sync()
    t1 = time.time()
    out = {"taped_fwd": t1 - t0, "spans": [(t0, t1)]}
    if backward:
        seeds = [dev.seed(wm, mo), dev.seed(wz, zo)]
        dev.sync()
        lv.phase = "bwd"
        t2, c2 = time.time(), time.process_time()
        ag.backward([mo, zo], seeds)
        dev.sync()
        t3, c3 = time.time(), time.process_time()
        out.update(bwd=t3 - t2, bwd_cpu=c3 - c2)
        out["spans"].append((t2, t3))
        del seeds
    if ckpt:
        ag.release_pins()
    del mo, zo, ml, zl
    lv.phase = "fwd"
    gc.collect()
    return out


def cmd_step(args):
    lv, dev, ref = open_all(args, args.arm)
    clock = Clock()
    ke, kv = args.extra, args.evo
    blob = {"stamp": stamp(args, clock), "arm": args.arm, "k_extra": ke, "k_evo": kv,
            "reps": args.reps, "backward": not args.no_backward, "points": []}
    for n in [int(x) for x in args.ns.split(",")]:
        m0, z0, wm, wz = inputs(ref, n, args.seed)
        untaped_fwd(dev, m0, z0, ke, kv)                      # warm: JIT + program cache
        rec = {"recycle_fwd": [], "taped_fwd": [], "bwd": [], "bwd_cpu": []}
        spans = []
        t_start = time.time()
        for rep in range(args.reps):
            t0, t1 = untaped_fwd(dev, m0, z0, ke, kv)
            rec["recycle_fwd"].append(t1 - t0)
            spans.append((t0, t1))
            r = taped_step(dev, lv, m0, z0, wm, wz, ke, kv, ckpt=not args.no_ckpt,
                           backward=not args.no_backward)
            rec["taped_fwd"].append(r["taped_fwd"])
            if "bwd" in r:
                rec["bwd"].append(r["bwd"])
                rec["bwd_cpu"].append(r["bwd_cpu"])
            spans += r["spans"]
            print(n, rep, {k: round(v[-1], 4) for k, v in rec.items() if v}, flush=True)
        pt = {"n": n, "loadavg": os.getloadavg(), "wall": [t_start, time.time()],
              "aiclk": clock.window(spans),
              "per": {k: dist(v) for k, v in rec.items() if v}}
        p = pt["per"]
        pt["forward_seconds_per_gradient_step"] = (p["recycle_fwd"]["median"]
                                                   + p["taped_fwd"]["median"])
        if "bwd" in p:
            tot = pt["forward_seconds_per_gradient_step"] + p["bwd"]["median"]
            pt["trunk_step_seconds"] = tot
            pt["forward_share_of_trunk"] = pt["forward_seconds_per_gradient_step"] / tot
        blob["points"].append(pt)
        print(json.dumps(pt, default=str), flush=True)
        save(args.out or f"step_n{'_'.join(args.ns.split(','))}.json", blob)
    clock.stop()


def cmd_blocks(args):
    """Marginal forward seconds per block: K=1,2,4 untaped and taped, slope by least squares."""
    lv, dev, ref = open_all(args, args.arm)
    clock = Clock()
    ks = [int(x) for x in args.ks.split(",")]
    blob = {"stamp": stamp(args, clock), "arm": args.arm, "ks": ks, "steps": args.steps,
            "points": []}
    for n in [int(x) for x in args.ns.split(",")]:
        m0, z0, wm, wz = inputs(ref, n, args.seed)
        for stack_name in args.stacks.split(","):
            rec = {k: {"untaped": [], "taped": []} for k in ks}
            spans = []
            for k in ks:                                      # warm each shape
                ke, kv = (k, 0) if stack_name == "extra" else (0, k)
                untaped_fwd(dev, m0, z0, ke, kv)
                taped_step(dev, lv, m0, z0, wm, wz, ke, kv, ckpt=args.ckpt, backward=False)
            for step in range(args.steps):
                for k in (ks if step % 2 == 0 else ks[::-1]):
                    ke, kv = (k, 0) if stack_name == "extra" else (0, k)
                    t0, t1 = untaped_fwd(dev, m0, z0, ke, kv)
                    rec[k]["untaped"].append(t1 - t0)
                    spans.append((t0, t1))
                    r = taped_step(dev, lv, m0, z0, wm, wz, ke, kv, ckpt=args.ckpt,
                                   backward=False)
                    rec[k]["taped"].append(r["taped_fwd"])
                    spans += r["spans"]
            pt = {"n": n, "stack": stack_name, "ckpt": args.ckpt, "loadavg": os.getloadavg(),
                  "aiclk": clock.window(spans),
                  "per_k": {k: {m: dist(v[m]) for m in v} for k, v in rec.items()}}
            for mode in ("untaped", "taped"):
                y = np.array([np.median(rec[k][mode]) for k in ks], float)
                slope, intercept = np.polyfit(np.array(ks, float), y, 1)
                pt[mode] = {"per_block": float(slope), "intercept": float(intercept)}
            print(json.dumps(pt, default=str), flush=True)
            blob["points"].append(pt)
            save(args.out or f"blocks_n{'_'.join(args.ns.split(','))}.json", blob)
    clock.stop()


# ------------------------------------------------------------------------------ census


SKIP = {"deallocate", "synchronize_device", "from_torch", "to_torch", "from_device",
        "to_device", "close_device", "open_device", "get_memory_view", "buffer_address",
        "get_max_worker_l1_unreserved_size", "create_sharded_memory_config", "Tensor",
        "release_trace", "copy_host_to_device_tensor"}

def op_names(mod):
    """Every registered ttnn operation on a module, by TYPE rather than by a hand list.

    The first cut of this census named the ops it expected from a grep of `tt_bio`, and a fused
    kernel it had not named enqueued 3.9 ms that the sync of the NEXT op then charged to an
    `unsqueeze` of 2 MB. An unpatched op does not go missing, it lands on its successor."""
    import ttnn
    t = type(ttnn.matmul)
    return [n for n in dir(mod)
            if n not in SKIP and not n.startswith("_") and isinstance(getattr(mod, n, None), t)]


def _shape(t):
    v = getattr(t, "value", t)
    try:
        return [int(d) for d in v.shape]
    except Exception:
        return None


def _dtype(t):
    v = getattr(t, "value", t)
    return str(getattr(v, "dtype", "")).replace("DataType.", "")


def _bytes(t):
    s, d = _shape(t), _dtype(t)
    if not s:
        return 0
    w = {"BFLOAT16": 2, "FLOAT32": 4, "UINT32": 4, "INT32": 4, "UINT16": 2, "UINT8": 1,
         "BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625}.get(d, 2)
    n = 1
    for x in s:
        n *= x
    return n * w


class OpRecorder:
    """Wrap every ttnn entry point the trunk uses. `sync=False` records the host enqueue cost
    and leaves the pipeline alone; `sync=True` serializes, so each op's reading is its own
    dispatch plus its own kernel with an empty queue in front of it."""

    def __init__(self, device, sync):
        import ttnn
        self.ttnn, self.device, self.sync = ttnn, device, sync
        self.rows = collections.defaultdict(lambda: {"calls": 0, "t": 0.0, "in_bytes": 0.0,
                                                     "out_bytes": 0.0})
        self.order = []
        self.on = False
        self._saved = []
        for name in op_names(ttnn):
            self._patch(ttnn, name, name)
        for sub in ("experimental", "transformer", "operations"):
            m = getattr(ttnn, sub, None)
            if m is None:
                continue
            for name in op_names(m):
                self._patch(m, name, f"{sub}.{name}")

    def _patch(self, mod, attr, label):
        real = getattr(mod, attr, None)
        if real is None or not callable(real):
            return
        self._saved.append((mod, attr, real))
        rec = self

        def wrapped(*a, _real=real, _label=label, **k):
            if not rec.on:
                return _real(*a, **k)
            ins = [x for x in list(a) + list(k.values()) if _shape(x)]
            t0 = time.perf_counter()
            out = _real(*a, **k)
            if rec.sync:
                rec.ttnn.synchronize_device(rec.device)
            t1 = time.perf_counter()
            outs = out if isinstance(out, (tuple, list)) else [out]
            key = (_label, "|".join(f"{_shape(x)}{_dtype(x)[:3]}" for x in ins),
                   "|".join(str(_shape(x)) for x in outs if _shape(x)))
            r = rec.rows[key]
            r["calls"] += 1
            r["t"] += t1 - t0
            r["in_bytes"] += sum(_bytes(x) for x in ins)
            r["out_bytes"] += sum(_bytes(x) for x in outs if _shape(x))
            return out

        setattr(mod, attr, wrapped)

    def start(self):
        self.rows.clear()
        self.on = True

    def stop(self):
        self.on = False
        return {" ".join(k): dict(v) for k, v in
                sorted(self.rows.items(), key=lambda kv: -kv[1]["t"])}


def roofs(dev, clock):
    """Same process, same clock, this card: a matmul roof and a copy/DRAM roof."""
    import ttnn
    d = dev.device
    out = {}
    t = ttnn.from_torch(torch.randn(4096, 4096), layout=ttnn.TILE_LAYOUT, device=d,
                        dtype=ttnn.bfloat16)
    for name, fn, flops, byt in [
            ("matmul_4096_bf16", lambda: ttnn.matmul(t, t), 2 * 4096 ** 3, 3 * 4096 ** 2 * 2),
            ("clone_4096_bf16", lambda: ttnn.clone(t), 0, 2 * 4096 ** 2 * 2),
            ("add_4096_bf16", lambda: ttnn.add(t, t), 0, 3 * 4096 ** 2 * 2)]:
        fn()
        ttnn.synchronize_device(d)
        ts = []
        for _ in range(8):
            t0 = time.time()
            r = fn()
            ttnn.synchronize_device(d)
            ts.append(time.time() - t0)
            ttnn.deallocate(r)
        s = float(np.median(ts))
        out[name] = {"seconds": s, "tflops": flops / s / 1e12 if flops else None,
                     "gbytes_s": byt / s / 1e9, "aiclk": clock.window([(time.time() - 1, time.time())])}
    big = ttnn.from_torch(torch.randn(256, 256, 128), layout=ttnn.TILE_LAYOUT, device=d,
                          dtype=ttnn.bfloat16)
    ts = []
    for _ in range(8):
        t0 = time.time()
        r = ttnn.permute(big, [1, 0, 2])
        ttnn.synchronize_device(d)
        ts.append(time.time() - t0)
        ttnn.deallocate(r)
    s = float(np.median(ts))
    out["permute_256_256_128"] = {"seconds": s, "gbytes_s": 2 * 2 * 256 * 256 * 128 / s / 1e9}
    ttnn.deallocate(t)
    ttnn.deallocate(big)
    return out


def cmd_census(args):
    lv, dev, ref = open_all(args, args.arm)
    clock = Clock()
    m0, z0, wm, wz = inputs(ref, args.n, args.seed)
    blob = {"stamp": stamp(args, clock), "n": args.n, "arm": args.arm, "blocks": {}}
    blob["roofs"] = roofs(dev, clock)
    print(json.dumps(blob["roofs"], default=str), flush=True)
    for stack_name in args.stacks.split(","):
        ke, kv = (1, 0) if stack_name == "extra" else (0, 1)
        untaped_fwd(dev, m0, z0, ke, kv)
        walls = []
        for _ in range(3):
            t0, t1 = untaped_fwd(dev, m0, z0, ke, kv)
            walls.append(t1 - t0)
        rec = {}
        for sync in (False, True):
            r = OpRecorder(dev.device, sync)
            untaped_fwd(dev, m0, z0, ke, kv)      # warm with the wrapper installed
            r.start()
            t0, t1 = untaped_fwd(dev, m0, z0, ke, kv)
            rows = r.stop()
            for mod, attr, real in r._saved:      # uninstall before the next pass
                setattr(mod, attr, real)
            rec["synced" if sync else "enqueue"] = {
                "wall": t1 - t0, "sum_t": sum(v["t"] for v in rows.values()),
                "calls": sum(v["calls"] for v in rows.values()), "rows": rows}
        blob["blocks"][stack_name] = {
            "wall_unwrapped": dist(walls), "loadavg": os.getloadavg(),
            "aiclk": clock.window([(time.time() - 60, time.time())]), **rec}
        print(stack_name, "wall", round(float(np.median(walls)), 4),
              "enqueue_sum", round(rec["enqueue"]["sum_t"], 4),
              "synced_sum", round(rec["synced"]["sum_t"], 4),
              "calls", rec["enqueue"]["calls"], flush=True)
        save(args.out or f"census_n{args.n}.json", blob)
    clock.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["step", "blocks", "census"])
    ap.add_argument("--params", default=A.DEFAULT_PARAMS)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--arm", default="stack")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--ns", default="256")
    ap.add_argument("--ks", default="1,2,4")
    ap.add_argument("--stacks", default="evo,extra")
    ap.add_argument("--extra", type=int, default=4)
    ap.add_argument("--evo", type=int, default=48)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--no-backward", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    {"step": cmd_step, "blocks": cmd_blocks, "census": cmd_census}[args.cmd](args)


if __name__ == "__main__":
    main()
