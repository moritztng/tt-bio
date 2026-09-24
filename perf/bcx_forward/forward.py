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
        import tt_bio.tenstorrent as tn
        routes = {"off": frozenset(), "on": frozenset(["extra_msa", "evoformer"])}
        arms = args.routes.split(",")
        for a in arms:                                        # warm: JIT + program cache
            dev.dm.set_triatt_fused(routes[a])
            untaped_fwd(dev, m0, z0, ke, kv)
        rec = {a: {"recycle_fwd": [], "taped_fwd": [], "bwd": [], "bwd_cpu": []} for a in arms}
        spans, served = [], {a: 0 for a in arms}
        t_start = time.time()
        for rep in range(args.reps):
            for a in (arms if rep % 2 == 0 else arms[::-1]):
                dev.dm.set_triatt_fused(routes[a])
                s0 = tn.TRIATT_FUSED_HIFI_STATS["served"]
                t0, t1 = untaped_fwd(dev, m0, z0, ke, kv)
                rec[a]["recycle_fwd"].append(t1 - t0)
                spans.append((t0, t1))
                r = taped_step(dev, lv, m0, z0, wm, wz, ke, kv, ckpt=not args.no_ckpt,
                               backward=not args.no_backward)
                served[a] = tn.TRIATT_FUSED_HIFI_STATS["served"] - s0
                rec[a]["taped_fwd"].append(r["taped_fwd"])
                if "bwd" in r:
                    rec[a]["bwd"].append(r["bwd"])
                    rec[a]["bwd_cpu"].append(r["bwd_cpu"])
                spans += r["spans"]
                print(n, rep, a, {k: round(v[-1], 4) for k, v in rec[a].items() if v}, flush=True)
        pt = {"n": n, "loadavg": os.getloadavg(), "wall": [t_start, time.time()],
              "aiclk": clock.window(spans), "served_last_rep": served, "arms": {}}
        for a in arms:
            p = {k: dist(v) for k, v in rec[a].items() if v}
            q = {"per": p, "forward_seconds_per_gradient_step":
                 p["recycle_fwd"]["median"] + p["taped_fwd"]["median"]}
            if "bwd" in p:
                q["trunk_step"] = dist([f + t + b for f, t, b in zip(
                    rec[a]["recycle_fwd"], rec[a]["taped_fwd"], rec[a]["bwd"])])
                q["forward_share_of_trunk"] = (q["forward_seconds_per_gradient_step"]
                                               / q["trunk_step"]["median"])
            pt["arms"][a] = q
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
    """Same process, same clock, this card: one roof per op CLASS, each at the trunk's own config.

    matmul at HiFi4 + fp32_dest_acc (`af2.compute_kernel_config`), since a roof at the default
    fidelity is a rate the shipped kernel never runs at; copy (`clone`) for layout ops; DRAM
    eltwise from an add of two DISTINCT tensors; layernorm, softmax, a reduction and the shipped
    permute measured as themselves, on shapes large enough to be DRAM-bound."""
    import ttnn
    from tt_bio.af2 import compute_kernel_config
    d, ckc = dev.device, compute_kernel_config()

    def up(*shape):
        return ttnn.from_torch(torch.randn(*shape), layout=ttnn.TILE_LAYOUT, device=d,
                               dtype=ttnn.bfloat16)

    a, b = up(4096, 4096), up(4096, 4096)
    p = up(1, 512, 512, 128)
    w, bb = up(1, 1, 1, 128), up(1, 1, 1, 128)
    s4 = up(64, 4, 256, 256)
    N, P, S = 4096 ** 2 * 2, 512 * 512 * 128 * 2, 64 * 4 * 256 * 256 * 2
    cases = [
        ("matmul", lambda: ttnn.matmul(a, b, compute_kernel_config=ckc), 2 * 4096 ** 3, 3 * N),
        ("copy", lambda: ttnn.clone(a), 0, 2 * N),
        ("eltwise", lambda: ttnn.add(a, b), 0, 3 * N),
        ("layernorm", lambda: ttnn.layer_norm(p, weight=w, bias=bb, epsilon=1e-5,
                                              compute_kernel_config=ckc), 0, 2 * P),
        ("softmax", lambda: ttnn.softmax(s4, dim=-1, compute_kernel_config=ckc), 0, 2 * S),
        ("reduction", lambda: ttnn.sum(p, dim=-1), 0, P),
        ("permute", lambda: ttnn.permute(p, (0, 2, 1, 3)), 0, 2 * P),
    ]
    out = {}
    for name, fn, flops, byt in cases:
        ttnn.deallocate(fn())
        ttnn.synchronize_device(d)
        # eight back-to-back ops per sync, so a loaded host's dispatch latency is amortised
        # instead of timed: one synced op on a box at loadavg 59 read clone at 80 GB/s
        ts, t_a = [], time.time()
        for _ in range(5):
            t0 = time.time()
            rs = [fn() for _ in range(8)]
            ttnn.synchronize_device(d)
            ts.append((time.time() - t0) / 8)
            for r in rs:
                ttnn.deallocate(r)
        sec = float(np.min(ts))
        out[name] = {"seconds": sec, "tflops": flops / sec / 1e12 if flops else None,
                     "gbytes_s": byt / sec / 1e9,
                     "aiclk": clock.window([(t_a - 0.5, time.time() + 0.5)])}
    for t in (a, b, p, w, bb, s4):
        ttnn.deallocate(t)
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


# ------------------------------------------------------------------------------ fused route


def cmd_fused(args):
    """The one route change the forward's own census points at, A/B'd in one process.

    AF2's triangle attentions ship on `_fp32_softmax_attention`, which materialises the whole
    [rows, heads, S, S] score tensor and pays four typecasts and twelve memory-config moves a
    block for it. `AF2DeviceModel.set_triatt_fused` swaps them onto the fused SDPA at the same
    HiFi4 / fp32_dest_acc the block's other matmuls already use, without rebuilding the blocks.
    A declined fused config is indistinguishable from an absent one from the outside, so every
    reading here carries `TRIATT_FUSED_HIFI_STATS`, and the accuracy of both arms is graded
    against the float64 reference block rather than against each other.
    """
    lv, dev, ref = open_all(args, args.arm)
    import tt_bio.tenstorrent as tn
    clock = Clock()
    m0, z0, wm, wz = inputs(ref, args.n, args.seed)
    stacks = {"off": frozenset(), "on": frozenset(["extra_msa", "evoformer"])}
    blob = {"stamp": stamp(args, clock), "n": args.n, "arm": args.arm, "steps": args.steps,
            "blocks": {}}
    for stack_name in args.stacks.split(","):
        ke, kv = (1, 0) if stack_name == "extra" else (0, 1)
        rec, spans, stats = {a: [] for a in stacks}, [], {}
        for s in stacks.values():                                    # warm both arms
            dev.dm.set_triatt_fused(s)
            untaped_fwd(dev, m0, z0, ke, kv)
        for step in range(args.steps):
            for a in (list(stacks) if step % 2 == 0 else list(stacks)[::-1]):
                dev.dm.set_triatt_fused(stacks[a])
                before = dict(tn.TRIATT_FUSED_HIFI_STATS)
                t0, t1 = untaped_fwd(dev, m0, z0, ke, kv)
                rec[a].append(t1 - t0)
                spans.append((t0, t1))
                stats[a] = {k: v - before.get(k, 0)
                            for k, v in tn.TRIATT_FUSED_HIFI_STATS.items()}
        names = ("z",) if stack_name == "extra" else ("m", "z")
        r = A.ref_stack(ref["f64"], m0.double(), z0.double(), ke, kv)
        tgt = (r[1],) if stack_name == "extra" else (r[0], r[1])
        acc = {}
        for a, s in stacks.items():
            dev.dm.set_triatt_fused(s)
            out = dev.stack(dev.up(m0), dev.up(z0), ke, kv)           # fresh uploads per arm
            got = ([dev.down(out[1], z0.shape)] if stack_name == "extra"
                   else [dev.down(out[0], m0.shape), dev.down(out[1], z0.shape)])
            acc[a] = {nm: A.cmp(g, t) for nm, g, t in zip(names, got, tgt)}
        blob["blocks"][stack_name] = {
            "seconds": {a: dist(v) for a, v in rec.items()},
            "x_vs_off": float(np.median(rec["off"]) / np.median(rec["on"])),
            "fused_stats": stats, "vs_f64": acc, "loadavg": os.getloadavg(),
            # the A/B's whole window, not the individual spans: a block forward is 28 ms and the
            # clock samples every 250 ms, so a per-span window is empty and stamps nothing
            "aiclk": clock.window([(spans[0][0], spans[-1][1])])}
        print(stack_name, json.dumps(blob["blocks"][stack_name], default=str), flush=True)
        save(args.out or f"fused_n{args.n}.json", blob)

    if args.whole:
        ke, kv = args.extra, args.evo
        rec = {a: {"untaped": [], "taped": []} for a in stacks}
        spans, stats, out_t = [], {}, {}
        for s_ in stacks.values():
            dev.dm.set_triatt_fused(s_)
            untaped_fwd(dev, m0, z0, ke, kv)
        for step in range(args.reps):
            for a in (list(stacks) if step % 2 == 0 else list(stacks)[::-1]):
                dev.dm.set_triatt_fused(stacks[a])
                before = dict(tn.TRIATT_FUSED_HIFI_STATS)
                t0, t1 = untaped_fwd(dev, m0, z0, ke, kv)
                rec[a]["untaped"].append(t1 - t0)
                spans.append((t0, t1))
                r = taped_step(dev, lv, m0, z0, wm, wz, ke, kv, ckpt=True, backward=False)
                rec[a]["taped"].append(r["taped_fwd"])
                spans += r["spans"]
                stats[a] = {k: v - before.get(k, 0)
                            for k, v in tn.TRIATT_FUSED_HIFI_STATS.items()}
        for a, s_ in stacks.items():
            dev.dm.set_triatt_fused(s_)
            o = dev.stack(dev.up(m0), dev.up(z0), ke, kv)
            out_t[a] = [dev.down(o[0], m0.shape), dev.down(o[1], z0.shape)]
        r64 = A.ref_stack(ref["f64"], m0.double(), z0.double(), ke, kv) if args.f64 else None
        blob["whole"] = {
            "k_extra": ke, "k_evo": kv,
            "seconds": {a: {m: dist(v) for m, v in d.items()} for a, d in rec.items()},
            "x_untaped": float(np.median(rec["off"]["untaped"]) / np.median(rec["on"]["untaped"])),
            "x_taped": float(np.median(rec["off"]["taped"]) / np.median(rec["on"]["taped"])),
            "forward_per_step": {a: float(np.median(d["untaped"]) + np.median(d["taped"]))
                                 for a, d in rec.items()},
            "fused_stats": stats, "loadavg": os.getloadavg(),
            "on_vs_off": {nm: A.cmp(x, y) for nm, x, y in
                          zip(("m", "z"), out_t["on"], out_t["off"])},
            "vs_f64": ({a: {nm: A.cmp(x, y) for nm, x, y in zip(("m", "z"), out_t[a], r64)}
                        for a in out_t} if r64 else None),
            "aiclk": clock.window([(spans[0][0], spans[-1][1])])}
        print("whole", json.dumps(blob["whole"], default=str), flush=True)
        save(args.out or f"fused_n{args.n}.json", blob)
    clock.stop()


# ------------------------------------------------------------------------------ latch


def cmd_latch(args):
    """Does one taped forward retire the fused route for every later UNTAPED forward?

    BC2's loop runs the recycle pass untaped and then the differentiated pass taped, in one
    process, 125 times. The fused SDPA declines under a tape by design; this asks whether that
    decline is remembered as an L1 refusal. Sequence: untaped, taped, untaped, untaped, one
    Evoformer block, fused on throughout, stats and the retired-config set read after each."""
    lv, dev, ref = open_all(args, args.arm)
    import tt_bio.tenstorrent as tn
    m0, z0, wm, wz = inputs(ref, args.n, args.seed)
    dev.dm.set_triatt_fused(frozenset(["extra_msa", "evoformer"]))
    rows = []
    for stage in ("untaped", "taped", "untaped", "untaped"):
        before = dict(tn.TRIATT_FUSED_HIFI_STATS)
        if stage == "untaped":
            t0, t1 = untaped_fwd(dev, m0, z0, 0, 1)
            sec = t1 - t0
        else:
            sec = taped_step(dev, lv, m0, z0, wm, wz, 0, 1, ckpt=True, backward=False)["taped_fwd"]
        rows.append({"stage": stage, "seconds": sec,
                     "stats": {k: v - before.get(k, 0) for k, v in tn.TRIATT_FUSED_HIFI_STATS.items()},
                     "retired_configs": len(tn._TRIATT_HIFI_OVER_L1)})
        print(json.dumps(rows[-1]), flush=True)
    save(args.out or f"latch_n{args.n}.json", {"stamp": stamp(args), "n": args.n, "rows": rows})

# ------------------------------------------------------------------------------ serves


def cmd_serves(args):
    """Which sizes does the fused route actually serve on one untaped Evoformer block, and when
    it declines, what did the kernel say? A decline is invisible from outside, so ask per size."""
    lv, dev, ref = open_all(args, args.arm)
    import tt_bio.tenstorrent as tn
    from tt_bio import triatt_sdpa as ts
    dev.dm.set_triatt_fused(frozenset(["extra_msa", "evoformer"]))
    rows = []
    for n in [int(x) for x in args.ns.split(",")]:
        m0, z0, _, _ = inputs(ref, n, args.seed)
        before, rej0 = dict(tn.TRIATT_FUSED_HIFI_STATS), dict(ts.REJECTS)
        l1_0 = len(tn.L1_CLASH_CENSUS)
        untaped_fwd(dev, m0, z0, 0, 1)
        rows.append({"n": n,
                     "stats": {k: v - before.get(k, 0) for k, v in tn.TRIATT_FUSED_HIFI_STATS.items()},
                     "rejects": {str(k): v - rej0.get(k, 0) for k, v in ts.REJECTS.items()
                                 if v - rej0.get(k, 0)},
                     "l1_refusals": [str(x)[:300] for x in tn.L1_CLASH_CENSUS[l1_0:]],
                     "picks": {str(k): v for k, v in tn.TRIATT_FUSED_HIFI_PICKS.items()}})
        print(json.dumps(rows[-1]), flush=True)
    save(args.out or "serves.json", {"stamp": stamp(args), "rows": rows})

# ------------------------------------------------------------------------------ exact


def cmd_exact(args):
    """torch bf16 and fp32 against float64 on the SAME inputs the device runs drew
    (`inputs(ref, n, seed)`), so the device's distance to float64 has the lab's own precision
    beside it. CPU only; no device is opened."""
    _, ref = A.load_models(args.params, device_arm=False)
    m0, z0, _, _ = inputs(ref, args.n, args.seed)
    blob = {"stamp": A.stamp(-1), "n": args.n, "k_extra": args.extra, "k_evo": args.evo}
    t0 = time.time()
    with torch.no_grad():
        r64 = A.ref_stack(ref["f64"], m0.double(), z0.double(), args.extra, args.evo)
    blob["f64_seconds"] = time.time() - t0
    for name in args.precisions.split(","):
        dt = {"bf16": torch.bfloat16, "f32": torch.float32}[name]
        t0 = time.time()
        with torch.no_grad():
            r = A.ref_stack(ref[name], m0.to(dt), z0.to(dt), args.extra, args.evo)
        blob[name] = {"seconds": time.time() - t0,
                      "m": A.cmp(r[0].double(), r64[0]), "z": A.cmp(r[1].double(), r64[1])}
        print(name, json.dumps(blob[name]), flush=True)
        save(args.out or f"exact_n{args.n}_e{args.extra}_v{args.evo}.json", blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["step", "blocks", "census", "fused", "latch", "serves", "exact"])
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
    ap.add_argument("--routes", default="off", help="step: triangle-attention routes, off,on")
    ap.add_argument("--precisions", default="bf16", help="exact: bf16,f32")
    ap.add_argument("--f64", action="store_true", help="fused --whole: grade both arms against float64")
    ap.add_argument("--whole", action="store_true",
                    help="fused: also A/B the whole 4+48 forward, taped and untaped")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    {"step": cmd_step, "blocks": cmd_blocks, "census": cmd_census,
     "fused": cmd_fused, "latch": cmd_latch, "serves": cmd_serves, "exact": cmd_exact}[args.cmd](args)


if __name__ == "__main__":
    main()
