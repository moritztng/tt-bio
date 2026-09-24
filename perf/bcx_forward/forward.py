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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["step", "blocks", "census", "fused", "latch", "serves"])
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
    ap.add_argument("--f64", action="store_true", help="fused --whole: grade both arms against float64")
    ap.add_argument("--whole", action="store_true",
                    help="fused: also A/B the whole 4+48 forward, taped and untaped")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    {"step": cmd_step, "blocks": cmd_blocks, "census": cmd_census,
     "fused": cmd_fused, "latch": cmd_latch, "serves": cmd_serves}[args.cmd](args)


if __name__ == "__main__":
    main()
