#!/usr/bin/env python3
"""bcx-stack: do the two proxy levers reach the real AF2 trunk, and what do they buy as a stack?

`bcx-mm2d` (`autograd._via2d`) and `bcx-triatt` (`bmm_program_config` in
`autograd.triangle_attention`, and the tile-moving head backwards in `taped_ttnn`) were measured
on the census's Boltz-shaped pair unit. This harness puts them on `perf/bcx_afgrad`'s taped
`AF2DeviceModel` blocks, which is what BindCraft 2 differentiates, and switches each one per arm
inside ONE process. Nothing in a lever is changed: the switches sit around it.

  mm2d   `_via2d` runs `fn(x)` unchanged when off, i.e. the rank-3 matmul it replaces
  bmm    `autograd.TRIATT_BMM_CONFIG`, the module switch `bcx-triatt` A/B'd itself with
  heads  the `nlp_concat_heads` / `nlp_create_qkv_heads` tape entries: the branch's, or main's
         `1f338604e` backwards (permute + reshape), copied verbatim below

Arms: base (all off), mm2d, triatt (bmm + heads), stack (all on); `bmm` and `heads` alone too.

Subcommands, each one device open on the card TT_VISIBLE_DEVICES names:

  reach  per real Evoformer and extra-MSA block: calls `_via2d` collapsed vs declined (reason,
         call site, fwd/bwd), `bmm_program_config` config vs None (reason), head tape entries
         and their backwards, which attention route each call took, and every taped matmul
  time   fwd and bwd per block, arms interleaved step by step with rotating order, p10/p50/p90,
         AICLK from the card's own sysfs node sampled every 0.25 s inside each timed window
  whole  the whole 4 + 48 stack, checkpointed as `afgrad stack --ckpt` runs it, from sequence
         logits: wall per arm, and each arm's logit gradient against the float64 chain
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

OUT = ROOT / "perf" / "bcx_stack"
ARMS = {"base": (False, False, False), "mm2d": (True, False, False),
        "bmm": (False, True, False), "heads": (False, False, True),
        "triatt": (False, True, True), "stack": (True, True, True)}


# ------------------------------------------------------------------------------ clock


def sysfs_node(visible=None):
    """TT_VISIBLE_DEVICES counts cards in PCI bus order; sysfs `tenstorrent!N` does not
    (`bcx-oplin`: on qb1 the naive node read an idle card at 800 MHz)."""
    visible = visible or os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    root = "/sys/class/tenstorrent"
    nodes = sorted(os.listdir(root),
                   key=lambda n: os.path.basename(os.path.realpath(f"{root}/{n}/device")))
    node = nodes[int(visible)]
    return f"{root}/{node}", os.path.basename(os.path.realpath(f"{root}/{node}/device"))


class Clock:
    def __init__(self, dt=0.25):
        node, self.pci = sysfs_node()
        self.path, self.dt, self.samples = f"{node}/tt_aiclk", dt, []
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append((time.time(), int(open(self.path).read().split()[0])))
            except Exception:
                pass
            self._stop.wait(self.dt)

    def stop(self):
        self._stop.set()

    def window(self, spans):
        xs = sorted(c for t, c in self.samples if any(a <= t <= b for a, b in spans))
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "min": xs[0], "median": xs[len(xs) // 2], "max": xs[-1]}


# ------------------------------------------------------------------------------ levers


def _old_concat_heads(shipped, args, kwargs, T, ttnn, count):
    """main `1f338604e` `taped_ttnn._v_concat_heads`, verbatim apart from the counter."""
    x = T._wrap(args[0])
    B, H, L, dh = (int(d) for d in x.value.shape)
    ra, rk = T._raw(args, kwargs)
    out_v = shipped(*ra, **rk)

    def make():
        def bw(g):
            count("concat_heads_bw_old")
            x.add_grad(ttnn.permute(ttnn.reshape(g, [B, L, H, dh]), [0, 2, 1, 3]))
        return bw

    return T._tape(out_v, [x], make)


def _old_create_qkv_heads(shipped, args, kwargs, T, ttnn, count):
    """main `1f338604e` `taped_ttnn._v_create_qkv_heads`, verbatim apart from the counter."""
    x = T._wrap(args[0])
    B, _, L, wide = (int(d) for d in x.value.shape)
    H = int(kwargs.get("num_heads", 1))
    dh = wide // (3 * H)
    outs = shipped(x.value, *[T._unwrap(a) for a in args[1:]],
                   **{k: T._unwrap(v) for k, v in kwargs.items()})

    def slot(s):
        def make():
            def bw(g):
                count("create_qkv_heads_bw_old")
                rows = ttnn.reshape(ttnn.permute(g, [0, 2, 1, 3]), [B, L, 1, H * dh])
                zero = ttnn.zeros([B, L, 1, H * dh], dtype=rows.dtype,
                                  layout=ttnn.TILE_LAYOUT, device=rows.device())
                parts = [rows if i == s else zero for i in range(3)]
                x.add_grad(ttnn.reshape(ttnn.concat(parts, dim=2), [B, 1, L, 3 * H * dh]))
            return bw
        return make

    return tuple(T._tape(o, [x], slot(s)) for s, o in enumerate(outs))


def _site(depth=2):
    f = sys._getframe(depth)
    return f"{pathlib.Path(f.f_code.co_filename).name}:{f.f_code.co_name}:{f.f_lineno}"


def _shape(t):
    v = getattr(t, "value", t)
    try:
        return [int(d) for d in v.shape]
    except Exception:
        return None


class Levers:
    """Arm switches and reach counters around the three lever sites. Install BEFORE any taped
    call: `taped_ttnn`'s shim caches a verb's tape entry on first use."""

    def __init__(self):
        import ttnn
        from tt_bio import af2, autograd as ag, taped_ttnn as T, tenstorrent as tn
        self.ttnn, self.ag, self.T = ttnn, ag, T
        self.mm2d = self.bmm = self.heads = True
        self.phase = "fwd"
        self.counts = collections.Counter()
        self.shapes = collections.defaultdict(collections.Counter)
        count = self.count

        via2d = ag._via2d

        def _via2d(x, fn, kw=None):
            s = [int(d) for d in x.shape]
            mc = (kw or {}).get("memory_config")
            if len(s) <= 2:
                why = "declined:rank<=2"
            elif s[-2] % ttnn.TILE_SIZE:
                why = "declined:rows-not-tile"
            elif x.layout != ttnn.TILE_LAYOUT:
                why = "declined:layout"
            elif x.is_sharded():
                why = "declined:sharded-operand"
            elif (kw or {}).get("program_config") is not None:
                why = "declined:program_config"
            elif mc is not None and mc.is_sharded():
                why = "declined:sharded-memory_config"
            else:
                why = "collapsed"
            site = _site()
            count("via2d", why, site)
            self.shapes[("via2d", why, site)][str(s)] += 1
            if why == "collapsed" and not self.mm2d:
                return fn(x)
            return via2d(x, fn, kw)

        ag._via2d = T._via2d = _via2d

        bmm = ag.bmm_program_config

        def _bmm(a, b, transpose_a=False, transpose_b=False):
            pc = bmm(a, b, transpose_a, transpose_b)
            sa, sb = _shape(a), _shape(b)
            if pc is not None:
                why = "config"
            elif len(sa) < 3 or len(sa) != len(sb) or sa[:-2] != sb[:-2]:
                why = "None:batch"
            else:
                M = sa[-1] if transpose_a else sa[-2]
                N = sb[-2] if transpose_b else sb[-1]
                K = sa[-2] if transpose_a else sa[-1]
                why = "None:not-tiles" if (M % 32 or N % 32 or K % 32) else "None:out>64tiles"
            count("bmm_program_config", why)
            self.shapes[("bmm", why)][f"{sa}x{sb} ta={transpose_a} tb={transpose_b}"] += 1
            return pc

        ag.bmm_program_config = _bmm

        tri = ag.triangle_attention

        def _tri(*a, **k):
            count("autograd.triangle_attention")
            return tri(*a, **k)

        ag.triangle_attention = _tri

        for name in ("_split_heads_v", "_merge_heads_v"):
            real = getattr(ag, name)

            def wrapped(*a, _real=real, _name=name, **k):
                count(_name)
                return _real(*a, **k)

            setattr(ag, name, wrapped)

        V = T._VERBS
        new_concat, new_qkv = V["experimental.nlp_concat_heads"], V["experimental.nlp_create_qkv_heads"]

        def concat(s, a, k):
            count("verb:nlp_concat_heads", "new" if self.heads else "old")
            return new_concat(s, a, k) if self.heads else _old_concat_heads(s, a, k, T, ttnn, count)

        def qkv(s, a, k):
            count("verb:nlp_create_qkv_heads", "new" if self.heads else "old")
            return new_qkv(s, a, k) if self.heads else _old_create_qkv_heads(s, a, k, T, ttnn, count)

        V["experimental.nlp_concat_heads"], V["experimental.nlp_create_qkv_heads"] = concat, qkv

        for verb in ("transformer.scaled_dot_product_attention", "matmul", "linear", "softmax"):
            real = V[verb]

            def counted(s, a, k, _real=real, _verb=verb):
                sa = _shape(a[0]) if a else None
                sb = _shape(a[1]) if len(a) > 1 else _shape(k.get("input_tensor_b"))
                tag = f"a{len(sa) if sa else '?'}d b{len(sb) if sb else '?'}d"
                if _verb == "matmul" and k.get("program_config") is not None:
                    tag += " program_config"
                count(f"verb:{_verb}", tag)
                return _real(s, a, k)

            V[verb] = counted

        fp32 = tn._fp32_softmax_attention

        def _fp32(*a, **k):
            count("route:_fp32_softmax_attention", _site())
            return fp32(*a, **k)

        tn._fp32_softmax_attention = af2._fp32_softmax_attention = _fp32
        # the shim must not have cached any of these yet
        stale = [n for n in ("matmul", "linear", "softmax", "experimental", "transformer")
                 if n in vars(T._SHIM)]
        assert not stale, f"taped_ttnn shim already cached {stale}; install Levers earlier"

    def count(self, *key):
        self.counts[(self.phase,) + key] += 1

    def arm(self, name):
        self.mm2d, self.bmm, self.heads = ARMS[name]
        self.ag.TRIATT_BMM_CONFIG = self.bmm
        self.name = name

    def take(self):
        c, s = self.counts, self.shapes
        self.counts, self.shapes = collections.Counter(), collections.defaultdict(collections.Counter)
        return ({" | ".join(map(str, k)): v for k, v in sorted(c.items())},
                {" | ".join(map(str, k)): dict(v) for k, v in s.items()})


# ------------------------------------------------------------------------------ one step


def block_step(dev, lv, m0, z0, wm, wz, stack_name, k=1, ckpt=False):
    """One taped forward + one backward of K blocks. Synced walls, host CPU of the backward."""
    ag = dev.ag
    ke, kv = (k, 0) if stack_name == "extra" else (0, k)
    gc.collect()
    ml, zl = dev.leaf(m0), dev.leaf(z0)
    dev.sync()
    lv.phase = "fwd"
    t0 = time.time()
    with dev.tt.tape():
        mo, zo = dev.stack(ml, zl, ke, kv, ckpt=ckpt)
    dev.sync()
    t1 = time.time()
    roots = [zo] if stack_name == "extra" else [mo, zo]
    seeds = [dev.seed(wz, zo)] if stack_name == "extra" else [dev.seed(wm, mo), dev.seed(wz, zo)]
    dev.sync()
    lv.phase = "bwd"
    t2, c2 = time.time(), time.process_time()
    ag.backward(roots, seeds)
    dev.sync()
    t3, c3 = time.time(), time.process_time()
    g = (dev.grad(ml, m0.shape), dev.grad(zl, z0.shape))
    if ckpt:
        ag.release_pins()
    del mo, zo, ml, zl, roots, seeds
    lv.phase = "fwd"
    return {"fwd": t1 - t0, "bwd": t3 - t2, "bwd_cpu": c3 - c2, "spans": [(t0, t1), (t2, t3)]}, g


def inputs(ref, n, seed):
    torch.manual_seed(seed)
    m0, z0 = A.embed(ref["bf16"], torch.randn(n, 20), torch.arange(n))
    m0, z0 = m0.detach(), z0.detach()
    return m0, z0, torch.randn(m0.shape) / m0.numel() ** 0.5, torch.randn(z0.shape) / z0.numel() ** 0.5


def dist(xs):
    xs = np.asarray(xs, float)
    return {"median": float(np.median(xs)), "p10": float(np.percentile(xs, 10)),
            "p90": float(np.percentile(xs, 90)), "mean": float(xs.mean()),
            "min": float(xs.min()), "max": float(xs.max()), "n": int(len(xs))}


def open_all(args):
    lv = Levers()
    dm, ref = A.load_models(args.params)
    return lv, A.Dev(dm), ref


def stamp(args, clock=None):
    s = A.stamp(args.card)
    s["pci"] = sysfs_node()[1]
    s["aiclk_node"] = clock.path if clock else None
    return s


def save(name, blob):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(blob, indent=1, default=str))
    print(f"wrote {OUT / name}", flush=True)


# ------------------------------------------------------------------------------ reach


def cmd_reach(args):
    lv, dev, ref = open_all(args)
    out = {"stamp": stamp(args), "n": args.n, "arms": {}}
    for arm in args.arms.split(","):
        lv.arm(arm)
        out["arms"][arm] = {}
        for stack_name in args.stacks.split(","):
            m0, z0, wm, wz = inputs(ref, args.n, args.seed)
            block_step(dev, lv, m0, z0, wm, wz, stack_name)       # warm; its counts are dropped
            lv.take()
            per = {}
            for k in (1, 2):
                block_step(dev, lv, m0, z0, wm, wz, stack_name, k=k)
                per[k] = lv.take()
            # per block = K2 - K1, so anything outside the blocks cancels
            c1, c2 = per[1][0], per[2][0]
            delta = {key: c2.get(key, 0) - c1.get(key, 0) for key in sorted(set(c1) | set(c2))}
            out["arms"][arm][stack_name] = {"per_block": {k: v for k, v in delta.items() if v},
                                            "K1": c1, "K2": c2, "K1_shapes": per[1][1]}
            print(arm, stack_name, flush=True)
            for key, v in delta.items():
                if v:
                    print(f"   {v:5d}  {key}", flush=True)
    save(args.out or f"reach_n{args.n}.json", out)


# ------------------------------------------------------------------------------ time


def cmd_time(args):
    lv, dev, ref = open_all(args)
    clock = Clock()
    arms = args.arms.split(",")
    blob = {"stamp": stamp(args, clock), "arms": arms, "steps": args.steps, "warm": args.warm,
            "order": "arms rotate by one position every step", "points": []}
    for n in [int(x) for x in args.ns.split(",")]:
        m0, z0, wm, wz = inputs(ref, n, args.seed)
        for stack_name in args.stacks.split(","):
            rec = {a: [] for a in arms}
            t_start = time.time()
            for step in range(args.warm + args.steps):
                order = arms[step % len(arms):] + arms[:step % len(arms)]
                for arm in order:
                    lv.arm(arm)
                    r, _ = block_step(dev, lv, m0, z0, wm, wz, stack_name)
                    if step >= args.warm:
                        rec[arm].append(r)
                lv.take()
            pt = {"n": n, "stack": stack_name, "loadavg": os.getloadavg(),
                  "wall": [t_start, time.time()], "arms": {}}
            for arm in arms:
                rs = rec[arm]
                pt["arms"][arm] = {
                    "fwd": dist([r["fwd"] for r in rs]), "bwd": dist([r["bwd"] for r in rs]),
                    "step": dist([r["fwd"] + r["bwd"] for r in rs]),
                    "bwd_host_cpu": dist([r["bwd_cpu"] for r in rs]),
                    "aiclk": clock.window([s for r in rs for s in r["spans"]])}
            base = pt["arms"].get("base")
            for arm in arms:
                a = pt["arms"][arm]
                if base:
                    a["x_vs_base"] = {m: base[m]["median"] / a[m]["median"]
                                      for m in ("fwd", "bwd", "step")}
                print(json.dumps({"n": n, "stack": stack_name, "arm": arm,
                                  "fwd": round(a["fwd"]["median"], 4),
                                  "bwd": round(a["bwd"]["median"], 4),
                                  "bwd_p10_p90": [round(a["bwd"]["p10"], 4), round(a["bwd"]["p90"], 4)],
                                  "x": a.get("x_vs_base"), "aiclk": a["aiclk"]}), flush=True)
            blob["points"].append(pt)
            save(args.out or "time.json", blob)
    clock.stop()


# ------------------------------------------------------------------------------ whole


def cmd_whole(args):
    """`afgrad stack`'s gradient, per arm: same seed, logits, readout and checkpointing."""
    lv, dev, ref = open_all(args)
    clock = Clock()
    ag = dev.ag
    n, ke, kv = args.n, args.extra, args.evo
    torch.manual_seed(args.seed)                  # the draws `afgrad stack` makes, in its order
    ridx = torch.arange(n)
    logits = torch.randn(n, 20) * 2.0
    wm = torch.randn(1, n, 256, dtype=torch.float64) / (n * 256) ** 0.5
    wz = torch.randn(n, n, 128, dtype=torch.float64) / (n * n * 128) ** 0.5

    def device_grad():
        gc.collect()
        lgt = logits.clone().float().requires_grad_(True)
        m0, z0 = A.embed(ref["bf16"], lgt, ridx)
        ml, zl = dev.leaf(m0), dev.leaf(z0)
        dev.sync()
        lv.phase = "fwd"
        t0 = time.time()
        with dev.tt.tape():
            mo, zo = dev.stack(ml, zl, ke, kv, ckpt=True)
        dev.sync()
        t1 = time.time()
        seeds = [dev.seed(wm, mo), dev.seed(wz, zo)]
        dev.sync()
        lv.phase = "bwd"
        t2, c2 = time.time(), time.process_time()
        ag.backward([mo, zo], seeds)
        dev.sync()
        t3, c3 = time.time(), time.process_time()
        lv.phase = "fwd"
        gm0, gz0 = dev.grad(ml, m0.shape), dev.grad(zl, z0.shape)
        torch.autograd.backward([m0, z0], [gm0.to(m0.dtype), gz0.to(z0.dtype)])
        ag.release_pins()
        del mo, zo, ml, zl, seeds
        gc.collect()
        return lgt.grad.double(), {"fwd": t1 - t0, "bwd": t3 - t2, "bwd_cpu": c3 - c2,
                                   "step": t3 - t0, "spans": [(t0, t1), (t2, t3)]}

    arms = args.arms.split(",")
    blob = {"stamp": stamp(args, clock), "n": n, "k_extra": ke, "k_evo": kv, "ckpt": True,
            "seed": args.seed, "arms": arms, "reps": args.reps, "per_arm": {}}
    grads, runs = {a: [] for a in arms}, {a: [] for a in arms}
    lv.arm(arms[0])
    device_grad()                                  # warm: JIT + program cache, discarded
    lv.take()
    for rep in range(args.reps):
        for arm in (arms if rep % 2 == 0 else arms[::-1]):
            lv.arm(arm)
            g, r = device_grad()
            grads[arm].append(g)
            runs[arm].append(r)
            print(arm, rep, {k: round(v, 3) for k, v in r.items() if k != "spans"}, flush=True)
            lv.take()
    lg64 = logits.double().clone().requires_grad_(True)
    t0 = time.time()
    m, z = A.embed(ref["f64"], lg64, ridx)
    m, z = A.ref_stack(ref["f64"], m, z, ke, kv)
    ((wm * m).sum() + (wz * z).sum()).backward()
    g64 = lg64.grad.detach()
    blob["f64"] = {"grad_norm": float(g64.norm()), "seconds": time.time() - t0}
    for arm in arms:
        rs = runs[arm]
        blob["per_arm"][arm] = {
            "fwd": dist([r["fwd"] for r in rs]), "bwd": dist([r["bwd"] for r in rs]),
            "step": dist([r["step"] for r in rs]), "bwd_host_cpu": dist([r["bwd_cpu"] for r in rs]),
            "aiclk": clock.window([s for r in rs for s in r["spans"]]),
            "grad_norm": float(grads[arm][0].norm()),
            "vs_f64": A.cmp(grads[arm][0], g64),
            "reps_bit_identical": all(torch.equal(grads[arm][0], g) for g in grads[arm][1:])}
        print(arm, json.dumps({k: blob["per_arm"][arm][k] for k in ("vs_f64", "reps_bit_identical")}),
              "step", round(blob["per_arm"][arm]["step"]["median"], 3), flush=True)
    if "base" in arms:
        for arm in arms:
            blob["per_arm"][arm]["vs_base"] = A.cmp(grads[arm][0], grads["base"][0])
            blob["per_arm"][arm]["bit_identical_to_base"] = torch.equal(grads[arm][0], grads["base"][0])
            blob["per_arm"][arm]["x_step_vs_base"] = (blob["per_arm"]["base"]["step"]["median"]
                                                      / blob["per_arm"][arm]["step"]["median"])
    clock.stop()
    save(args.out or f"whole_n{n}_e{ke}_v{kv}.json", blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reach", "time", "whole"])
    ap.add_argument("--params", default=A.DEFAULT_PARAMS)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--arms", default="base,mm2d,triatt,stack",
                    help=f"comma list from {sorted(ARMS)}")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--ns", default="128,256")
    ap.add_argument("--stacks", default="evo,extra")
    ap.add_argument("--extra", type=int, default=4)
    ap.add_argument("--evo", type=int, default=48)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    {"reach": cmd_reach, "time": cmd_time, "whole": cmd_whole}[args.cmd](args)


if __name__ == "__main__":
    main()
