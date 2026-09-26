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
`bcx-bwdplan` adds its three levers on top of `stack`: `bwd` (all three) or any of `bwd1` (the
batched plan for the backward's batched products and the one-row fold in `_via2d`), `bwd2`
(slice gradients joined once instead of padded with host-built zeros), `bwd4` (a gradient
arrives in its parent's own layout, so a reshape the forward made in ROW_MAJOR is undone in
ROW_MAJOR), combined as e.g. `bwd12`. Off means the code this row replaced, reproduced here: no
program config, the one-row operand left rank-3, the per-slice `_pad_slice`, and an `add_grad`
that keeps the layout the gradient came in.

Subcommands, each one device open on the card TT_VISIBLE_DEVICES names:

  reach  per real Evoformer and extra-MSA block: calls `_via2d` collapsed vs declined (reason,
         call site, fwd/bwd), `bmm_program_config` config vs None (reason), head tape entries
         and their backwards, which attention route each call took, and every taped matmul
  time   fwd and bwd per block, arms interleaved step by step with rotating order, p10/p50/p90,
         AICLK from the card's own sysfs node sampled every 0.25 s inside each timed window
  vjp    `afgrad vjp` under one arm: every block teacher-forced against float64
  bits   one block's input gradients per arm against base: bit-identical, rel L2, repeatable
  whole  the whole 4 + 48 stack, checkpointed as `afgrad stack --ckpt` runs it, from sequence
         logits: wall per arm, and each arm's logit gradient against the float64 chain
"""
from __future__ import annotations

import argparse
import collections
import contextlib
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
from tt_bio.aiclk import ARC_DEAD, parse as parse_aiclk  # noqa: E402

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
        self.dead = 0
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            try:
                raw = open(self.path).read()
            except Exception:
                raw = ""
            # A dead ARC answers 4294967295 and raises nothing, so a node that lies has to
            # be kept out of `samples` just as firmly as one that will not read.
            mhz = parse_aiclk(raw)
            if mhz is not None:
                self.samples.append((time.time(), mhz))
            elif raw:
                self.dead += 1
            self._stop.wait(self.dt)

    def stop(self):
        self._stop.set()

    def window(self, spans):
        xs = sorted(c for t, c in self.samples if any(a <= t <= b for a, b in spans))
        out = ({"n": 0} if not xs else
               {"n": len(xs), "min": xs[0], "median": xs[len(xs) // 2], "max": xs[-1]})
        if self.dead:
            out.update(dead_arc_reads=self.dead, sentinel=ARC_DEAD)
        return out


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


def _old_checkpoint(fn, *inputs, params=()):
    """`8ae35c46b` `autograd.checkpoint` (comments dropped, module names prefixed): one
    recompute and one inner backward per OUTPUT, and a `gc.collect()` after each."""
    from tt_bio import autograd as ag
    held = [t for t in inputs if isinstance(t, ag.Tensor)]
    for t in held:
        t.pinned = True
        ag._CKPT_PINS.append(t)
    ag._TOUCHED.clear()
    with ag.no_grad():
        produced = fn(*inputs)
    touched = [t for k, t in ag._PARAMS.items() if k in ag._TOUCHED] if params is ag._ALL_PARAMS \
        else list(params)
    ag._TOUCHED.clear()
    parents = list(held) + touched

    def _recompute(k, g):
        """Re-run the segment on fresh nodes over the same input VALUES, seed output `k`."""
        from tt_bio.taped_ttnn import recompute_scope
        inner = [ag.Tensor(t.value, requires_grad=t.requires_grad) if isinstance(t, ag.Tensor) else t
                 for t in inputs]
        with recompute_scope():
            y = fn(*inner)
        y = y[k] if k is not None else y
        if y.node is None:
            raise RuntimeError("checkpoint(fn): the recomputed segment built no tape; fn must "
                               "use taped ops and at least one input must require a gradient")
        y.backward(seed=g)
        for src, dup in zip(inputs, inner):
            if isinstance(src, ag.Tensor) and isinstance(dup, ag.Tensor) and dup.grad is not None:
                src.add_grad(dup.grad)
        del y, inner
        gc.collect()

    if not isinstance(produced, (tuple, list)):
        return ag._tape(produced.value if isinstance(produced, ag.Tensor) else produced, parents,
                     lambda: (lambda g: _recompute(None, g)))

    outs = []
    for k, prod in enumerate(produced):
        if prod is None:
            outs.append(None)
            continue
        outs.append(ag._tape(prod.value if isinstance(prod, ag.Tensor) else prod, parents,
                          (lambda k=k: (lambda g: _recompute(k, g)))))
    return tuple(outs)


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
        self._new_checkpoint = ag.checkpoint
        self.mm2d = self.bmm = self.heads = True
        self.bwd = set("124")
        self.mask = False
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
            elif s[-2] % ttnn.TILE_SIZE and s[-2] != 1:
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
            if why == "collapsed" and s[-2] % ttnn.TILE_SIZE:
                why = "collapsed:one-row"
            site = _site()
            count("via2d", why, site)
            self.shapes[("via2d", why, site)][str(s)] += 1
            if why == "collapsed" and not self.mm2d:
                return fn(x)
            if why == "collapsed:one-row" and "1" not in self.bwd:
                return fn(x)
            return via2d(x, fn, kw)

        ag._via2d = T._via2d = _via2d

        bmm = ag.bmm_program_config

        def _bmm(a, b, transpose_a=False, transpose_b=False):
            # AF2 never reaches `autograd.triangle_attention` (`reach_n256.json`), so every call
            # here is a VJP product through `autograd.bmm`, which had no config before bwd1.
            pc = bmm(a, b, transpose_a, transpose_b) if "1" in self.bwd else None
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

        join = ag.Tensor.add_grad_slice

        def add_grad_slice(t, g, starts, ends):
            count("slice_grad", "join" if "2" in self.bwd else "pad")
            if "2" in self.bwd:
                return join(t, g, starts, ends)
            return t.add_grad(ag._pad_slice(g, starts, ends, [int(d) for d in t.value.shape]))

        ag.Tensor.add_grad_slice = add_grad_slice

        add_grad = ag.Tensor.add_grad

        def _old_add_grad(t, grad):
            """`237f53064` `Tensor.add_grad`: the gradient kept whatever layout it came in."""
            if not t.requires_grad:
                return
            want, got = tuple(t.value.shape), tuple(grad.shape)
            if want != got:
                raise ValueError(f"gradient shape {got} does not match value shape {want}")
            if t._grad is None:
                t._grad = grad
                return
            if t._grad.dtype != ttnn.float32:
                t._grad = ttnn.typecast(t._grad, ttnn.float32)
            t._grad = ttnn.add(t._grad, grad if grad.dtype == ttnn.float32
                               else ttnn.typecast(grad, ttnn.float32))

        def add_grad_(t, grad):
            if t.requires_grad and grad.layout != t.value.layout:
                count("add_grad_relayout", str(grad.layout), _site())
            return (add_grad if "4" in self.bwd else _old_add_grad)(t, grad)

        ag.Tensor.add_grad = add_grad_

        tri = ag.triangle_attention

        def _tri(*a, **k):
            count("autograd.triangle_attention")
            return tri(*a, **k)

        ag.triangle_attention = _tri

        for name in ("split_heads_value", "merge_heads_value"):
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
        """`<lever arm>[+mask][@old|@new]`: `@` picks `autograd.checkpoint`, default new, and
        `+mask` hands the Evoformer an MSA mask, which runs `bcx-predictor`'s masked sites."""
        name, _, impl = name.partition("@")
        name, plus, _ = name.partition("+mask")
        self.mask = bool(plus)
        self.ag.checkpoint = _old_checkpoint if impl == "old" else self._new_checkpoint
        if name.startswith("bwd"):
            self.bwd = set(name[3:] or "124")
            self.mm2d, self.bmm, self.heads = ARMS["stack"]
        else:
            self.bwd = set()
            self.mm2d, self.bmm, self.heads = ARMS[name]
        self.ag.TRIATT_BMM_CONFIG = self.bmm
        self.name = name + plus + ("@" + impl if impl else "")

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
    mask = dev.up(torch.ones(1, m0.shape[-2])) if lv.mask else None
    with dev.tt.tape():
        mo, zo = dev.stack(ml, zl, ke, kv, ckpt=ckpt, msa_mask=mask)
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


#: A harness that is free to pick its own `n` will pick one the card never runs. BindCraft 2
#: pads the token axis to a multiple of 32 and masks what it added BEFORE anything is uploaded
#: (`tt_bio.bindcraft2.EvoformerOnDevice._pad`), so the shape on the card is `_pad32(n)`, never
#: `n`. The host hands 275 across the seam and the card executes 288. Two rows measured a block
#: at 275 -- `bcx-p10-devmap`'s whole per-family table and `bcx-p10-trimul`'s census, which
#: inherited the default -- and at 275 three fast paths that are open in production decline on
#: `% 32`, so the table was inflated and a root cause was found that the fold does not have.
#: Refusing here is the cheapest place to stop it: every harness in this campaign funnels
#: through `inputs`.
def inputs(ref, n, seed, ragged: bool = False):
    if n % 32 and not ragged:
        raise ValueError(
            f"n={n} is not a multiple of 32, so it is not a shape a BindCraft 2 round puts on "
            f"the card: bindcraft2.EvoformerOnDevice._pad rounds the token axis up and masks "
            f"the padding, so a complex of {n} residues executes at {-(-n // 32) * 32}. Pass "
            f"ragged=True only to measure the pre-pad host shape on purpose, and say so in the "
            f"write-up -- a number taken at a ragged n is not a round contribution.")
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
    out = {"stamp": stamp(args), "n": args.n, "ckpt": args.ckpt, "arms": {}}
    for arm in args.arms.split(","):
        lv.arm(arm)
        out["arms"][arm] = {}
        for stack_name in args.stacks.split(","):
            m0, z0, wm, wz = inputs(ref, args.n, args.seed)
            block_step(dev, lv, m0, z0, wm, wz, stack_name, ckpt=args.ckpt)   # warm, dropped
            lv.take()
            per = {}
            for k in (1, 2):
                block_step(dev, lv, m0, z0, wm, wz, stack_name, k=k, ckpt=args.ckpt)
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
        for stack_name, k in [(s_, int(k_)) for s_ in args.stacks.split(",")
                              for k_ in args.ks.split(",")]:
            rec = {a: [] for a in arms}
            t_start = time.time()
            for step in range(args.warm + args.steps):
                order = arms[step % len(arms):] + arms[:step % len(arms)]
                for arm in order:
                    lv.arm(arm)
                    r, _ = block_step(dev, lv, m0, z0, wm, wz, stack_name, k=k, ckpt=args.ckpt)
                    if step >= args.warm:
                        rec[arm].append(r)
                lv.take()
            pt = {"n": n, "stack": stack_name, "K": k, "ckpt": args.ckpt, "loadavg": os.getloadavg(),
                  "wall": [t_start, time.time()], "arms": {},
                  "aiclk_point": clock.window([(t_start, time.time())])}
            for arm in arms:
                rs = rec[arm]
                pt["arms"][arm] = {
                    "fwd": dist([r["fwd"] for r in rs]), "bwd": dist([r["bwd"] for r in rs]),
                    "step": dist([r["fwd"] + r["bwd"] for r in rs]),
                    "bwd_host_cpu": dist([r["bwd_cpu"] for r in rs]),
                    "aiclk": clock.window([s for r in rs for s in r["spans"]])}
            base = pt["arms"].get("base" if "base" in arms else arms[0])
            for arm in arms:
                a = pt["arms"][arm]
                if base:
                    a["x_vs_base"] = {m: base[m]["median"] / a[m]["median"]
                                      for m in ("fwd", "bwd", "step")}
                print(json.dumps({"n": n, "stack": stack_name, "K": k, "ckpt": args.ckpt, "arm": arm,
                                  "fwd": round(a["fwd"]["median"], 4),
                                  "bwd": round(a["bwd"]["median"], 4),
                                  "bwd_p10_p90": [round(a["bwd"]["p10"], 4), round(a["bwd"]["p90"], 4)],
                                  "bwd_cpu": round(a["bwd_host_cpu"]["median"], 4),
                                  "x": a.get("x_vs_base"), "aiclk": a["aiclk"],
                                  "aiclk_point": pt["aiclk_point"]}), flush=True)
            blob["points"].append(pt)
            save(args.out or "time.json", blob)
    clock.stop()


# ------------------------------------------------------------------------------ vjp


def cmd_vjp(args):
    """`afgrad vjp` (every block teacher-forced against float64) with one arm's levers set."""
    lv = Levers()
    lv.arm(args.arm)
    A.OUT = OUT / f"vjp_{args.arm}"
    A.cmd_vjp(args)


# ------------------------------------------------------------------------------ bits


def cmd_bits(args):
    """One block's input gradients per arm against the base arm, on identical inputs and seeds:
    bit-identical or not, and the relative L2 between them (the first arm is the base). The head entries are pure
    rearrangements, so an arm that differs from base only in them must be bit-identical."""
    lv, dev, ref = open_all(args)
    arms = args.arms.split(",")
    out = {"stamp": stamp(args), "n": args.n, "arms": arms, "stacks": {}}
    for stack_name in args.stacks.split(","):
        m0, z0, wm, wz = inputs(ref, args.n, args.seed)
        g = {}
        for arm in arms:
            lv.arm(arm)
            block_step(dev, lv, m0, z0, wm, wz, stack_name)
            g[arm] = [block_step(dev, lv, m0, z0, wm, wz, stack_name)[1] for _ in range(2)]
            lv.take()
        rec = {}
        for arm in arms:
            names = ("dz",) if stack_name == "extra" else ("dm", "dz")
            pick = (lambda gs: gs[1:]) if stack_name == "extra" else (lambda gs: gs)
            a, b, a2 = pick(g[arm][0]), pick(g[arms[0]][0]), pick(g[arm][1])
            rec[arm] = {nm: {"bit_identical_to_base": bool(torch.equal(x, y)),
                             "vs_base": A.cmp(x, y), "norm": float(x.norm()),
                             "repeat_bit_identical": bool(torch.equal(x, x2))}
                        for nm, x, y, x2 in zip(names, a, b, a2)}
            print(stack_name, arm, json.dumps(rec[arm]), flush=True)
        out["stacks"][stack_name] = rec
    save(args.out or f"bits_n{args.n}.json", out)


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
    # All ones, so a `+mask` arm computes the float64 chain's function through the masked
    # program and its distance to float64 stays a correctness check. Its cost is the ops.
    mask = dev.up(torch.ones(1, n))

    def device_grad():
        gc.collect()
        lgt = logits.clone().float().requires_grad_(True)
        m0, z0 = A.embed(ref["bf16"], lgt, ridx)
        ml, zl = dev.leaf(m0), dev.leaf(z0)
        dev.sync()
        lv.phase = "fwd"
        t0 = time.time()
        with dev.tt.tape():
            mo, zo = dev.stack(ml, zl, ke, kv, ckpt=True, msa_mask=mask if lv.mask else None)
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
        # loadavg at the end of the window: this cost is host-bound, so it scales with co-tenants
        return lgt.grad.double(), {"fwd": t1 - t0, "bwd": t3 - t2, "bwd_cpu": c3 - c2,
                                   "step": t3 - t0, "load1": os.getloadavg()[0],
                                   "spans": [(t0, t1), (t2, t3)]}

    arms = args.arms.split(",")
    blob = {"stamp": stamp(args, clock), "n": n, "k_extra": ke, "k_evo": kv, "ckpt": True,
            "seed": args.seed, "arms": arms, "reps": args.reps, "per_arm": {}}
    grads, runs = {a: [] for a in arms}, {a: [] for a in arms}
    for arm in arms:                               # warm every arm: each compiles its own
        lv.arm(arm)                                # programs, and an unwarmed `+mask` arm read
        device_grad()                              # 34.8 s against 15.0 s on its next rep
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
            "load1": [r["load1"] for r in rs],
            "aiclk": clock.window([s for r in rs for s in r["spans"]]),
            "grad_norm": float(grads[arm][0].norm()),
            "vs_f64": A.cmp(grads[arm][0], g64),
            "reps_bit_identical": all(torch.equal(grads[arm][0], g) for g in grads[arm][1:])}
        print(arm, json.dumps({k: blob["per_arm"][arm][k] for k in ("vs_f64", "reps_bit_identical")}),
              "step", round(blob["per_arm"][arm]["step"]["median"], 3), flush=True)
    ref = "base" if "base" in arms else arms[0]           # the reference arm
    blob["ref_arm"] = ref
    for arm in arms:
        blob["per_arm"][arm]["vs_base"] = A.cmp(grads[arm][0], grads[ref][0])
        blob["per_arm"][arm]["bit_identical_to_base"] = torch.equal(grads[arm][0], grads[ref][0])
        blob["per_arm"][arm]["x_step_vs_base"] = (blob["per_arm"][ref]["step"]["median"]
                                                  / blob["per_arm"][arm]["step"]["median"])
    clock.stop()
    save(args.out or f"whole_n{n}_e{ke}_v{kv}.json", blob)


# ------------------------------------------------------------------------------ fit


def cmd_fit(args):
    """`afgrad fit` under one arm: the largest K of blocks whose taped forward + backward
    completes, with the DRAM the tape holds after the forward and after the backward.
    Per-block tape bytes are the slope over K. `--stacks whole` runs `--extra` + `--evo`."""
    lv, dev, ref = open_all(args)
    lv.arm(args.arm)
    ag, ttnn = dev.ag, dev.ttnn
    n = args.n
    m0, z0, wm, wz = inputs(ref, n, args.seed)
    res = {"stamp": stamp(args), "n": n, "arm": args.arm, "ckpt": args.ckpt, "tries": []}

    def dram():
        mv = ttnn.get_memory_view(dev.device, ttnn.BufferType.DRAM)
        return int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)

    for stack_name in args.stacks.split(","):
        for k in [int(x) for x in args.ks.split(",")]:
            ke, kv = {"extra": (k, 0), "evo": (0, k), "whole": (args.extra, args.evo)}[stack_name]
            gc.collect()
            rec = {"stack": stack_name, "k_extra": ke, "k_evo": kv, "base_bytes": dram()}
            mo = zo = ml = zl = roots = seeds = None
            try:
                ml, zl = dev.leaf(m0), dev.leaf(z0)
                t0 = time.time()
                with dev.tt.tape():
                    mo, zo = dev.stack(ml, zl, ke, kv, ckpt=args.ckpt)
                dev.sync()
                rec["fwd_s"] = time.time() - t0
                rec["after_fwd_bytes"] = dram() - rec["base_bytes"]
                roots = [zo] if stack_name == "extra" else [mo, zo]
                seeds = ([dev.seed(wz, zo)] if stack_name == "extra"
                         else [dev.seed(wm, mo), dev.seed(wz, zo)])
                t0 = time.time()
                ag.backward(roots, seeds)
                dev.sync()
                rec["bwd_s"] = time.time() - t0
                rec["after_bwd_bytes"] = dram() - rec["base_bytes"]
                rec["ok"] = True
            except Exception as e:                  # an allocation refusal is the answer
                rec["ok"] = False
                rec["error"] = str(e).splitlines()[0][:300]
            finally:
                if args.ckpt:
                    ag.release_pins()
                mo = zo = ml = zl = roots = seeds = None
                gc.collect()
            rec["loadavg"] = os.getloadavg()
            res["tries"].append(rec)
            print(json.dumps(rec), flush=True)
            save(args.out or f"fit_n{n}_{args.arm}{'_ckpt' if args.ckpt else ''}.json", res)
            if not rec["ok"]:
                break


# ------------------------------------------------------------------------------ ckprof


def cmd_ckprof(args):
    """Where a checkpointed block's backward goes, against the same block uncheckpointed.
    Inside the backward: every `autograd.checkpoint` recompute (count), the taped re-forward in
    `recompute_scope`, the inner `Tensor.backward`, and every `gc.collect`, each synced on exit
    so device work is charged to the part that queued it. The remainder is the outer tape walk."""
    lv, dev, ref = open_all(args)
    lv.arm(args.arm)
    ag, T, ttnn = dev.ag, dev.tt, dev.ttnn
    parts = collections.defaultdict(float)
    counts = collections.Counter()

    def timed(name, fn):
        def wrapped(*a, **k):
            if lv.phase != "bwd":
                return fn(*a, **k)
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                if args.sync:
                    dev.sync()
                parts[name] += time.perf_counter() - t0
                counts[name] += 1
        return wrapped

    gc.collect = timed("gc.collect", gc.collect)
    ag.Tensor.backward = timed("inner backward", ag.Tensor.backward)
    outer, depth = ag.backward, [0]

    def backward(*a, **k):                  # the recompute's own replay is a nested call
        depth[0] += 1
        try:
            return (inner if depth[0] > 1 else outer)(*a, **k)
        finally:
            depth[0] -= 1

    inner = timed("inner backward", outer)
    ag.backward = backward
    real_scope = T.recompute_scope

    @contextlib.contextmanager
    def scope():
        t0 = time.perf_counter()
        with real_scope():
            yield
        if args.sync:
            dev.sync()
        parts["recompute fwd"] += time.perf_counter() - t0
        counts["recompute fwd"] += 1

    T.recompute_scope = scope
    clock = Clock()
    blob = {"stamp": stamp(args, clock), "arm": args.arm, "sync": args.sync, "points": []}
    for n in [int(x) for x in args.ns.split(",")]:
        m0, z0, wm, wz = inputs(ref, n, args.seed)
        for stack_name in args.stacks.split(","):
            pt = {"n": n, "stack": stack_name, "K": args.k, "gc_objects": len(gc.get_objects())}
            for ckpt in (False, True):
                rows = []
                for step in range(args.warm + args.steps):
                    parts.clear()
                    counts.clear()
                    r, _ = block_step(dev, lv, m0, z0, wm, wz, stack_name, k=args.k, ckpt=ckpt)
                    if step >= args.warm:
                        r["parts"], r["counts"] = dict(parts), dict(counts)
                        rows.append(r)
                keys = sorted({k_ for r in rows for k_ in r["parts"]})
                pt["ckpt" if ckpt else "plain"] = {
                    "fwd": dist([r["fwd"] for r in rows]), "bwd": dist([r["bwd"] for r in rows]),
                    "bwd_cpu": dist([r["bwd_cpu"] for r in rows]),
                    "parts": {k_: dist([r["parts"].get(k_, 0.0) for r in rows]) for k_ in keys},
                    "counts": rows[-1]["counts"],
                    "aiclk": clock.window([s_ for r in rows for s_ in r["spans"]])}
            pt["loadavg"] = os.getloadavg()
            blob["points"].append(pt)
            c, p_ = pt["ckpt"], pt["plain"]
            print(json.dumps({"n": n, "stack": stack_name, "plain_bwd": round(p_["bwd"]["median"], 4),
                              "ckpt_bwd": round(c["bwd"]["median"], 4),
                              "ckpt_fwd": round(c["fwd"]["median"], 4),
                              "parts": {k_: round(v["median"], 4) for k_, v in c["parts"].items()},
                              "counts": c["counts"], "aiclk": c["aiclk"],
                              "loadavg": pt["loadavg"]}), flush=True)
            save(args.out or f"ckprof_{args.arm}{'' if args.sync else '_nosync'}.json", blob)
    clock.stop()


# ------------------------------------------------------------------------------ gcdiag


def _cycles(objs):
    """Strongly connected components (size > 1, or a self-reference) of the reference graph
    restricted to `objs`, each described by its member types and, for a `Tensor` member, which
    attribute points back into the component."""
    idx = {id(o): i for i, o in enumerate(objs)}
    adj = [[idx[id(r)] for r in gc.get_referents(o) if id(r) in idx] for o in objs]
    index, low, on, st, comps, counter = {}, {}, set(), [], [], [0]
    sys.setrecursionlimit(max(10000, 4 * len(objs)))

    def strong(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        st.append(v)
        on.add(v)
        for w in adj[v]:
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = st.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1 or v in adj[v]:
                comps.append(comp)

    for v in range(len(objs)):
        if v not in index:
            strong(v)
    kinds = collections.Counter()
    for comp in comps:
        members = set(comp)
        sig = collections.Counter(type(objs[i]).__name__ for i in comp)
        back = []
        for i in comp:
            o = objs[i]
            for attr in getattr(type(o), "__slots__", ()):
                v = getattr(o, attr, None)
                if v is not None and id(v) in idx and idx[id(v)] in members:
                    back.append(f"{type(o).__name__}.{attr}")
            if type(o).__name__ == "function":
                back.append(f"fn:{o.__qualname__}")
        kinds[json.dumps({"types": dict(sig), "via": sorted(set(back))[:8]})] += 1
    return dict(kinds.most_common(10))


def cmd_gcdiag(args):
    """What the recompute's `gc.collect()` actually reclaims: the unreachable objects it finds,
    by type, and for the taped `Tensor`s among them which reference could close a cycle."""
    lv, dev, ref = open_all(args)
    lv.arm(args.arm)
    ag = dev.ag
    real = gc.collect
    found = []

    def collect(*a, **k):
        if lv.phase != "bwd":
            return real(*a, **k)
        gc.set_debug(gc.DEBUG_SAVEALL)
        t0 = time.perf_counter()
        n_ = real()
        dt = time.perf_counter() - t0
        gc.set_debug(0)
        kinds = collections.Counter(type(o).__name__ for o in gc.garbage)
        tens = [o for o in gc.garbage if isinstance(o, ag.Tensor)]
        cyc = _cycles(gc.garbage)
        found.append({"seconds": dt, "unreachable": n_, "tracked": len(gc.get_objects()),
                      "cycles": cyc,
                      "types": dict(kinds.most_common(15)),
                      "tensors": len(tens),
                      "tensors_with_node": sum(t.node is not None for t in tens),
                      "tensors_with_shares": sum(t.shares is not None for t in tens)})
        gc.garbage.clear()
        real()
        return n_

    gc.collect = collect
    out = {"stamp": stamp(args), "arm": args.arm, "points": []}
    for n in [int(x) for x in args.ns.split(",")]:
        m0, z0, wm, wz = inputs(ref, n, args.seed)
        for stack_name in args.stacks.split(","):
            block_step(dev, lv, m0, z0, wm, wz, stack_name, ckpt=True)       # warm
            found.clear()
            block_step(dev, lv, m0, z0, wm, wz, stack_name, ckpt=True)
            in_bwd = list(found)
            lv.phase = "bwd"                 # what a collect AFTER the step still finds
            gc.collect()
            lv.phase = "fwd"
            out["points"].append({"n": n, "stack": stack_name, "collects": in_bwd,
                                  "after_step": found[len(in_bwd):]})
            print(n, stack_name, json.dumps(out["points"][-1]), flush=True)
    save(args.out or f"gcdiag_{args.arm}.json", out)


# ------------------------------------------------------------------------------ ckbits


def cmd_ckbits(args):
    """One block's input gradients, checkpointed under each implementation, against the SAME
    block uncheckpointed. A checkpoint that recomputes once and replays the recomputed tape with
    every output's seed runs the uncheckpointed backward exactly, so it must be bit-identical to
    it; one that replays once per output sums partial passes and need not be."""
    lv, dev, ref = open_all(args)
    out = {"stamp": stamp(args), "arm": args.arm, "points": []}
    for n in [int(x) for x in args.ns.split(",")]:
        m0, z0, wm, wz = inputs(ref, n, args.seed)
        for stack_name in args.stacks.split(","):
            names = ("dz",) if stack_name == "extra" else ("dm", "dz")
            pick = (lambda gs: gs[1:]) if stack_name == "extra" else (lambda gs: gs)
            g = {}
            for tag, impl, ckpt in (("plain", "new", False), ("new", "new", True), ("old", "old", True)):
                lv.arm(f"{args.arm}@{impl}")
                block_step(dev, lv, m0, z0, wm, wz, stack_name, k=args.k, ckpt=ckpt)
                g[tag] = pick(block_step(dev, lv, m0, z0, wm, wz, stack_name, k=args.k, ckpt=ckpt)[1])
            pt = {"n": n, "stack": stack_name, "K": args.k}
            # float64 on the same bf16-rounded inputs and cotangents, so a disagreement between
            # the device arms can be told apart: which one is wrong, not only that they differ
            f64 = ref["f64"]
            if stack_name == "extra":
                r64, _ = A.ref_vjp(lambda z: A.ref_stack(f64, None, z, args.k, 0)[1],
                                   [A.bf(z0)], [A.bf(wz)])
            else:
                r64, _ = A.ref_vjp(lambda m, z: A.ref_stack(f64, m, z, 0, args.k),
                                   [A.bf(m0), A.bf(z0)], [A.bf(wm), A.bf(wz)])
            pt["vs_f64"] = {tag: {nm: A.cmp(x, y) for nm, x, y in zip(names, g[tag], r64)}
                            for tag in g}
            for tag in ("new", "old"):
                pt[tag] = {nm: {"bit_identical_to_plain": bool(torch.equal(x, y)),
                                "vs_plain": A.cmp(x, y), "max_abs_diff": float((x - y).abs().max())}
                           for nm, x, y in zip(names, g[tag], g["plain"])}
            out["points"].append(pt)
            print(json.dumps(pt), flush=True)
    save(args.out or f"ckbits_{args.arm}{'_k%d' % args.k if args.k > 1 else ''}.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reach", "time", "bits", "vjp", "whole", "fit", "ckprof", "gcdiag", "ckbits"])
    ap.add_argument("--params", default=A.DEFAULT_PARAMS)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--arms", default="base,mm2d,triatt,stack",
                    help=f"comma list from {sorted(ARMS)}")
    ap.add_argument("--arm", default="stack", help="vjp, fit: the one arm this process runs")
    ap.add_argument("--blocks", default=None, help="vjp: boundary indices to score")
    ap.add_argument("--controls-all", action="store_true")
    ap.add_argument("--ks", default="1", help="time: blocks per timed step")
    ap.add_argument("--ckpt", action="store_true", help="time, reach: checkpoint each block")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=1, help="ckprof: blocks per step")
    ap.add_argument("--no-sync", dest="sync", action="store_false",
                    help="ckprof: do not sync at the end of each timed part")
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
    {"reach": cmd_reach, "time": cmd_time, "bits": cmd_bits, "vjp": cmd_vjp, "whole": cmd_whole, "fit": cmd_fit, "ckprof": cmd_ckprof, "gcdiag": cmd_gcdiag, "ckbits": cmd_ckbits}[args.cmd](args)


if __name__ == "__main__":
    main()
