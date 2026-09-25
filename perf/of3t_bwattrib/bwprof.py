#!/usr/bin/env python3
"""J0: where the 2.107 ms of a backward verb call actually goes.

`of3t-gpugap` priced the backward's 168,922 ttnn verb calls at 356.00 s -- 2.107 ms each,
44.3x the forward's 0.0475 ms on the same program in the same warm process -- and nothing in
the record says why. A mean cannot tell a fixed per-call overhead from a few verb classes
carrying the whole thing, and those are different defects with different fixes. So this
measures the DISTRIBUTION: per call, the verb name, the output shape, dtype and layout, the
wall nanoseconds, and the bytes the call has to move.

The bytes are the point. This workload is DRAM-bound -- `of3t/BACKWARD.md` puts the machine
balance at 247-338 FLOP/byte against a bandwidth roof near 440 GB/s -- so a verb's honest
floor is its operand traffic divided by that roof, not a per-call constant. Every row of the
histogram carries its achieved GB/s beside its microseconds, which is what turns "2.107 ms is
mysterious" into "2.107 ms is N % of the DRAM roof", or does not.

The instrument is the same rebinding `perf/of3t_memory/alloc_profile.py` uses and
`tt_bio/taped_ttnn.py` uses before it: rebind the name `ttnn` inside every tt-bio module,
`tt_bio.autograd` INCLUDED. That inclusion is the whole point here -- the tape deliberately
excludes autograd so its closures reach the real verbs, and the backward is therefore the one
phase a tape-side counter cannot see.

Two further attributions come free and neither is a verb count:

  * PER CLOSURE. `_Node.fn` is wrapped at tape-build time, so every verb is also attributed
    to the backward closure that issued it, by `__qualname__`. That is the axis that answers
    "concentrated or uniform", because a fixed per-call overhead spreads evenly over closures
    and a bad op class does not.
  * THE NAMED SUSPECTS, counted rather than argued: `Tensor.evict`, `add_grad`'s typecast to
    fp32, `add_grad`'s `to_layout` at fan-in, and the score-block count inside
    `autograd.triangle_attention`, which is what settles J3's 14x bracket.

ARMS, and each is a break control that must MOVE the number or the suspect is not the cause:

  base        production
  fanin       `ag.FANIN_MIXED = True` -- kills add_grad's typecast-to-fp32 pair on every
              second contribution. If the typecast is the cost, this moves it.
  noexact     `ag.exact_training(False)` -- the decisive break control. `EXACT_TRAINING_OPS`
              = ("softmax", "layer_norm") is default-ON under a tape and its own docstring
              says what it costs: "a host round trip per softmax and per layer norm, in the
              forward and in the backward's recompute". It is an ACCURACY instrument, not the
              model, and this arm prices it. It is not a shippable lever on its own --
              of3t-stackexact reads 0.9823x the bar with it on and 1.4512x with it off -- so
              what this arm buys is the correct DENOMINATOR for every other lever.
  sync        `ttnn.synchronize_device` after every verb. ttnn dispatch is asynchronous, so
              a bare wall clock per call is host time that may or may not be hiding device
              time behind it. This arm drains the queue per call: if the total barely moves,
              the host was already blocking and the time is real device work; if it explodes,
              the base arm was overlapping and the per-call attribution is smeared.

    bwprof.py --tokens 384 --arm base --out perf/of3t_bwattrib/out/hist_384_base.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.clocksample import during                                      # noqa: E402
from perf.of3t_perf import step as S                                     # noqa: E402

NS = time.perf_counter_ns
_ITEM = {}


def _itemsize(dt):
    s = _ITEM.get(dt)
    if s is None:
        n = str(dt)
        s = 4 if ("float32" in n or "uint32" in n or "int32" in n) else (1 if "8" in n else 2)
        _ITEM[dt] = s
    return s


def _desc(t):
    """(shape, dtype, layout, bytes) for a ttnn tensor, or None. Metadata only, no sync."""
    try:
        shp = tuple(int(d) for d in t.shape)
    except Exception:                                                    # noqa: BLE001
        return None
    try:
        dt, lay = t.dtype, t.layout
    except Exception:                                                    # noqa: BLE001
        return None
    n = 1
    for d in shp:
        n *= d
    return shp, str(dt), str(lay), n * _itemsize(dt)


class Rec:
    """Aggregate as we go. 168,922 per-call rows would be 40 MB of JSON and no more answer."""

    def __init__(self):
        # [n, incl_ns, max_ns, bytes, self_ns]. INCLUSIVE is what a bare stopwatch around the
        # call gives you and it is wrong to sum: `tt_bio.taped_ttnn` is itself a rebound
        # tt_bio module, so a model-level verb call re-enters the wrapper on its way to the
        # raw op and the outer row already contains the inner one. Summing the base run's
        # inclusive column gave a forward verb total of 432.02 s inside a forward that took
        # 274.95 s -- 157 s of a 275 s leg counted twice. SELF time subtracts the children,
        # so the column sums to the leg and a share is a share.
        self.verb = {}
        self.node = {}          # closure __qualname__ -> [n_fired, ns, verbs]
        self.cur = None
        self.calls = 0
        self.outer_calls = 0    # calls entered at depth 0; the population a ratio may use
        self.ns = 0             # inclusive, kept only to show the double-count
        self.self_ns = 0        # exclusive; THIS is the leg's verb time
        self.depth = 0
        self.slowest = []
        self.on = False

    def bump(self, name, out, args):
        self.calls += 1
        d = _desc(out)
        if d is None:
            for a in args:
                d = _desc(a)
                if d is not None:
                    break
        if d is None:
            d = ((), "-", "-", 0)
        shp, dt, lay, ob = d
        # Traffic: every ttnn tensor operand read, plus the output written.
        nb = ob
        for a in args:
            da = _desc(a)
            if da is not None:
                nb += da[3]
        return (name, shp, dt, lay), nb

    def add(self, key, nb, ns, sns=None):
        if sns is None:
            sns = ns
        r = self.verb.get(key)
        if r is None:
            self.verb[key] = [1, ns, ns, nb, sns]
        else:
            r[0] += 1
            r[1] += ns
            r[3] += nb
            r[4] += sns
            if ns > r[2]:
                r[2] = ns
        self.ns += ns
        self.self_ns += sns
        if self.cur is not None:
            n = self.node.get(self.cur)
            if n is not None:
                n[2] += 1
        if ns > 2_000_000:                                               # over 2 ms, keep it
            if len(self.slowest) < 400:
                self.slowest.append((ns, key, self.cur))


REC = Rec()
CH: list = []   # per-frame child-nanosecond accumulator; see Rec.__init__


def _use(rec):
    """Swap the active recorder. The forward and the backward get one each, so the 44.3x
    ratio is finally two readings from ONE instrument rather than two counters with
    different populations -- which is the first thing that has to be ruled out about it."""
    global REC
    old, REC = REC, rec
    return old


class _W:
    """`alloc_profile._Watch` with a stopwatch. Same rebinding, same nested-namespace walk.

    NO `__slots__`. The wrapper caches each resolved verb in its own instance dict so a hot
    call site pays one attribute load instead of a rebuild per call, and `__slots__` turns
    that cache write into an AttributeError on the first verb the walk had not seen.
    """

    def __init__(self, real, prefix=""):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_prefix", prefix)

    def __getattr__(self, name):
        import types
        real = object.__getattribute__(self, "_real")
        attr = getattr(real, name)
        qual = object.__getattribute__(self, "_prefix") + name
        if isinstance(attr, (types.ModuleType, _W)) or type(attr).__name__ == "_Ttnn":
            out = _W(attr, qual + ".")
        elif callable(attr) and not isinstance(attr, type):

            def call(*a, _f=attr, _q=qual, **k):
                rec = REC
                if not rec.on:
                    return _f(*a, **k)
                if rec.depth == 0:
                    rec.outer_calls += 1
                rec.depth += 1
                CH.append(0)
                t0 = NS()
                try:
                    r = _f(*a, **k)
                finally:
                    ns = NS() - t0
                    kids = CH.pop()
                    rec.depth -= 1
                    if CH:
                        CH[-1] += ns
                sns = ns - kids
                if sns < 0:
                    sns = 0
                try:
                    key, nb = rec.bump(_q, r, a)
                    rec.add(key, nb, ns, sns)
                except Exception:                                        # noqa: BLE001
                    rec.calls += 1
                    rec.ns += ns
                    rec.self_ns += sns
                return r
            out = call
        else:
            return attr
        object.__setattr__(self, name, out)
        return out


def _swap(install, saved):
    """Rebind `ttnn` in every tt-bio module, autograd INCLUDED. Mirrors `_swap_watch`."""
    import ttnn
    if install:
        del saved[:]
        for nm, mod in list(sys.modules.items()):
            if not nm.startswith("tt_bio") or mod is None:
                continue
            cur = getattr(mod, "ttnn", None)
            if cur is None or isinstance(cur, _W):
                continue
            saved.append((mod, cur))
            mod.ttnn = _W(cur)
        REC.on = True
    else:
        REC.on = False
        for mod, cur in saved:
            mod.ttnn = cur
        del saved[:]


# --- the named suspects, counted --------------------------------------------------------
SUS = Counter()


def instrument(ag, sync_dev=None):
    """Wrap the closure constructor and the four suspects. Installed BEFORE the tape."""
    import ttnn

    orig_node = ag._Node.__init__

    def node_init(self, fn, parents, group=None):
        qn = getattr(fn, "__qualname__", repr(fn))

        def timed(g, _f=fn, _q=qn):
            prev, REC.cur = REC.cur, _q
            n = REC.node.get(_q)
            if n is None:
                n = REC.node[_q] = [0, 0, 0]
            n[0] += 1
            t0 = NS()
            try:
                return _f(g)
            finally:
                n[1] += NS() - t0
                REC.cur = prev
        orig_node(self, timed, parents, group)

    ag._Node.__init__ = node_init

    # add_grad: count the two conversions the brief names, without changing either.
    orig_add = ag.Tensor.add_grad

    def add_grad(self, grad):
        if self.requires_grad:
            SUS["add_grad"] += 1
            try:
                if grad.layout != self.value.layout:
                    SUS["add_grad_to_layout"] += 1
                if self._grad is not None:
                    SUS["add_grad_fanin"] += 1
                    if not ag.FANIN_MIXED:
                        if self._grad.dtype != ttnn.float32:
                            SUS["add_grad_typecast_acc"] += 1
                        if grad.dtype != ttnn.float32:
                            SUS["add_grad_typecast_in"] += 1
            except Exception:                                            # noqa: BLE001
                pass
        return orig_add(self, grad)
    ag.Tensor.add_grad = add_grad

    orig_evict = ag.Tensor.evict

    def evict(self):
        SUS["evict_called"] += 1
        try:
            if self.evictable and self.shares is None and \
               self.value.memory_config().buffer_type == ttnn.BufferType.L1:
                SUS["evict_moved_l1_to_dram"] += 1
        except Exception:                                                # noqa: BLE001
            pass
        return orig_evict(self)
    ag.Tensor.evict = evict

    # J3's bracket: the score-block count inside the chunked triangle-attention backward.
    orig_ta = ag.triangle_attention

    def triangle_attention(q, k, v, bias=None, **kw):
        try:
            B = int(q.value.shape[0])
            n_q = int(q.value.shape[2])
            cB = kw.get("chunk") or B
            cQ = kw.get("q_chunk") or n_q
            nb = -(-B // cB) * (-(-n_q // cQ))
            SUS["sdpa_taped_calls"] += 1
            SUS["sdpa_score_blocks_total"] += nb
            SDPA_SHAPES[(tuple(int(d) for d in q.value.shape),
                         int(cB), int(cQ), nb)] += 1
        except Exception:                                                # noqa: BLE001
            pass
        return orig_ta(q, k, v, bias, **kw)
    ag.triangle_attention = triangle_attention

    if sync_dev is not None:
        SUS["arm_sync"] = 1


SDPA_SHAPES = Counter()


def hostrt(ag):
    """The exact-training host round trips, READ off the counters `tt_bio.autograd` already
    keeps rather than wrapped. `exact_training` is default-ON under a tape and its own
    docstring prices it: "a host round trip per softmax and per layer norm, in the forward and
    in the backward's recompute", softmax at "166.8x the op". That is the largest single named
    cost on the taped path and nothing in the record had counted it against a step. The
    `noexact` arm turns it off, which is the break control that makes the count attributable.

    Read (not wrapped) because the exact verbs are reached through the module-level `_EXACT`
    dispatch dict, which captured the function objects at import; rebinding the module
    attribute would not be seen by the dict and would silently count zero."""
    import copy
    return {"softmax": copy.deepcopy(getattr(ag, "EXACT_SOFTMAX_STATS", {})),
            "layer_norm": copy.deepcopy(getattr(ag, "EXACT_LAYER_NORM_STATS", {})),
            "ops_active": list(ag.exact_training_ops())}


def hostrt_delta(before, after):
    d = {"ops_active": after.get("ops_active")}
    for op in ("softmax", "layer_norm"):
        b, a2 = before.get(op, {}), after.get(op, {})
        d[op] = {k: a2.get(k, 0) - b.get(k, 0) for k in a2}
    return d



def _rows(rec, top=60):
    rows = []
    for (name, shp, dt, lay), (n, ns, mx, nb, sns) in rec.verb.items():
        rows.append({"verb": name, "shape": list(shp), "dtype": dt.replace("DataType.", ""),
                     "layout": lay.replace("Layout.", ""), "n": n,
                     "self_s": round(sns / 1e9, 4), "incl_s": round(ns / 1e9, 4),
                     "us_per_call": round(sns / n / 1e3, 1),
                     "max_us": round(mx / 1e3, 1), "gb": round(nb / 1e9, 3),
                     "gb_s": round(nb / sns, 2) if sns else None})
    rows.sort(key=lambda r: -r["self_s"])
    return rows[:top], len(rows)


def _byverb(rec):
    agg = {}
    for (name, _s, _d, _l), (n, ns, _m, nb, sns) in rec.verb.items():
        a = agg.setdefault(name, [0, 0, 0, 0])
        a[0] += n
        a[1] += ns
        a[2] += nb
        a[3] += sns
    out = [{"verb": k, "n": v[0], "self_s": round(v[3] / 1e9, 3),
            "incl_s": round(v[1] / 1e9, 3),
            "us_per_call": round(v[3] / v[0] / 1e3, 1),
            "gb": round(v[2] / 1e9, 2),
            "gb_s": round(v[2] / v[3], 2) if v[3] else None}
           for k, v in agg.items()]
    out.sort(key=lambda r: -r["self_s"])
    return out


def _bynode(rec):
    out = [{"closure": k, "fired": v[0], "total_s": round(v[1] / 1e9, 3),
            "verbs": v[2], "ms_per_fire": round(v[1] / v[0] / 1e6, 2) if v[0] else None,
            "us_per_verb": round(v[1] / v[2] / 1e3, 1) if v[2] else None}
           for k, v in rec.node.items()]
    out.sort(key=lambda r: -r["total_s"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--arm", default="base", choices=("base", "fanin", "sync", "noexact"))
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {REPO} rev-parse --abbrev-ref HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg()}, "arm": a.arm,
        "config": {"crop": a.tokens, "cycles": a.cycles}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: a.out.write_text(json.dumps(out, indent=1, default=str))   # noqa: E731
    dump()

    with during() as clk:
        try:
            import ttnn
            from tt_bio import autograd as ag
            from tt_bio import taped_ttnn as TT
            from tt_bio.tenstorrent import get_device

            held, _meta = S.capture(a.tokens, out)
            trunk = held["trunk"][0]
            dev = get_device()
            out["env"]["arch"] = str(dev.arch())
            params = S.declare_weights(trunk, out)

            exact_ctx = None
            if a.arm == "noexact":
                exact_ctx = ag.exact_training(False)
                exact_ctx.__enter__()
            out["env"]["exact_training_ops"] = list(ag.exact_training_ops())
            if a.arm == "fanin":
                ag.FANIN_MIXED = True
            out["env"]["fanin_mixed"] = bool(ag.FANIN_MIXED)
            instrument(ag, dev if a.arm == "sync" else None)
            if a.arm == "sync":
                real_sync, real_w = ttnn.synchronize_device, _W.__getattr__

                def gattr(self, name, _g=real_w):
                    r = _g(self, name)
                    if not callable(r) or name in ("synchronize_device",):
                        return r

                    def s(*x, _r=r, **y):
                        v = _r(*x, **y)
                        if REC.on:
                            real_sync(dev)
                        return v
                    object.__setattr__(self, name, s)
                    return s
                _W.__getattr__ = gattr

            snap_args, snap_kwargs = held["trunk_snap"]
            args_ = S._rehydrate(snap_args, dev)
            kwargs_ = {k: v for k, v in S._rehydrate(snap_kwargs, dev).items()
                       if k != "progress_fn"}
            trunk.num_cycles = a.cycles
            gc.collect()

            saved: list = []
            z = None
            hrt0 = hostrt(ag)
            fwd = Rec()
            _use(fwd)
            t0 = time.perf_counter()
            with ag.tape():
                _swap(True, saved)
                try:
                    _s, z = trunk(*args_, **kwargs_)
                    ttnn.synchronize_device(dev)
                finally:
                    _swap(False, saved)
            fwall = time.perf_counter() - t0
            out["forward"] = {"s": round(fwall, 2),
                              "ok": True, "cycles": a.cycles,
                              "verb_calls": fwd.calls,
                              "outer_verb_calls": fwd.outer_calls,
                              "verb_self_s": round(fwd.self_ns / 1e9, 2),
                              "verb_incl_s_DOUBLECOUNTED": round(fwd.ns / 1e9, 2),
                              "verb_share_of_forward":
                                  round(fwd.self_ns / 1e9 / fwall, 4) if fwall else None,
                              "us_per_call": round(fwd.self_ns / fwd.calls / 1e3, 1)
                              if fwd.calls else None,
                              "host_roundtrips": hostrt_delta(hrt0, hostrt(ag)),
                              "sdpa_taped_calls": SUS["sdpa_taped_calls"],
                              "sdpa_score_blocks_total": SUS["sdpa_score_blocks_total"]}
            out["forward_by_verb"] = _byverb(fwd)[:25]
            out["forward_by_shape_class"] = _rows(fwd, 25)[0]
            out["sdpa_shapes"] = [{"q_shape": list(k[0]), "chunk_B": k[1], "chunk_Q": k[2],
                                   "blocks_per_call": k[3], "calls": v}
                                  for k, v in SDPA_SHAPES.most_common(12)]
            dump()

            if not isinstance(z, ag.Tensor):
                raise SystemExit("the trunk output is not taped; nothing to differentiate")

            import torch
            zr = z.value
            seed = ttnn.from_torch(torch.ones(tuple(int(d) for d in zr.shape)),
                                   layout=ttnn.TILE_LAYOUT, device=dev, dtype=zr.dtype)
            out["backward"] = {"tape_nodes": len(ag._reverse_topo([z]))}
            SUS.clear()
            hrt1 = hostrt(ag)
            bwd = Rec()
            _use(bwd)
            rs = TT.recompute_scope()
            rs.__enter__()
            t0 = time.perf_counter()
            try:
                _swap(True, saved)
                try:
                    ag.backward([z], [seed])
                    ttnn.synchronize_device(dev)
                finally:
                    _swap(False, saved)
                out["backward"]["ok"] = True
            except Exception as e:                                       # noqa: BLE001
                out["backward"]["ok"] = False
                out["backward"]["error_head"] = str(e).split("backtrace")[0][:900]
                out["backward"]["error"] = traceback.format_exc()[-4000:]
            finally:
                rs.__exit__(None, None, None)
            wall = time.perf_counter() - t0
            rows, distinct = _rows(bwd, a.top)
            out["backward"].update({
                "s": round(wall, 2),
                "verb_calls": bwd.calls,
                "outer_verb_calls": bwd.outer_calls,
                "verb_self_s": round(bwd.self_ns / 1e9, 2),
                "verb_incl_s_DOUBLECOUNTED": round(bwd.ns / 1e9, 2),
                "verb_share_of_backward": round(bwd.self_ns / 1e9 / wall, 4) if wall else None,
                "us_per_call": round(bwd.self_ns / bwd.calls / 1e3, 1) if bwd.calls else None,
                "host_roundtrips": hostrt_delta(hrt1, hostrt(ag)),
                "distinct_shape_classes": distinct,
                "suspects": dict(SUS),
                "params_with_grad": sum(1 for t in params.values()
                                        if getattr(t, "grad", None) is not None),
                "params_declared": len(params)})
            out["by_shape_class"] = rows
            out["by_verb"] = _byverb(bwd)
            out["by_closure"] = _bynode(bwd)
            out["slowest_calls_over_2ms"] = [
                {"us": round(n / 1e3, 1), "verb": k[0], "shape": list(k[1]),
                 "dtype": k[2].replace("DataType.", ""), "closure": c}
                for n, k, c in sorted(bwd.slowest, reverse=True)[:40]]
            f, b = out["forward"], out["backward"]
            if f.get("us_per_call") and b.get("us_per_call"):
                out["ratio_one_instrument"] = {
                    "fwd_us_per_call": f["us_per_call"], "bwd_us_per_call": b["us_per_call"],
                    "x": round(b["us_per_call"] / f["us_per_call"], 2),
                    "fwd_calls": f["verb_calls"], "bwd_calls": b["verb_calls"],
                    "note": "ONE instrument, one warm process, same rebinding on both legs. "
                            "of3t-gpugap's 44.3x came from a counter whose forward and "
                            "backward populations were never shown to be the same."}
            ag.release_pins()
            if exact_ctx is not None:
                exact_ctx.__exit__(None, None, None)
        except Exception:                                                # noqa: BLE001
            out["error"] = traceback.format_exc()[-6000:]
    out["env"]["aiclk_during"] = clk.summary()
    out["env"]["aiclk_line"] = clk.line(0)
    dump()
    b = out.get("backward", {})
    print(f"\nARM {a.arm}  crop {a.tokens}  backward {b.get('s')} s over "
          f"{b.get('verb_calls')} verb calls, {b.get('us_per_call')} us/call, "
          f"{b.get('verb_share_of_backward')} of the wall inside ttnn")
    print(out['env'].get('aiclk_line'))
    print(f"-> {a.out}")
    return 0 if b.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
