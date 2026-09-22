#!/usr/bin/env python3
"""Who owns the bytes at the backward's DRAM high-water, at a crop where it COMPLETES.

`of3t-l1` sized the wall at 640 and `of3t-orchestrator` priced one lever against it: spilling
the retained `(s, z)` block boundaries. That pricing rests on two numbers neither row measured
directly -- the boundary set at 384, which was SCALED from the 640 figure at N^2, and the split
of everything else into recompute versus tape, which nothing on the record splits at all. Both
are measurable at 384, where the backward completes, so no 640 run is needed to size a 640 lever.

The join is by ADDRESS. `reports.get_buffers` is the allocator's own live list and reports an
address per buffer; every live `autograd.Tensor` reports the address of its value, its gradient
and its box. Claiming an address for exactly one owner, in a fixed precedence, turns the census
into an attribution whose parts sum to the census total by construction -- the residual is
reported rather than assumed away, and it is the honest part of the answer.

WHO COUNTS AS WHAT. The generation is the discriminator, not a name:

  boundaries   `autograd._CKPT_PINS`, the inputs every checkpointed segment pinned across its
               untaped forward. This IS the retained boundary set -- the object the spill lever
               would move -- read off the list the code itself keeps.
  weights      the declared leaves, and `weight_grads` their gradients.
  tape         a tensor the FORWARD left behind: on the reverse-topological order built from the
               root before the backward starts, or one of the forward's own inputs.
  cotangents   the gradients in flight on those same tape tensors.
  recompute    everything live that is NEITHER -- born inside the backward, which at a
               checkpointed trunk means inside a `checkpoint._recompute`. This is the live
               recompute working set, identified by construction rather than by shape.
  unattributed the rest of the allocator's list, reported with its size histogram.

The walk is host-side only: `is_allocated`, `buffer_address` and the buffer list read state, they
do not allocate, so the attribution cannot move the number it attributes. It is not free in WALL
CLOCK though, so it fires only when the high-water has advanced by `--walk-step-mb`; the gap
between the last walk and the true peak is reported with every result.

`host_quiet.py` green is not required and was not sought: a DRAM high-water is a property of what
is resident on one device, not of how long anything took, and a co-tenant on another card cannot
change it. The card is pinned and recorded all the same.

    split.py --tokens 384 --out perf/of3t_crop640/out/split_384.json
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
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

from perf.clocksample import during                                    # noqa: E402
from perf.of3t_memory.alloc_profile import Peak, _swap_watch           # noqa: E402
from perf.of3t_l1.ladder import _tag_stacks, _restore                  # noqa: E402
from perf.of3t_perf import step as S                                   # noqa: E402

# Precedence. First claim on an address wins, so a recompute duplicate that SHARES a pinned
# boundary's buffer is charged to the boundary -- the buffer is the boundary's, and the lever
# that spills it moves those bytes whoever else is looking at them.
OWNERS = ("boundaries", "weights", "weight_grads", "tape", "cotangents", "recompute",
          "other_leaf")


class Attrib:
    """The address -> owner join, rebuilt from scratch at each sample."""

    def __init__(self, ag, ttnn, reports, dev, dram_banks, l1_banks):
        self.ag, self.ttnn, self.reports, self.dev = ag, ttnn, reports, dev
        self.banks = {"DRAM": dram_banks, "L1": l1_banks}
        self.outer_ids: set = set()      # ids of Tensors the FORWARD left behind
        self.param_ids: set = set()

    def note_outer(self, objs):
        for t in objs:
            if isinstance(t, self.ag.Tensor):
                self.outer_ids.add(id(t))

    def _key(self, v):
        """(buffer space, address) for a live device handle, or None."""
        if v is None:
            return None
        try:
            if not v.is_allocated():
                return None
            bt = "L1" if v.memory_config().buffer_type == self.ttnn.BufferType.L1 else "DRAM"
            return (bt, int(v.buffer_address()))
        except Exception:                                                   # noqa: BLE001
            return None

    def _table(self):
        """The allocator's live list keyed by (space, address). Bytes across banks, as the
        census reports them, so an owner's sum is comparable to the peak directly."""
        table, dup = {}, 0
        for b in self.reports.get_buffers(self.dev):
            bt = "L1" if str(b.buffer_type).upper().endswith("L1") else "DRAM"
            k = (bt, int(b.address))
            sz = int(b.max_size_per_bank) * self.banks[bt]
            if k in table:
                dup += 1
                table[k] = max(table[k], sz)
            else:
                table[k] = sz
        return table, dup

    def _live_tensors(self):
        seen, out = set(), []
        for obj in gc.get_objects():
            if type(obj).__name__ != "Tensor" or id(obj) in seen:
                continue
            if not isinstance(obj, self.ag.Tensor):
                continue
            seen.add(id(obj))
            out.append(obj)
        return out

    def sample(self, params, label, dram_now):
        t0 = time.perf_counter()
        table, dup = self._table()
        claimed: dict = {}
        shapes = {o: Counter() for o in OWNERS}

        def claim(v, owner, shape_from=None):
            k = self._key(v)
            if k is None or k in claimed or k not in table:
                return False
            claimed[k] = owner
            try:
                s = tuple(int(d) for d in (shape_from if shape_from is not None else v).shape)
                shapes[owner][(s, str(v.dtype), k[0])] += 1
            except Exception:                                               # noqa: BLE001
                pass
            return True

        pins = list(self.ag._CKPT_PINS)
        pin_ids = {id(t) for t in pins}
        for t in pins:
            claim(getattr(t, "value", None), "boundaries")
        for t in params.values():
            claim(getattr(t, "value", None), "weights")
        for t in params.values():
            claim(getattr(t, "grad", None), "weight_grads", shape_from=t)

        live = self._live_tensors()
        buckets = {o: [] for o in OWNERS}
        for t in live:
            if id(t) in pin_ids or id(t) in self.param_ids:
                continue
            if id(t) in self.outer_ids:
                buckets["tape"].append(t)
            elif t.node is not None:
                buckets["recompute"].append(t)
            else:
                buckets["other_leaf"].append(t)
        # Values before gradients, and the outer tape before the inner one: a buffer an inner
        # duplicate shares with an outer tensor belongs to the forward that made it.
        for o in ("tape", "recompute", "other_leaf"):
            for t in buckets[o]:
                claim(getattr(t, "value", None), o)
                box = getattr(t, "box", None)
                if box:
                    claim(box[0], o, shape_from=t)
        for t in buckets["tape"]:
            claim(getattr(t, "grad", None), "cotangents", shape_from=t)
        for o in ("recompute", "other_leaf"):
            for t in buckets[o]:
                claim(getattr(t, "grad", None), o, shape_from=t)

        by = {o: {"n": 0, "b": 0, "dram_b": 0, "l1_b": 0} for o in OWNERS}
        for k, owner in claimed.items():
            r = by[owner]
            r["n"] += 1
            r["b"] += table[k]
            r["dram_b" if k[0] == "DRAM" else "l1_b"] += table[k]
        rest = [table[k] for k in table if k not in claimed]
        rest_dram = [table[k] for k in table if k not in claimed and k[0] == "DRAM"]
        total = sum(table.values())
        dram_total = sum(v for k, v in table.items() if k[0] == "DRAM")

        top = lambda c: [{"shape": list(s), "dtype": d, "space": sp, "n": n}
                         for (s, d, sp), n in c.most_common(8)]
        return {
            "label": label,
            "dram_now_b": dram_now,
            "census_total_b": total,
            "census_dram_b": dram_total,
            "buffers": len(table),
            "duplicate_addresses": dup,
            "live_taped_tensors": len(live),
            "by_owner": by,
            "owner_share_of_dram_pct": {o: round(100 * by[o]["dram_b"] / dram_total, 2)
                                        for o in OWNERS} if dram_total else {},
            "unattributed": {
                "n": len(rest), "b": sum(rest), "dram_b": sum(rest_dram),
                "dram_share_pct": round(100 * sum(rest_dram) / dram_total, 2) if dram_total else 0,
                "top_by_bytes": [{"each_b": s, "n": n, "total_b": s * n} for s, n in
                                 sorted(Counter(rest).items(), key=lambda kv: -kv[0] * kv[1])[:10]],
            },
            "shapes": {o: top(shapes[o]) for o in OWNERS},
            "walk_s": round(time.perf_counter() - t0, 3),
        }


def _thin(w):
    """One walk, owner totals only. The shape censuses are what make a sample large, and the
    profile needs the sums; the full sample is kept at the peak and at the first recompute."""
    t = {"dram_b": w["dram_now_b"], "buffers": w["buffers"], "walk_s": w["walk_s"],
         "unattributed_b": w["unattributed"]["dram_b"]}
    for o in OWNERS:
        t[o] = w["by_owner"][o]["dram_b"]
    top = w["shapes"]["recompute"][:1]
    t["recompute_top_shape"] = top[0]["shape"] if top else None
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--walk-step-mb", type=int, default=32,
                    help="advance in the DRAM high-water that earns a fresh attribution walk; "
                         "the gap between the last walk and the peak is reported")
    ap.add_argument("--walk-from-gb", type=float, default=0.0,
                    help="do not walk below this DRAM high-water. The ramp up to the peak is "
                         "not the question; the peak is, and a walk costs wall clock")
    ap.add_argument("--walk-budget-s", type=float, default=240.0,
                    help="total wall clock the walks may spend. On overrun the step doubles, so "
                         "a slow walk costs resolution at the peak rather than the run")
    ap.add_argument("--probe-every", type=int, default=1)
    ap.add_argument("--dead-values", choices=("on", "off"), default=None,
                    help="tt_bio.autograd.DROP_DEAD_VALUES for this rung. Recorded in the "
                         "artifact either way: which arm a byte figure came from is not "
                         "something a later reader should have to infer from a commit hash")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_lease_cards": os.environ.get("TT_BIO_LEASE_CARDS"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {REPO} rev-parse --abbrev-ref HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        a.out.write_text(json.dumps(out, indent=1, default=str))
    dump()

    with during() as clk:
        try:
            import ttnn
            from ttnn._ttnn import reports
            from tt_bio import autograd as ag
            if a.dead_values is not None:
                ag.DROP_DEAD_VALUES = a.dead_values == "on"
            out["env"]["drop_dead_values"] = bool(getattr(ag, "DROP_DEAD_VALUES", False))
            from tt_bio import taped_ttnn as TT
            from tt_bio.tenstorrent import get_device

            held, _meta = S.capture(a.tokens, out)
            trunk = held["trunk"][0]
            dev = get_device()
            out["env"]["arch"] = str(dev.arch())
            params = S.declare_weights(trunk, out)
            dump()

            peak = Peak(dev, ttnn, reports)
            peak.every = a.probe_every
            blocks, origs = _tag_stacks(peak)
            att = Attrib(ag, ttnn, reports, dev, peak.dram_banks, peak.l1_banks)
            att.param_ids = {id(t) for t in params.values()}

            trace: list = []
            state = {"last_walk_b": 0, "arm": False, "n": 0, "spent": 0.0, "first_rec": False,
                     "step_b": a.walk_step_mb << 20, "floor_b": int(a.walk_from_gb * (1 << 30))}
            _probe = peak.probe

            def probe(verb):
                before = peak.dram_hw
                _probe(verb)
                if peak.dram_hw != before:
                    peak.stage_at_peak = peak.stage
                    if (state["arm"] and peak.dram_hw >= state["floor_b"]
                            and peak.dram_hw >= state["last_walk_b"] + state["step_b"]):
                        state["last_walk_b"] = peak.dram_hw
                        state["n"] += 1
                        w = att.sample(params, "backward high-water", peak.dram_hw)
                        out["backward"]["walk"] = w
                        out["backward"]["walks"] = state["n"]
                        # The PROFILE, not just its maximum. At 384 the peak lands late, with
                        # the boundary set already drained by `_retire`, while 640 died early
                        # with it full -- two different moments of the same backward, and a
                        # single sample at the maximum cannot tell them apart.
                        trace.append(_thin(w))
                        out["backward"]["trace"] = trace
                        if not state["first_rec"] and w["by_owner"]["recompute"]["dram_b"]:
                            state["first_rec"] = True
                            out["backward"]["first_recompute"] = w
                        state["spent"] += w["walk_s"]
                        # Resolution at the peak is worth wall clock, but not the run. Doubling
                        # the step on overrun degrades the former and protects the latter.
                        if state["spent"] > a.walk_budget_s:
                            state["step_b"] *= 2
                            state["spent"] = 0.0
                            out["backward"]["walk_step_doublings"] = \
                                out["backward"].get("walk_step_doublings", 0) + 1
            peak.probe = probe

            gc.collect()
            saved: list = []
            peak.reset()
            peak.stage_at_peak = None

            snap_args, snap_kwargs = held["trunk_snap"]
            args_ = S._rehydrate(snap_args, dev)
            kwargs_ = {k: v for k, v in S._rehydrate(snap_kwargs, dev).items()
                       if k != "progress_fn"}
            trunk.num_cycles = a.cycles
            out["forward"] = {"cycles": a.cycles, "inputs_dram_b": peak.dram_now()}

            z = None
            t0 = time.perf_counter()
            with ag.tape():
                _swap_watch(peak, True, saved)
                try:
                    _s, z = trunk(*args_, **kwargs_)
                finally:
                    _swap_watch(peak, False, saved)
            ttnn.synchronize_device(dev)
            out["forward"]["ok"] = True
            out["forward"]["s"] = round(time.perf_counter() - t0, 2)
            out["forward"]["dram_peak_b"] = peak.dram_hw
            out["forward"]["l1_peak_b"] = peak.l1_hw
            out["forward"]["at_verb"] = peak.at_verb
            out["forward"]["blocks_entered"] = dict(blocks)
            dump()

            # The forward's inputs are the tape's too: a tensor the trunk was HANDED is not
            # something the backward made, and calling it recompute would flatter the lever.
            def _walk_args(o, depth=0):
                if depth > 3:
                    return
                if isinstance(o, ag.Tensor):
                    att.note_outer([o])
                elif isinstance(o, (list, tuple)):
                    for x in o:
                        _walk_args(x, depth + 1)
                elif isinstance(o, dict):
                    for x in o.values():
                        _walk_args(x, depth + 1)
            _walk_args(args_)
            _walk_args(kwargs_)
            att.note_outer([_s, z])
            order = ag._reverse_topo([z])
            att.note_outer(order)
            out["backward"] = {"tape_nodes": len(order),
                               "outer_tensors_noted": len(att.outer_ids),
                               "ckpt_pins": len(ag._CKPT_PINS)}
            # BEFORE the backward allocates anything: the retained boundary set complete, and
            # nothing from the backward yet. This is deliverable 1's direct measurement.
            out["after_forward"] = att.sample(params, "after forward, before backward",
                                              peak.dram_now())
            dump()

            import torch
            zr = z.value
            seed = ttnn.from_torch(torch.ones(tuple(int(d) for d in zr.shape)),
                                   layout=ttnn.TILE_LAYOUT, device=dev, dtype=zr.dtype)
            peak.reset()
            peak.stage_at_peak = None
            state["arm"] = True
            t0 = time.perf_counter()
            rs = TT.recompute_scope()
            rs.__enter__()
            try:
                _swap_watch(peak, True, saved)
                try:
                    ag.backward([z], [seed])
                    ttnn.synchronize_device(dev)
                finally:
                    _swap_watch(peak, False, saved)
                out["backward"]["ok"] = True
            except Exception as e:                                          # noqa: BLE001
                out["backward"]["ok"] = False
                out["backward"]["error"] = traceback.format_exc()[-6000:]
                out["backward"]["error_head"] = str(e).split("backtrace")[0][:900]
                out["backward"]["stage_at_failure"] = peak.stage
            finally:
                rs.__exit__(None, None, None)
                state["arm"] = False
            out["backward"]["s"] = round(time.perf_counter() - t0, 2)
            out["backward"]["dram_peak_b"] = peak.dram_hw
            out["backward"]["l1_peak_b"] = peak.l1_hw
            out["backward"]["at_verb"] = peak.at_verb
            out["backward"]["at_stage"] = peak.stage_at_peak
            out["backward"]["dram_live_allocs"] = (peak.census or {}).get("DRAM", {}).get("count")
            out["backward"]["census"] = peak.census
            out["backward"]["free"] = peak.free
            out["backward"]["params_with_grad"] = sum(
                1 for t in params.values() if getattr(t, "grad", None) is not None)
            out["backward"]["params_declared"] = len(params)
            if trace:
                colive = max(trace, key=lambda t: t["boundaries"] + t["recompute"])
                out["backward"]["max_colive_boundaries_plus_recompute"] = {
                    "dram_b": colive["dram_b"], "boundaries_b": colive["boundaries"],
                    "recompute_b": colive["recompute"],
                    "sum_b": colive["boundaries"] + colive["recompute"],
                    "share_of_that_moment_pct": round(
                        100 * (colive["boundaries"] + colive["recompute"]) / colive["dram_b"], 2)}
                out["backward"]["max_boundaries_b"] = max(t["boundaries"] for t in trace)
                out["backward"]["max_recompute_b"] = max(t["recompute"] for t in trace)
            w = out["backward"].get("walk")
            if w:
                gap = peak.dram_hw - w["dram_now_b"]
                out["backward"]["walk_gap_to_peak_b"] = gap
                out["backward"]["walk_gap_to_peak_pct"] = round(100 * gap / peak.dram_hw, 3)
                out["backward"]["walk_step_b_final"] = state["step_b"]
            ag.release_pins()
            _restore(origs)
        except Exception:                                                   # noqa: BLE001
            out["error"] = traceback.format_exc()[-6000:]
    out["env"]["aiclk_during"] = clk.summary()
    out["env"]["aiclk_line"] = clk.line(0)
    dump()

    f, b = out.get("forward", {}), out.get("backward", {})
    GB = 1 << 30
    print("tokens %d  fwd %s %.3f GB   bwd %s %.3f GB / %s allocs  grads %s/%s  walks %s"
          % (a.tokens, "PASS" if f.get("ok") else "FAIL", f.get("dram_peak_b", 0) / GB,
             "PASS" if b.get("ok") else "FAIL", b.get("dram_peak_b", 0) / GB,
             b.get("dram_live_allocs"), b.get("params_with_grad"), b.get("params_declared"),
             b.get("walks")), flush=True)
    for key in ("after_forward", None):
        s = out.get(key) if key else b.get("walk")
        if not s:
            continue
        print("  %-34s dram %.3f GB" % (s["label"], s["census_dram_b"] / GB), flush=True)
        for o in OWNERS:
            r = s["by_owner"][o]
            if r["dram_b"]:
                print("    %-13s %8.3f GB  %5.2f %%  %5d buffers"
                      % (o, r["dram_b"] / GB, s["owner_share_of_dram_pct"][o], r["n"]), flush=True)
        u = s["unattributed"]
        print("    %-13s %8.3f GB  %5.2f %%  %5d buffers"
              % ("unattributed", u["dram_b"] / GB, u["dram_share_pct"], u["n"]), flush=True)
    if b.get("walk"):
        print("  walk taken %.3f GB below the peak (%.3f %%)"
              % (b.get("walk_gap_to_peak_b", 0) / GB, b.get("walk_gap_to_peak_pct", 0)), flush=True)
    print(out["env"]["aiclk_line"], flush=True)
    if out.get("error"):
        print("ERROR:", out["error"][-1200:], flush=True)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
