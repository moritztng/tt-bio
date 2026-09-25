#!/usr/bin/env python3
"""Every ttnn verb of one OF3 training step, with its operand SHAPES, labelled by step part.

The question this row owns is which roof binds each of the six parts of the 466.70 s step.
You cannot answer it from a wall clock: a part that is issuing 0-FLOP data movement has no
compute roof to be a fraction of, and `of3t-intensity`'s brief exists because this campaign
twice quoted a %-of-peak against a roof nobody had established.

So this script measures the two numerators directly. For every call the shipped code makes
into ttnn it records the verb, the part of the step it was issued from, and the logical AND
padded shape, dtype and buffer type of every operand and every result. FLOP and bytes are
then arithmetic over that census (`cost.py`), auditable line by line and re-runnable without
a card.

WHERE THE HOOK SITS, and why it is not `alloc_profile._Watch`. That watcher rebinds the name
`ttnn` inside each tt_bio module. `taped_ttnn._swap` decides what to shim with
`getattr(mod, "ttnn", None) is ttnn`, so a module holding a watcher is invisible to it -- and
the boundary that matters is not the forward's (ladder.py installs the watcher INSIDE the
tape, which works) but `recompute_scope()`, which re-shims from inside a backward closure. A
watcher live during the backward would silently stop every checkpointed segment from
re-taping. This script wraps the callables ON THE REAL ttnn MODULE instead. Module globals
keep holding `ttnn` itself, every identity test still answers what it answered, and the tape
proxy's own `getattr(real, name)` resolves to the wrapper.

The consequence for the AXIS, stated rather than glossed: this counts REAL verb calls, one
level below the tape proxy. For the backward that is the same population `of3t-stepfloor`
counted (168,922) because `tt_bio.autograd` is never shimmed and its closures call real verbs
directly. For the taped forward it is NOT: one proxy verb can issue several real ones, so
this count is at or above the 84,996 that row reports, and the two are not the same
instrument.

COUNTING, NOT TIMING. Every number here is a shape, a byte or a call count, which is why a
loud box and a cold rep cost it nothing: `--reps 1` is enough, where a timing row needs three.
No wall clock from this script is a measurement and none is reported.

    census.py --tokens 384 --cycles 4 --samples 4 --out out/census_384.json
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
import types
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.clocksample import during                       # noqa: E402
from perf.of3t_perf import step as S                      # noqa: E402
from perf.of3t_stepfloor import fullstep as F             # noqa: E402

SEED = 20260921            # fullstep's, so the two harnesses see the same schedule draw

# --- the census ---------------------------------------------------------------------------

PART = ["setup"]                   # one-slot box, cheap to read in the hot path
COUNTS: Counter = Counter()        # (part, verb, signature) -> calls
DEPTH = [0]                        # a verb that calls another verb is counted once, at the top

# Bytes per element. bfloat8_b is a block format: 16 mantissa bytes plus one shared exponent
# per 16-element face, so 17/16. Reading it off the tensor's own padded volume is what makes
# the byte count a measurement rather than a shape guess.
ESIZE = {"BFLOAT16": 2.0, "FLOAT32": 4.0, "UINT32": 4.0, "INT32": 4.0, "UINT16": 2.0,
         "UINT8": 1.0, "BFLOAT8_B": 17.0 / 16.0, "BFLOAT4_B": 9.0 / 16.0}


def _tinfo(x):
    """(logical volume, padded volume, dtype, buffer type) for anything tensor-shaped."""
    try:
        shp = tuple(int(d) for d in x.shape)
    except Exception:
        return None
    try:
        pad = tuple(int(d) for d in x.padded_shape)
    except Exception:
        pad = shp
    try:
        dt = str(x.dtype).rsplit(".", 1)[-1].upper()
    except Exception:
        dt = "?"
    try:
        bt = str(x.memory_config().buffer_type).rsplit(".", 1)[-1].upper()
    except Exception:
        bt = "HOST"
    return (shp, pad, dt, bt)


def _walk(v, seen, out, budget=[0]):
    """Tensor-shaped leaves of an argument, deduped by identity within the call."""
    if budget[0] > 64:
        return
    if isinstance(v, (list, tuple)):
        for e in v:
            _walk(e, seen, out, budget)
        return
    if isinstance(v, dict):
        for e in v.values():
            _walk(e, seen, out, budget)
        return
    if id(v) in seen:
        return
    t = _tinfo(v)
    if t is None:
        return
    seen.add(id(v))
    budget[0] += 1
    out.append(t)


def _sig(ins, outs):
    f = lambda t: f"{'x'.join(map(str, t[0]))}|{'x'.join(map(str, t[1]))}|{t[2]}|{t[3]}"
    return ";".join(map(f, ins)) + " -> " + ";".join(map(f, outs))


def _wrap(fn, qual):
    def call(*a, _f=fn, _q=qual, **k):
        if DEPTH[0]:
            return _f(*a, **k)
        seen, ins = set(), []
        budget = [0]
        _walk(a, seen, ins, budget)
        _walk(k, seen, ins, budget)
        DEPTH[0] += 1
        try:
            r = _f(*a, **k)
        finally:
            DEPTH[0] -= 1
        outs = []
        _walk(r, set(), outs, [0])
        COUNTS[(PART[0], _q, _sig(ins, outs))] += 1
        return r
    call.__name__ = getattr(fn, "__name__", qual)
    return call


def install(ttnn) -> int:
    """Wrap every plain callable on the ttnn module and its namespaces, in place."""
    n = 0
    stack = [(ttnn, "")]
    seen = {id(ttnn)}
    while stack:
        mod, pre = stack.pop()
        for name in dir(mod):
            if name.startswith("__"):
                continue
            try:
                attr = getattr(mod, name)
            except Exception:
                continue
            qual = pre + name
            if isinstance(attr, types.ModuleType):
                if id(attr) in seen or not getattr(attr, "__name__", "").startswith("ttnn"):
                    continue
                seen.add(id(attr))
                stack.append((attr, qual + "."))
            elif callable(attr) and not isinstance(attr, type) and \
                    not getattr(attr, "_of3t_census", False):
                w = _wrap(attr, qual)
                w._of3t_census = True
                try:
                    setattr(mod, name, w)
                    n += 1
                except Exception:
                    pass
    return n


def dump_counts(path: Path):
    rows = [{"part": p, "verb": v, "sig": s, "n": n} for (p, v, s), n in COUNTS.items()]
    rows.sort(key=lambda r: (r["part"], -r["n"]))
    path.write_text(json.dumps(rows))
    return len(rows)


# --- the step -------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--stage", default="initial_training")
    ap.add_argument("--no-exact", action="store_true",
                    help="run the step inside `autograd.exact_training(False)`, which is the "
                         "configuration the 466.70 s reference step was measured at: `502ed112e` "
                         "put exact float64 HOST softmax and LayerNorm on by default on the "
                         "training tape two days after that run, so main's taped step and the "
                         "step this sprint ranks against are different objects. This arm counts "
                         "the reference one")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {REPO} rev-parse --abbrev-ref HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg_start": os.getloadavg()},
        "config": {"crop": a.tokens, "cycles_pinned": a.cycles,
                   "diffusion_samples": a.samples, "stage": a.stage, "reps": 1,
                   "exact_training": not a.no_exact,
                   "axis": "real ttnn verb calls, one level below the tape proxy"}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    counts_path = a.out.with_suffix(".counts.json")
    dump = lambda: a.out.write_text(json.dumps(out, indent=1, default=str))
    dump()

    with during() as clk:
        try:
            import numpy as np
            import ttnn
            from tt_bio import autograd as ag
            from tt_bio.tenstorrent import get_device
            from tt_bio.train.losses import of3_loss_weights
            from tt_bio.train.optim import AdamW

            out["env"]["wrapped_callables"] = install(ttnn)
            print(f"[census] wrapped {out['env']['wrapped_callables']} ttnn callables",
                  flush=True)

            held, _meta = S.capture(a.tokens, out)
            trunk = held["trunk"][0]
            if "sampler" not in held:
                raise SystemExit("the fold never reached the sampler; no diffusion half")
            sampler, sargs, _skw = held["sampler"]
            dev = get_device()
            out["env"]["arch"] = str(dev.arch())
            params = F.declare_all(trunk, sampler, out)
            weights = of3_loss_weights(a.stage)
            opt = AdamW(params, lr=3e-4)
            dump()

            rng = np.random.default_rng(SEED)
            for t in params.values():
                t.grad = None
            ttnn.synchronize_device(dev)
            COUNTS.clear()
            # The arm switch. `exact_training` is a stack, not a flag, so this covers the tape,
            # the backward and any recompute inside it -- which is the whole of the difference.
            exact_ctx = ag.exact_training(not a.no_exact)
            exact_ctx.__enter__()
            out["exact"] = {"ops_a_tape_would_install": list(ag.exact_training_ops()),
                            "softmax_before": dict(ag.EXACT_SOFTMAX_STATS),
                            "layer_norm_before": dict(ag.EXACT_LAYER_NORM_STATS)}

            # --- 1. trunk ---------------------------------------------------------------
            PART[0] = "trunk_nograd_prefix"
            if a.cycles > 1:
                F.trunk_forward(trunk, held, a.cycles - 1, taped=False)
                ttnn.synchronize_device(dev)

            ctx = ag.tape()
            d_out = {}
            with ctx:
                PART[0] = "trunk_taped_cycle"
                s_tr, z_tr = F.trunk_forward(trunk, held, 1, taped=True)
                ttnn.synchronize_device(dev)
                # --- 2. diffusion --------------------------------------------------------
                PART[0] = "diffusion"
                roots, _keep, rep = F.diffusion_train(sampler, sargs, s_tr, z_tr,
                                                      a.samples, rng, d_out)
            out["diffusion"] = d_out

            # --- 3. loss heads ---------------------------------------------------------
            PART[0] = "loss_heads"
            l_out = {}
            seeds = F.host_losses(roots, rep, weights, rng, l_out)
            out["losses"] = l_out

            # --- 4. seed upload --------------------------------------------------------
            PART[0] = "seed_upload"
            import torch
            cot = []
            for r, g in zip(roots, seeds):
                raw = ag._unwrap(r)
                shp = tuple(int(d) for d in raw.shape)
                h = torch.from_numpy(np.asarray(g, "float32")).reshape(shp) \
                    if g is not None else torch.ones(shp)
                cot.append(ttnn.from_torch(h, layout=ttnn.TILE_LAYOUT, device=dev,
                                           dtype=raw.dtype))
            out["tape_nodes"] = len(ag._reverse_topo(list(roots)))
            dump_counts(counts_path)
            dump()

            # --- 5. backward -----------------------------------------------------------
            PART[0] = "backward"
            print(f"[census] backward over {out['tape_nodes']} nodes", flush=True)
            ag.backward(list(roots), cot)
            ttnn.synchronize_device(dev)
            got = sum(1 for t in params.values() if getattr(t, "grad", None) is not None)
            out["params_with_grad"] = f"{got} of {len(params)}"
            out["backward_valid"] = bool(got)
            dump_counts(counts_path)
            dump()

            # --- 6. optimizer ----------------------------------------------------------
            PART[0] = "optimizer"
            if got:
                opt.step()
                params.rebind()
            PART[0] = "teardown"
            exact_ctx.__exit__(None, None, None)
            out["exact"]["softmax_after"] = dict(ag.EXACT_SOFTMAX_STATS)
            out["exact"]["layer_norm_after"] = dict(ag.EXACT_LAYER_NORM_STATS)

            out["calls_by_part"] = dict(Counter(
                {p: 0 for p in ("trunk_nograd_prefix", "trunk_taped_cycle", "diffusion",
                                "loss_heads", "seed_upload", "backward", "optimizer")}))
            for (p, _v, _s), n in COUNTS.items():
                out["calls_by_part"][p] = out["calls_by_part"].get(p, 0) + n
            out["distinct_signatures"] = dump_counts(counts_path)
            out["counts_file"] = str(counts_path)
            out["ok"] = True
        except BaseException as e:                                    # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"
            out["traceback"] = traceback.format_exc()
            out["ok"] = False
            try:
                dump_counts(counts_path)
                out["counts_file"] = str(counts_path)
            except Exception:
                pass
            print(out["traceback"], flush=True)
        out["env"]["aiclk_during"] = clk.summary()
        out["env"]["aiclk_line"] = clk.line(0)
        out["env"]["loadavg_end"] = os.getloadavg()
    dump()
    print(out["env"]["aiclk_line"], flush=True)
    print(json.dumps(out.get("calls_by_part", {}), indent=1), flush=True)
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
