#!/usr/bin/env python3
"""BindCraft 2's on-card stacks, driven without JAX: wall per seam, then one per-op census.

Calls `EvoformerOnDevice`, `ExtraMsaOnDevice` and `TemplateOnDevice` through the same seams
(`_taped`, `_backward`) the custom_vjp in `as_jax` calls, with numpy inputs at the shapes one
design iteration hands them on the shipped `examples/pdl1.json` (multimer_v3 checkpoint):

* token layout as BindCraft 2 builds it: binder first, padded to its 32 bucket, then the
  115-residue target unpadded (`padded_prediction_complex`); 80 + 16 + 115 = 211 host tokens,
  which `_pad` rounds to 224 on card;
* Evoformer MSA depth 2 (one `msa_feat` row + one template single row), msa 256 / pair 128 ch;
  extra-MSA pair 128 ch under an all-zero `extra_msa_mask`; template act 64 ch, one template;
* dropout keeps drawn at AF2's rates (`DROPOUT_DRAWS`), as `dropout_masks` shapes them;
* one iteration = (recycles + 1) x [template, extra, evo] `_taped`, then evo, extra, template
  `_backward` once: AF2 stop-gradients every recycle but the last, and JAX still routes all of
  them through the forward rule (`EvoformerOnDevice._taped`'s sweep).

`--census` adds one iteration with every ttnn call timed between two syncs
(`perf/hallgrad/census.py`'s Recorder), tagged by seam.

    TT_VISIBLE_DEVICES=<chip> PYTHONPATH=. python perf/spd/bc2_evo_devprof.py \\
        --params ~/spd/spd-mbench/af2_params/params_model_1_multimer_v3.npz --out out/bc2evo --census

Output: <out>/devprof.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--params", required=True, help="AF2 multimer_v3 params npz (BC2's design models)")
p.add_argument("--tokens", type=int, default=224, help="card token axis; must equal pad32 of the host axis")
p.add_argument("--binder", type=int, default=80)
p.add_argument("--target", type=int, default=115, help="PD-L1 in examples/pdl1.json")
p.add_argument("--recycles", type=int, default=1, help="design_recycles (settings/core/default.json)")
p.add_argument("--stacks", default="template,extra,evo")
p.add_argument("--no-dropout", action="store_true", help="the harden stage: keeps all true, scales 1")
p.add_argument("--no-fast", action="store_true", help="skip bindcraft2.fast_round (the shipped levers)")
p.add_argument("--warm", type=int, default=2)
p.add_argument("--reps", type=int, default=3)
p.add_argument("--census", action="store_true", help="one more iteration with every ttnn call synced")
p.add_argument("--tape-census", action="store_true",
               help="after the reps: one taped Evoformer forward, then the device bytes its tape holds, by "
                    "the op that saved them (run with TT_BIO_BC2_RESIDENT_BLOCKS>=1 to see a whole block)")
p.add_argument("--wrap", choices=("census", "all"), default="all")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--skip", default="none",
               help="comma list of tt_bio.af2.SUBSTITUTION_CLASSES dropped from the Evoformer and extra-MSA "
                    "blocks (AF2PairBlock.skip): a timing leg whose arithmetic is wrong on purpose; a class's "
                    "card cost is the incumbent minus its leg")
p.add_argument("--alternate", metavar="MODULE.ATTR[,MODULE.ATTR]",
               help="boolean module flags read at call time (e.g. tt_bio.reblock_permute._PFPK), set together per "
                    "rep in ABBA order, warm-up included: an in-process A/B that host load and clock drift hit "
                    "equally; the summary is split by arm")
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402

from tt_bio import bindcraft2 as B  # noqa: E402  (jax-free at import; jax only inside as_jax & co.)

bpad = B._pad32(a.binder)
N = a.target + bpad
if B._pad32(N) != a.tokens:
    sys.exit(f"binder {a.binder} (+{bpad - a.binder} pad) + target {a.target} = {N} host tokens -> "
             f"{B._pad32(N)} on card, not --tokens {a.tokens}")
stacks = a.stacks.split(",")

from tt_bio import autograd  # noqa: E402  (imports ttnn)
from tt_bio.tenstorrent import get_device  # noqa: E402
import ttnn  # noqa: E402
from perf.hallgrad.census import CLASS, Recorder, stamp  # noqa: E402

# What `predictor` does before it builds anything: open the chip on this thread.
get_device(trace="bc2" if autograd.TRACE_RECOMPUTE else None)
name = Path(a.params).stem.removeprefix("params_")
pool = B.TrunkPool({name: a.params}, template="template" in stacks)
mem = B._Memory("auto")
evo = B.EvoformerOnDevice(pool, memory=mem)
extra = B.ExtraMsaOnDevice(pool, memory=mem)
tmpl = B.TemplateOnDevice(pool, memory=mem) if "template" in stacks else None
trunk = pool.trunk_for("")
if a.skip != "none":
    from tt_bio.af2 import SUBSTITUTION_CLASSES
    trunk.model.set_skip(n for c in a.skip.split(",") for n in SUBSTITUTION_CLASSES[c])
if not trunk.multimer:
    print(f"warning: {a.params} is not multimer_v3; BC2 designs on multimer", file=sys.stderr)

# ---------------------------------------------------------------- inputs, BC2's layout
rng = np.random.default_rng(a.seed)
f32 = np.float32
seq = np.zeros(N, f32)
seq[:a.binder] = 1
seq[bpad:] = 1
tgt = np.zeros(N, f32)
tgt[bpad:] = 1
pair_mask = np.outer(seq, seq)
msa_mask = np.stack([seq, tgt])                    # [2, N]: msa_feat row, template single row
extra_mask = np.zeros((1, N), f32)                 # BC2's extra_msa_mask (bindcraft/af2.py:134)
tmpl_mask = np.outer(tgt, tgt)                     # template pair mask: the target block only
msa = rng.standard_normal((2, N, 256), f32)
pair = rng.standard_normal((N, N, 128), f32)
act = rng.standard_normal((N, N, 64), f32)
g_msa = rng.standard_normal((2, N, 256), f32) * 1e-3
g_pair = rng.standard_normal((N, N, 128), f32) * 1e-3
g_act = rng.standard_normal((N, N, 64), f32) * 1e-3


def keeps(blocks, draws, c, msa_c=None):
    """`stack_dropout_masks`'s outputs in numpy: (msa_keep, pair_keep, scales)."""
    rate = lambda r: 0.0 if a.no_dropout else r  # noqa: E731
    pk = np.stack([rng.random((blocks, N, c)) >= rate(r) for _, _, r, _ in draws], axis=1)
    scales = [1 / (1 - rate(r)) for _, _, r, _ in draws]
    mk = None
    if msa_c:
        r0 = B.DROPOUT_DRAWS[0][2]
        mk = rng.random((blocks, 1, N, msa_c)) >= rate(r0)
        scales.insert(0, 1 / (1 - rate(r0)))
    return mk, pk, np.asarray(scales, f32)


def draw():
    out = {"evo": keeps(trunk.blocks, B.DROPOUT_DRAWS[1:], 128, 256),
           "extra": keeps(trunk.extra_blocks, B.DROPOUT_DRAWS[1:], 128)}
    if tmpl:
        out["template"] = keeps(trunk.template_blocks, B.TEMPLATE_DROPOUT_DRAWS, 64)
    return out


# ---------------------------------------------------------------- one design iteration
rec = None


def seam(phase, fn, *args):
    if rec is not None:
        rec.phase = phase
    t0 = time.perf_counter()
    out = fn("", *args)           # every seam ends in trunk.sync(), so this is its device wall
    return out, time.perf_counter() - t0


def iteration():
    T = collections.defaultdict(list)
    for _ in range(a.recycles + 1):
        d = draw()                # a fresh key per recycle; drawing is host work outside the seams
        z = pair
        if tmpl:
            _, pk, sc = d["template"]
            (_, t_tok), s = seam("template.taped", tmpl._taped, act, tmpl_mask, pk, sc)
            T["template.taped"].append(s)
        if "extra" in stacks:
            _, pk, sc = d["extra"]
            (z, x_tok), s = seam("extra.taped", extra._taped, z, extra_mask, pair_mask, pk, sc)
            T["extra.taped"].append(s)
        if "evo" in stacks:
            mk, pk, sc = d["evo"]
            (_, _, e_tok), s = seam("evo.taped", evo._taped, msa, z, msa_mask, pair_mask, mk, pk, sc)
            T["evo.taped"].append(s)
    if "evo" in stacks:
        T["evo.backward"].append(seam("evo.backward", evo._backward, e_tok, g_msa, g_pair)[1])
    if "extra" in stacks:
        T["extra.backward"].append(seam("extra.backward", extra._backward, x_tok, g_pair)[1])
    if tmpl:
        T["template.backward"].append(seam("template.backward", tmpl._backward, t_tok, g_act)[1])
    return dict(T)


_ITEM = {"BFLOAT16": 2, "FLOAT32": 4, "UINT32": 4, "INT32": 4, "UINT16": 2, "BFLOAT8_B": 1.0625,
         "BFLOAT4_B": 0.5625, "UINT8": 1}


def tape_census():
    """Walk one taped Evoformer forward's graph from its roots and sum the live device buffers its
    backward closures hold, once per buffer, by the closure that holds them and the buffer's shape."""
    mk, pk, sc = draw()["evo"]
    _, _, tok = evo._taped("", msa, pair, msa_mask, pair_mask, mk, pk, sc)
    trunk.sync()
    roots = evo._tapes._live[int(tok)]["roots"]
    pair_bytes = 256 * a.tokens ** 2
    seen, bufs = set(), set()
    by = collections.defaultdict(lambda: [0, 0.0])
    stack = [r.node for r in roots if r.node is not None]
    while stack:
        nd = stack.pop()
        if id(nd) in seen:
            continue
        seen.add(id(nd))
        stack += [q.node for q in nd.parents if getattr(q, "node", None) is not None]
        fn = nd.fn
        name = getattr(fn, "__qualname__", type(fn).__name__).replace(".<locals>", "")
        vals = []
        for c in getattr(fn, "__closure__", None) or ():
            try:
                v = c.cell_contents
            except ValueError:
                continue
            vals += list(v) if isinstance(v, (list, tuple)) else [v]
        for v in vals:
            if isinstance(v, autograd.Tensor):
                try:
                    v = v.value
                except Exception:  # a value the tape already freed holds nothing
                    continue
            if isinstance(v, list) and len(v) == 1:          # a one-slot output box
                v = v[0]
            if not isinstance(v, ttnn.Tensor):
                continue
            key = autograd._storage(v)
            if key is None or key in bufs:
                continue
            bufs.add(key)
            shape = tuple(v.padded_shape)
            nbytes = float(np.prod(shape)) * _ITEM.get(str(v.dtype).split(".")[-1], 2)
            row = by[(name, str(list(shape)), str(v.dtype).split(".")[-1])]
            row[0] += 1
            row[1] += nbytes
    rows = sorted(({"closure": k[0], "shape": k[1], "dtype": k[2], "buffers": n, "gb": b / 1e9,
                    "pair_tensors": b / pair_bytes} for k, (n, b) in by.items()), key=lambda r: -r["gb"])
    total = sum(r["gb"] for r in rows)
    print(f"\ntape census: {len(seen)} nodes, {len(bufs)} buffers, {total:.3f} GB "
          f"= {total * 1e9 / pair_bytes:.1f} pair tensors")
    for r in rows[:40]:
        print(f"{r['gb']:7.3f} GB {r['pair_tensors']:6.1f} pt {r['buffers']:5d}x {r['dtype']:9} {r['shape']:24} "
              f"{r['closure'][:80]}")
    return dict(nodes=len(seen), buffers=len(bufs), gb=total, pair_tensors=total * 1e9 / pair_bytes,
                resident=os.environ.get("TT_BIO_BC2_RESIDENT_BLOCKS"), rows=rows)


def summary(reps):
    keys = sorted({k for r in reps for k in r})
    return {k: {"median_s": float(np.median([sum(r[k]) for r in reps])),
                "per_call_median_s": float(np.median([x for r in reps for x in r[k]])),
                "calls_per_iter": len(reps[0][k])} for k in keys}


head = dict(stamp(), sha_env=os.environ.get("SPD_TREE_SHA"), argv=sys.argv, params=a.params,
            checkpoint=name, multimer=trunk.multimer, host_tokens=N, card_tokens=a.tokens,
            binder=a.binder, binder_padded=bpad, target=a.target, recycles=a.recycles,
            stacks=stacks, dropout=not a.no_dropout,
            env={k: v for k, v in sorted(os.environ.items()) if k.startswith(("TT_BIO_", "TT_METAL_"))},
            shapes=dict(evo_msa=list(msa.shape), evo_pair=list(pair.shape), msa_mask=list(msa_mask.shape),
                        pair_mask=list(pair_mask.shape), extra_mask=list(extra_mask.shape),
                        template_act=list(act.shape)))
print(json.dumps(head)[:2000], flush=True)

with autograd.exact_training(False), \
        (B.fast_round() if not a.no_fast else __import__("contextlib").nullcontext({})) as armed:
    head["fast_round"] = armed
    flag = None
    if a.alternate:
        flags = [(__import__("importlib").import_module(m), at)
                 for m, at in (f.rsplit(".", 1) for f in a.alternate.split(","))]
        flag = lambda on: [setattr(m, at, on) for m, at in flags]  # noqa: E731
    arm_of = lambda i: (0, 1, 1, 0)[i % 4]  # noqa: E731
    arms = []
    for i in range(a.warm):
        if flag:
            flag(bool(arm_of(i)))
        t0 = time.perf_counter()
        iteration()
        print(f"warm {i}: {time.perf_counter() - t0:.3f} s", flush=True)
    reps, walls = [], []
    for i in range(a.reps):
        if flag:
            flag(bool(arm_of(i)))
            arms.append(arm_of(i))
        t0 = time.perf_counter()
        reps.append(iteration())
        walls.append(time.perf_counter() - t0)
        print(f"rep {i}{f' arm {arms[-1]}' if flag else ''}: {walls[-1]:.3f} s " + " ".join(f"{k}={sum(v):.3f}" for k, v in reps[-1].items()),
              flush=True)
    doc = dict(head, iteration_s=walls, reps=reps, summary=summary(reps),
               memory_mode=dict(mem.used), calls=dict(evo=evo.calls, extra=extra.calls,
                                                        template=tmpl.calls if tmpl else None))
    from tt_bio import trimul_tail as _tt
    doc["gin_ragged_stats"] = list(_tt.GIN_RAGGED_STATS)
    if flag:
        flag(False)
        doc["arms"] = arms
        doc["by_arm"] = {str(k): dict(iteration_s=[w for w, m in zip(walls, arms) if m == k],
                                      summary=summary([r for r, m in zip(reps, arms) if m == k]))
                         for k in (0, 1)}

    if a.census:
        rec = Recorder(ttnn, trunk.device, 0)
        rec.rec = []
        rec.install_all() if a.wrap == "all" else rec.install()
        rec.active = True
        t0 = time.perf_counter()
        T = iteration()
        rec.active = False
        agg = collections.defaultdict(lambda: dict(calls=0, synced_s=0.0, flops=0.0, bytes=0.0))
        for r in rec.rec:
            d = agg[(r["phase"], r["name"], r["origin"], r["sig"], r["out"])]
            d["calls"] += 1
            for k in ("synced_s", "flops", "bytes"):
                d[k] += r[k]
        rows = sorted((dict(phase=ph, name=n, origin=o, sig=s, out=out, cls=CLASS.get(n, "other"), **d)
                       for (ph, n, o, s, out), d in agg.items()), key=lambda r: -r["synced_s"])
        by_phase, by_cls = collections.defaultdict(float), collections.defaultdict(float)
        for r in rows:
            by_phase[r["phase"]] += r["synced_s"]
            by_cls[(r["phase"].split(".")[1], r["cls"])] += r["synced_s"]
        doc["census"] = dict(iteration_s=time.perf_counter() - t0, seams=T, calls=len(rec.rec),
                             synced_total_s=sum(by_phase.values()), by_phase=dict(by_phase),
                             by_class={f"{p}/{c}": v for (p, c), v in sorted(by_cls.items())}, ops=rows)
        print(f"\ncensus: {len(rec.rec)} outermost ttnn calls, synced {sum(by_phase.values()):.3f} s "
              f"over {doc['census']['iteration_s']:.3f} s; by phase "
              + ", ".join(f"{k} {v:.3f}" for k, v in by_phase.items()))
        print(f"{'phase':18} {'op':30} {'origin':20} {'calls':>6} {'s':>8}  shape")
        for r in rows[:40]:
            print(f"{r['phase'][:17]:18} {r['name'][:29]:30} {r['origin'][:19]:20} {r['calls']:6d} "
                  f"{r['synced_s']:8.3f}  {r['sig'][:70]}")

        rec = None

    if a.tape_census:
        doc["tape_census"] = tape_census()

(a.out / "devprof.json").write_text(json.dumps(doc, indent=1, default=str))
print(json.dumps(doc["summary"], indent=1))
print(f"wrote {a.out / 'devprof.json'}", flush=True)
