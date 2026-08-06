#!/usr/bin/env python3
"""Interleaved eager-vs-traced A/B for the ESMC single-sequence embed path.

Reproduces (or refutes) the trace-capture win measured on the ttnn-0.75 scout
branch before it ships: ESMC-300M single-sequence embed, eager vs traced, on
shipped ttnn 0.68. Two cells:

  * single  — one ubiquitin (76 aa) through the shipped ``embed_sequences``
    path (B=1, bucketed to Lb=128 with padding masks), tracing auto-selected
    by ``ESMC.forward`` exactly as in production.
  * batch4  — four proteins through ``_batch_tokens`` (B=4) driven straight
    into ``_forward_eager`` / ``_forward_traced`` to re-check the measured
    batch-path ceiling (~1.02x: batching already amortizes dispatch, so the
    shipped policy keeps B>1 eager).

Method (skill ttnn-perf-profiling): legs interleaved rep-by-rep against
thermal drift; every timed region bracketed by ``ttnn.synchronize_device``;
kernel/program caches warmed and discarded first; min/median/max per leg plus
the per-leg spread as the noise floor. Every interleaved rep also bit-compares
traced vs eager outputs — the run doubles as the two-gate corruption check
(traced==eager bit-identical; eager legs running between replays, the
ttnn-trace-interleaved-eager-corruption trap).

Usage:
    TT_VISIBLE_DEVICES=0 python3 scripts/esmc_trace_ab.py [--model esmc-300m] \
        [--reps 50] [--out trace_ab.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esmc6b_embed_parity import ESMC_SEQS  # noqa: E402

from tt_bio import esmc as tt_esmc  # noqa: E402


def stats(ms):
    a = np.array(ms)
    return {"ms": [round(x, 3) for x in ms], "min": float(a.min()),
            "median": float(np.median(a)), "mean": float(a.mean()),
            "max": float(a.max()), "spread_max_over_min": float(a.max() / a.min())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m", choices=list(tt_esmc.CONFIGS))
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", default="")
    ap.add_argument("--mode", default="ab", choices=["ab", "eager"],
                    help="eager: pure-eager control process (no trace region, no "
                         "capture) — removes any live-trace allocator effect from "
                         "the eager legs.")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    print(f"tt_bio: {tt_esmc.__file__}", flush=True)
    model = tt_esmc.load_esmc(args.model, trace=(args.mode == "ab"))
    dev = model.tt_device
    res = {"model": args.model, "reps": args.reps,
           "trace_region_size": int(tt_esmc.trace_region_size()),
           "tt_bio_file": tt_esmc.__file__, "cells": {}}

    def sync():
        ttnn.synchronize_device(dev)

    def emb_equal(a, b):
        return all(np.array_equal(np.asarray(x.per_residue), np.asarray(y.per_residue))
                   and np.array_equal(np.asarray(x.pooled), np.asarray(y.pooled))
                   for x, y in zip(a, b))

    # ── single-sequence cell (shipped embed_sequences path, B=1) ──
    single = {"ubiquitin": ESMC_SEQS["ubiquitin"]}
    if args.mode == "eager":
        for _ in range(3):
            tt_esmc.embed_sequences(model, single)
        sync()
        ms = []
        for _ in range(args.reps):
            sync()
            t0 = time.perf_counter()
            tt_esmc.embed_sequences(model, single)
            sync()
            ms.append((time.perf_counter() - t0) * 1e3)
        res["cells"]["single_pure_eager"] = stats(ms)
        print(f"single  PURE eager (no trace in process):  min {np.min(ms):7.2f}  "
              f"median {np.median(ms):7.2f}  max {np.max(ms):7.2f} ms "
              f"(spread {np.max(ms)/np.min(ms):.2f}x)", flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=1)
            print("wrote", args.out)
        return 0

    model.trace = False
    for _ in range(3):                       # warm eager compile + caches
        tt_esmc.embed_sequences(model, single)
    model.trace = True
    for _ in range(3):                       # 1st sight eager, 2nd captures, 3rd replays
        tt_esmc.embed_sequences(model, single)
    sync()
    assert len(model._trace_cache) == 1, "single-seq trace was not captured"
    if hasattr(ttnn, "get_trace_buffers_size"):
        res["trace_buffers_bytes"] = int(ttnn.get_trace_buffers_size(dev))

    eager_ms, traced_ms = [], []
    bit_ok = True
    for _ in range(args.reps):
        model.trace = False
        sync()
        t0 = time.perf_counter()
        out_e = tt_esmc.embed_sequences(model, single)
        sync()
        eager_ms.append((time.perf_counter() - t0) * 1e3)
        model.trace = True
        sync()
        t0 = time.perf_counter()
        out_t = tt_esmc.embed_sequences(model, single)
        sync()
        traced_ms.append((time.perf_counter() - t0) * 1e3)
        bit_ok &= emb_equal(out_e, out_t)    # eager ran between replays: gate (b)
    res["cells"]["single"] = {"eager": stats(eager_ms), "traced": stats(traced_ms),
                              "speedup_min": float(np.min(eager_ms) / np.min(traced_ms)),
                              "speedup_median": float(np.median(eager_ms) / np.median(traced_ms)),
                              "bit_identical_interleaved": bool(bit_ok)}
    print(f"single  eager   min {np.min(eager_ms):7.2f}  median {np.median(eager_ms):7.2f}  "
          f"max {np.max(eager_ms):7.2f} ms (spread {np.max(eager_ms)/np.min(eager_ms):.2f}x)", flush=True)
    print(f"single  traced  min {np.min(traced_ms):7.2f}  median {np.median(traced_ms):7.2f}  "
          f"max {np.max(traced_ms):7.2f} ms (spread {np.max(traced_ms)/np.min(traced_ms):.2f}x)", flush=True)
    print(f"single  speedup {np.median(eager_ms)/np.median(traced_ms):.3f}x median  "
          f"bit-identical interleaved: {'PASS' if bit_ok else 'FAIL'}", flush=True)
    if not bit_ok:
        sys.exit("TRACE CORRECTNESS GATE FAILED (single)")

    # ── batch4 cell (forward level, explicit eager vs traced) ──
    batch4 = {k: ESMC_SEQS[k] for k in ("trpcage", "gb1", "ubiquitin", "lysozyme")}
    ids, lens, am, kv = tt_esmc._batch_tokens(list(batch4.values()))
    for _ in range(3):
        model._forward_eager(ids, am, kv)
    ref_l, ref_e = None, None
    e_ms, t_ms = [], []
    bit4 = True
    for i in range(args.reps):
        sync()
        t0 = time.perf_counter()
        lg_e, em_e = model._forward_eager(ids, am, kv)
        sync()
        e_ms.append((time.perf_counter() - t0) * 1e3)
        sync()
        t0 = time.perf_counter()
        lg_t, em_t = model._forward_traced(ids, am, kv)   # captures on 1st call
        sync()
        t_ms.append((time.perf_counter() - t0) * 1e3)
        if i == 0:
            ref_l, ref_e = lg_t, em_t
        bit4 &= (torch.equal(lg_e, lg_t) and torch.equal(em_e, em_t)
                 and torch.equal(lg_t, ref_l) and torch.equal(em_t, ref_e))
    res["cells"]["batch4"] = {"eager": stats(e_ms), "traced": stats(t_ms),
                              "speedup_median": float(np.median(e_ms) / np.median(t_ms)),
                              "bit_identical_interleaved": bool(bit4),
                              "shape": list(ids.shape)}
    print(f"batch4  eager   min {np.min(e_ms):7.2f}  median {np.median(e_ms):7.2f} ms   "
          f"traced  min {np.min(t_ms):7.2f}  median {np.median(t_ms):7.2f} ms   "
          f"speedup {np.median(e_ms)/np.median(t_ms):.3f}x  "
          f"bit-identical: {'PASS' if bit4 else 'FAIL'}", flush=True)
    if not bit4:
        sys.exit("TRACE CORRECTNESS GATE FAILED (batch4)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
