#!/usr/bin/env python3
"""Fold-level A/B for the relaxed gate, with the K-block width measured instead of predicted.

Arm A is main. At 117 aa that is exactly the plain `ttnn.matmul` at every site, which the census
established as a fact and not an assumption: main's gate declines 12974 of 12974 calls on
protenix-v2 and 12982 of 12982 on opendde. So flipping `_BATCHED_MATMUL_ON` reproduces main's
behaviour at this size, and it is a faithful control rather than a proxy.

Arm B is the relaxed gate plus a self-calibrating width. `_batched_matmul_block_w` mispredicts on 4
of the 12 classes the relaxed gate admits (width_qb1c0.json), and a mispredicted width is not
bit-exact. Instead of predicting, arm B measures once per class on first sight: run the plain call,
try each width dividing Kt, keep the first that is `torch.equal`, cache it, fall back to the plain
call if none is. Bit-exactness becomes a checked property of the run instead of a rule fitted at two
protein lengths. The calibration call returns the plain result it already computed, so it costs one
extra matmul per candidate width, once per class per process.

This is a probe, not a proposal: nothing in `tt_bio/` is touched beyond the one-line gate already on
the branch. The open question a merge would have to answer first is what this does inside a ttnn
trace capture, where `to_torch` is not allowed.

Both arms in one process, arms rotated per round, one warm fold per arm before any timing. Parity is
the CIF sha256 and plDDT: a program config cannot change a value, so any difference means a width
was admitted that is not exact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

A3M = {"examples/prot.yaml": "prot117.a3m", "examples/prot300.yaml": "prot300.a3m"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", default="examples/prot.yaml")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    import ttnn
    import tt_bio.tenstorrent as T

    dev = T.get_device()
    GRID = tuple(int(v) for v in T.COMPUTE_GRID_MAIN)
    L1 = int(ttnn.get_max_worker_l1_unreserved_size())
    print(f"grid {GRID} cores {GRID[0]*GRID[1]} L1 {L1} "
          f"gate floor {T._BATCHED_MATMUL_SATURATION_BLOCKS}", flush=True)

    def search_w(batch, mt, kt, nt, eb, w):
        """The shipped search, with the width pinned instead of predicted."""
        orig = T._batched_matmul_block_w
        T._batched_matmul_block_w = lambda *_a: w
        try:
            T._batched_matmul_search.cache_clear()
            return T._batched_matmul_search(batch, mt, kt, nt, eb, GRID, L1)
        finally:
            T._batched_matmul_block_w = orig
            T._batched_matmul_search.cache_clear()

    wcache: dict = {}
    cfgcache: dict = {}
    decided: dict = {}
    stats = {"calls": 0, "declined_guard": 0, "calibrations": 0, "calib_s": 0.0,
             "no_exact_width": 0}
    orig_bmm = T.batched_matmul

    arm = {"name": "off"}

    def bmm(x, y, compute_kernel_config=None, dtype=None):
        kw = {} if dtype is None else {"dtype": dtype}
        plain = lambda: ttnn.matmul(x, y, compute_kernel_config=compute_kernel_config, **kw)
        stats["calls"] += 1
        if arm["name"] == "off":
            # helper absent entirely: no guard chain, no config. Not main -- main evaluates the
            # guard on every call and only then declines. This arm exists to price the guard.
            return plain()
        sx, sy = tuple(int(d) for d in x.shape), tuple(int(d) for d in y.shape)
        if arm["name"] == "cached":
            # The whole decision cached per class: two shape reads, one dict lookup, no
            # memory_config()/is_sharded()/L1 query per call. This is the recoverable ceiling, not a
            # shippable design -- a real key has to cover memory config too, since the same shape can
            # arrive sharded.
            k2 = (sx, sy, x.dtype, dtype)
            if k2 in decided:
                cfg = decided[k2]
                return plain() if cfg is None else ttnn.matmul(
                    x, y, program_config=cfg, compute_kernel_config=compute_kernel_config, **kw)
        if not (len(sx) >= 4 and len(sx) == len(sy)
                and sx[:-2] == sy[:-2] and x.dtype == y.dtype
                and T._dram_interleaved(x) and T._dram_interleaved(y)):
            stats["declined_guard"] += 1
            return plain()
        batch = 1
        for d in sx[:-2]:
            batch *= d
        mt, kt, nt = -(-sx[-2] // 32), -(-sx[-1] // 32), -(-sy[-1] // 32)
        eb = 4 if x.dtype == ttnn.float32 else 2
        key = (sx, sy, str(x.dtype), str(dtype))
        if arm["name"] == "cached":
            if key not in cfgcache:
                arm["name"] = "relaxed"          # calibrate through the relaxed path once
                out = bmm(x, y, compute_kernel_config=compute_kernel_config, dtype=dtype)
                arm["name"] = "cached"
                decided[(sx, sy, x.dtype, dtype)] = cfgcache[key]
                return out
            decided[(sx, sy, x.dtype, dtype)] = cfgcache[key]
            cfg = cfgcache[key]
            return plain() if cfg is None else ttnn.matmul(
                x, y, program_config=cfg, compute_kernel_config=compute_kernel_config, **kw)
        if arm["name"] == "main":
            # faithful main at this size: the whole guard chain runs, the L1 query runs, the gate
            # declines. Same host cost, same plain matmul.
            T._batched_matmul_config(batch, mt, kt, nt, eb)
            return plain()
        if key not in wcache:
            t0 = time.perf_counter()
            ref = plain()
            ref_t = ttnn.to_torch(ref)
            best = None
            for w in [w for w in (1, 2, 4, 8, 16) if kt % w == 0]:
                cfg = search_w(batch, mt, kt, nt, eb, w)
                if cfg is None:
                    continue
                try:
                    got = ttnn.matmul(x, y, program_config=cfg,
                                      compute_kernel_config=compute_kernel_config, **kw)
                except RuntimeError:
                    continue          # the CB model underestimates at some shapes; skip that width
                ok = bool(torch.equal(ttnn.to_torch(got), ref_t))
                ttnn.deallocate(got)
                if ok:
                    best = w
                    break
            wcache[key] = best
            # Steady state must not re-run the search: main pays one lru_cache hit plus the L1
            # query per call, and a probe that pays a full search instead measures the probe.
            cfgcache[key] = None if best is None else search_w(batch, mt, kt, nt, eb, best)
            stats["calibrations"] += 1
            stats["calib_s"] += time.perf_counter() - t0
            if best is None:
                stats["no_exact_width"] += 1
            return ref
        cfg = cfgcache[key]
        if cfg is None:
            return plain()
        return ttnn.matmul(x, y, program_config=cfg,
                           compute_kernel_config=compute_kernel_config, **kw)

    # Host-side cost per arm, measured inside the helper. The fold wall clock cannot resolve a
    # sub-500 ms effect on a box carrying other legs, but the time spent inside batched_matmul is a
    # host measurement with microsecond resolution and it is what the guard chain actually costs.
    # It includes the ttnn.matmul enqueue, which is identical across arms, so arm differences are
    # the helper's own overhead. Device time is NOT in here: matmul enqueues asynchronously.
    hstats: dict = {}
    _inner = bmm

    def bmm(x, y, compute_kernel_config=None, dtype=None):          # noqa: F811
        t0 = time.perf_counter()
        try:
            return _inner(x, y, compute_kernel_config=compute_kernel_config, dtype=dtype)
        finally:
            h = hstats.setdefault(arm["name"], [0, 0.0])
            h[0] += 1
            h[1] += time.perf_counter() - t0

    T.batched_matmul = bmm

    import tt_baseline as B
    B.SAMPLING_STEPS = a.sampling_steps
    msa_dir = Path(tempfile.mkdtemp(prefix="foldab-msa-"))
    one_fold, meta, _state = B.build_fold(
        a.model, msa_dir, ROOT / a.target, Path(B.FIXTURES) / A3M[a.target], samples=a.samples)
    late = [n for n, m in list(sys.modules.items())
            if getattr(m, "batched_matmul", None) is orig_bmm]
    for n in late:
        setattr(sys.modules[n], "batched_matmul", bmm)
    print("rebound late in", late, flush=True)

    struct_dir = Path(meta["struct_dir"])

    def run(name):
        arm["name"] = name
        hstats.pop(name, None)
        t, m = one_fold()
        n, hs = hstats.get(name, [0, 0.0])
        cifs = sorted(struct_dir.glob("*.cif"))
        sha = hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16] if cifs else None
        return dict(arm=name, s=round(t, 3), sha=sha, plddt=m.get("plddt"),
                    bmm_calls=n, host_in_bmm_ms=round(hs * 1e3, 1),
                    host_us_per_call=round(hs / n * 1e6, 2) if n else None)

    ARMS = ["off", "main", "relaxed", "cached"]
    warm = [run(a_) for a_ in ARMS]
    print("warm:", warm, flush=True)
    print(f"calibrations {stats['calibrations']} in {stats['calib_s']:.3f} s, "
          f"no exact width for {stats['no_exact_width']}", flush=True)
    print("width cache:", flush=True)
    for (sx, sy, dt, _o), w in wcache.items():
        print(f"    {list(sx)} @ {list(sy)} {dt.split('.')[-1]:8s} -> {w}", flush=True)

    runs = []
    order = ARMS
    for r in range(a.rounds):
        for a_ in order[r % len(order):] + order[:r % len(order)]:
            rec = run(a_)
            rec["round"] = r
            runs.append(rec)
            nm = rec["arm"]
            print(f"  round {r} {nm:8s} {rec['s']:8.3f} s "
                  f"host_in_bmm={rec['host_in_bmm_ms']:7.1f} ms "
                  f"({rec['host_us_per_call']} us x {rec['bmm_calls']}) plddt={rec['plddt']} "
                  f"sha={rec['sha']}", flush=True)

    per = {a_: [r["s"] for r in runs if r["arm"] == a_] for a_ in order}
    med = {a_: round(st.median(v), 3) for a_, v in per.items()}
    hper = {a_: [r["host_in_bmm_ms"] for r in runs if r["arm"] == a_] for a_ in order}
    hmed = {a_: round(st.median(v), 1) for a_, v in hper.items() if v}
    uper = {a_: [r["host_us_per_call"] for r in runs if r["arm"] == a_] for a_ in order}
    umed = {a_: round(st.median([x for x in v if x is not None]), 2)
            for a_, v in uper.items() if any(x is not None for x in v)}
    shas = {r["sha"] for r in runs} | {r["sha"] for r in warm}
    plddts = {r["plddt"] for r in runs}
    out = dict(model=a.model, target=a.target, samples=a.samples,
               sampling_steps=a.sampling_steps, recycling_steps=B.RECYCLING_STEPS,
               grid=list(GRID), gate_floor=T._BATCHED_MATMUL_SATURATION_BLOCKS,
               warm=warm, runs=runs, median_s=med,
               saved_ms=round((med["main"] - med["relaxed"]) * 1e3, 1),
               speedup=round(med["main"] / med["relaxed"], 4),
               guard_cost_ms=round((med["main"] - med["off"]) * 1e3, 1),
               relaxed_vs_off_ms=round((med["off"] - med["relaxed"]) * 1e3, 1),
               cached_vs_main_ms=round((med["main"] - med["cached"]) * 1e3, 1),
               cached_vs_off_ms=round((med["off"] - med["cached"]) * 1e3, 1),
               host_in_bmm_ms_median=hmed, host_us_per_call_median=umed,
               bit_exact_fold=len(shas) == 1, shas=sorted(shas), plddts=sorted(plddts),
               calibrations=stats["calibrations"], calib_s=round(stats["calib_s"], 3),
               no_exact_width=stats["no_exact_width"],
               width_cache={f"{list(k[0])}@{list(k[1])}|{k[2]}": v for k, v in wcache.items()},
               card_type=meta.get("card_type"), aiclk_mhz=meta.get("aiclk_mhz"))
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("runs", "warm")}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
