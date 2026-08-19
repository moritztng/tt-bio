#!/usr/bin/env python3
"""p64 -- screen the pair `Transition` with its intermediates kept in L1, row-chunked on dim 1.

The p63 census puts this op at 23.70 GB of the step's 63.43 GB of DRAM traffic (37 %) against an
irreducible 1.98 GB. Eight calls per step: `transition_2.{0,1}` at H=256 and
`pairformer_stack.{0,1}.z_transition` at H=512, each twice because `_forward_with_recycle` runs
`_process_` twice. Every one of the six ops in `Transition.__call__` round-trips DRAM, and `a`, `b`
and `m` are dead the instant `fc3` consumes them.

The route already ships in this repo for four other models: `tt_bio/tenstorrent.py:3934` keeps
`x_norm`, `x_1`, `x_2` in L1 and row-chunks the pair tensor on dim 1. RFD3's `Transition` is a
separate naive reimplementation that never got it.

This is a SCREEN, per `fusion-screen-prize-only-is-half-a-screen`: it prices the cost side (arm B,
chunking with DRAM intermediates, no residency return at all) BEFORE the prize side (arm C).

  A   shipped whole-tensor, H=512      splits z_transition out of the pairformer wall
  A2  shipped whole-tensor, H=256      must reproduce p46's 14.876 ms/call to ~10 %
  B   row-chunked dim 1, DRAM interm.  the COST alone: slice + concat + op count
  C   row-chunked dim 1, L1 interm.    the route
  D   C over a range of chunk heights  picks h; the Boltz-2 table says it is not monotonic
  E   bare [1,h,IP,128] @ [128,H], L1 out vs DRAM out    the TFLOP/s the lever turns on
  F   torch.equal(C_out, A_out) at the production shape  bit-exactness, not a tolerance

    ~/.coworker/scripts/benchlock.sh rfd3-b8-irreducible-traffic -- env TT_VISIBLE_DEVICES=2 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-b8-irreducible-traffic PYTHONPATH=$PWD RFD3_TUNE_MATMUL=1 \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p64_pair_transition_l1.py
"""
import json
import os
import pathlib
import statistics
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN            # noqa: E402
from tt_bio.rfd3 import model as M                                   # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p64/pair_transition_l1.json")
I = 685          # page fixture: 9q6y chain A, 585 target + 100 binder
C_Z = 128
NWARM, NREP = 2, 6
L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG

# What the census says these eight calls cost per step, so the screen can be read against it
# without leaving the artifact.
SHIPPED_MS_STEP = 135.1
CALLS_PER_STEP = {256: 4, 512: 4}


def mk_transition(hidden, seed):
    """A real `model.Transition` on random weights of the production shape.

    `torch_to_tt`'s default transform is `x.t()`, so the weights are stored torch-Linear style
    [out, in]. c/n are unused by `__init__` (the weights carry the shapes).
    """
    g = torch.Generator().manual_seed(seed)
    sd = {
        "layer_norm_1.weight": torch.randn(C_Z, generator=g),
        "linear_1.weight": torch.randn(hidden, C_Z, generator=g) * 0.05,
        "linear_2.weight": torch.randn(hidden, C_Z, generator=g) * 0.05,
        "linear_3.weight": torch.randn(C_Z, hidden, generator=g) * 0.05,
    }
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=False)
    return M.Transition(sd, ckc, C_Z, n=hidden // C_Z, dtype=ttnn.bfloat16)


def swiglu(mod, x, mem):
    """`Transition.__call__`'s six ops, with the three intermediates placed in `mem`.

    Byte-for-byte the same call sequence as `model.py:521-539`; the only difference is
    `memory_config=` on the intermediates and `output_tensor=a` on the multiply (which is what
    the shared tuned Transition does, `ttnn.multiply_`). The output always lands in DRAM.
    """
    xn = ttnn.rms_norm(x, weight=mod.norm_w, epsilon=1e-6,
                       compute_kernel_config=mod.compute_kernel_config, memory_config=mem)
    a = ttnn.linear(xn, mod.fc1_w, activation="silu",
                    compute_kernel_config=mod.compute_kernel_config, dtype=mod.dtype,
                    core_grid=M.BATCH_INVARIANT_GRID, memory_config=mem)
    b = M._tuned_linear(xn, mod.fc2_w, ckc=mod.compute_kernel_config, dtype=mod.dtype,
                        core_grid=M.BATCH_INVARIANT_GRID)
    ttnn.deallocate(xn)
    m = ttnn.multiply(a, b, memory_config=mem, output_tensor=a)
    ttnn.deallocate(b)
    out = M._tuned_linear(m, mod.fc3_w, ckc=mod.compute_kernel_config, dtype=mod.dtype,
                          core_grid=CORE_GRID_MAIN)
    ttnn.deallocate(m)
    return out


def shipped(mod, x):
    """The shipped call, verbatim, for the baseline arms."""
    return mod(x)


def chunked(mod, x, h, mem):
    """Row-chunk on dim 1, which is NOT a tiled dimension: no sub-tile cliff, no padding.

    The 685-row tail is ragged at every h (685 = 10x64 + 45) and gets its own shape, hence its
    own program-config cache entry, exactly as `perfwar-l1-residency-298aa` did.
    """
    H = x.shape[1]
    parts = []
    for s in range(0, H, h):
        c = x[:, s:min(s + h, H)]
        parts.append(swiglu(mod, c, mem))
        ttnn.deallocate(c)
    if len(parts) == 1:
        return parts[0]
    out = ttnn.concat(parts, dim=1)
    for p in parts:
        ttnn.deallocate(p)
    return out


def timeit(fn, dev, nrep=NREP):
    for _ in range(NWARM):
        t = fn()
        ttnn.deallocate(t)
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(nrep):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        t = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(t)
    return statistics.median(ts) * 1e3, min(ts) * 1e3, max(ts) * 1e3


def row(rows, arm, hidden, h, ms, note=""):
    med, lo, hi = ms
    rows.append({"arm": arm, "hidden": hidden, "h": h, "ms_median": round(med, 4),
                 "ms_min": round(lo, 4), "ms_max": round(hi, 4),
                 "spread_pct": round(100.0 * (hi - lo) / med, 2), "note": note})
    print("%-34s H=%3d h=%-4s %9.4f ms  [%8.4f, %8.4f]  %s"
          % (arm, hidden, h if h else "-", med, lo, hi, note), flush=True)
    return rows[-1]


def main():
    dev = get_device()
    print("tune_matmul=%s  core_grid_main=%s  batch_invariant_grid=%s"
          % (M._TUNE_MATMUL, CORE_GRID_MAIN, M.BATCH_INVARIANT_GRID), flush=True)
    rows = []
    gz = torch.Generator().manual_seed(7)
    z_t = torch.randn(1, I, I, C_Z, generator=gz)
    z = ttnn.from_torch(z_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    print("z logical %s padded %s = %.1f MB"
          % (list(z.shape), list(z.padded_shape),
             z.padded_shape[1] * z.padded_shape[2] * z.padded_shape[3] * 2 / 1e6), flush=True)

    heights = {512: [16, 32, 64], 256: [32, 64, 128]}
    if os.environ.get("P64_HEIGHTS"):  # `512:64,96 256:96,160` -- extend the sweep without a rerun
        heights = {int(k): [int(v) for v in vs.split(",")]
                   for k, vs in (s.split(":") for s in os.environ["P64_HEIGHTS"].split())}
    mods = {}
    ab_out = {}

    # --- A / A2: the shipped whole-tensor baseline, both hidden widths -------------------
    for hidden, tag in ((512, "A  shipped whole-tensor"), (256, "A2 shipped whole-tensor")):
        mods[hidden] = mk_transition(hidden, seed=100 + hidden)
        r = row(rows, tag, hidden, None, timeit(lambda: shipped(mods[hidden], z), dev),
                "baseline")
        ab_out[hidden] = r["ms_median"]

    # --- B: the COST side alone. Chunked, intermediates in DRAM, no residency return. ----
    for hidden in (512, 256):
        for h in heights[hidden]:
            row(rows, "B  chunked DRAM interm", hidden, h,
                timeit(lambda: chunked(mods[hidden], z, h, DRAM), dev), "cost side only")

    # --- C / D: the route, over the chunk heights -----------------------------------------
    best = {}
    for hidden in (512, 256):
        for h in heights[hidden]:
            try:
                ms = timeit(lambda: chunked(mods[hidden], z, h, L1), dev)
            except Exception as e:  # an L1 clash is a real datum, not a crash
                print("C  chunked L1 interm         H=%3d h=%-4s DID NOT FIT: %s"
                      % (hidden, h, str(e).splitlines()[0][:140]), flush=True)
                rows.append({"arm": "C  chunked L1 interm", "hidden": hidden, "h": h,
                             "ms_median": None, "note": "L1 clash: " + str(e).splitlines()[0][:200]})
                continue
            row(rows, "C  chunked L1 interm", hidden, h, ms, "the route")
            if hidden not in best or ms[0] < best[hidden][1]:
                best[hidden] = (h, ms[0])

    # --- E: the bare matmul the whole lever turns on, L1 out vs DRAM out ------------------
    for hidden in (512, 256):
        for h in (64, I):
            xt = ttnn.from_torch(torch.randn(1, h, I, C_Z, generator=gz), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=dev)
            w = mods[hidden].fc1_w
            gflop = 2.0 * h * z.padded_shape[2] * C_Z * hidden / 1e9
            for mem, mtag in ((L1, "L1 out"), (DRAM, "DRAM out")):
                if mem is L1 and h == I:
                    continue  # 493.9 MB does not fit; the point of chunking
                ms = timeit(lambda: ttnn.linear(
                    xt, w, compute_kernel_config=mods[hidden].compute_kernel_config,
                    dtype=ttnn.bfloat16, core_grid=M.BATCH_INVARIANT_GRID, memory_config=mem), dev)
                r = row(rows, "E  bare mm %s" % mtag, hidden, h, ms,
                        "%.1f GFLOP -> %.2f TFLOP/s" % (gflop, gflop / (ms[0] / 1e3) / 1e3))
                r["gflop"] = round(gflop, 2)
                r["tflop_s"] = round(gflop / (ms[0] / 1e3) / 1e3, 2)
            ttnn.deallocate(xt)

    # --- F: bit-exactness at the production shape, against the shipped arm ----------------
    exact = {}
    for hidden in (512, 256):
        h = best.get(hidden, (heights[hidden][0], None))[0]
        ref = shipped(mods[hidden], z)
        got = chunked(mods[hidden], z, h, L1)
        maxabs = M._mm_maxabs(got, ref)
        eq = torch.equal(ttnn.to_torch(got), ttnn.to_torch(ref))
        exact[hidden] = {"h": h, "maxabs": maxabs, "torch_equal": bool(eq)}
        print("F  bit-exact H=%3d h=%-4s maxabs=%.6e  torch.equal=%s"
              % (hidden, h, maxabs, eq), flush=True)
        ttnn.deallocate(ref)
        ttnn.deallocate(got)

    # --- the screen's verdict, in ms/step -------------------------------------------------
    def pick(arm, hidden):
        cand = [r for r in rows if r["arm"].startswith(arm) and r["hidden"] == hidden
                and r.get("ms_median")]
        return min(cand, key=lambda r: r["ms_median"]) if cand else None

    verdict = {}
    for hidden in (512, 256):
        a = pick("A", hidden) or pick("A2", hidden)
        b, c = pick("B", hidden), pick("C", hidden)
        verdict[hidden] = {
            "shipped_ms_call": a["ms_median"] if a else None,
            "cost_only_ms_call": b["ms_median"] if b else None, "cost_only_h": b["h"] if b else None,
            "route_ms_call": c["ms_median"] if c else None, "route_h": c["h"] if c else None,
            "calls_per_step": CALLS_PER_STEP[hidden],
        }
    tot_ship = sum(v["shipped_ms_call"] * v["calls_per_step"] for v in verdict.values()
                   if v["shipped_ms_call"])
    tot_route = sum(v["route_ms_call"] * v["calls_per_step"] for v in verdict.values()
                    if v["route_ms_call"])
    net = tot_ship - tot_route
    print("\nSCREEN  shipped %.1f ms/step  route %.1f ms/step  net %+.1f ms/step  = %+.2f s/design"
          % (tot_ship, tot_route, -net, -net * 200 / 1e3), flush=True)
    print("GATE    NO-GO if net < 15 ms/step, or if F is not bit-exact.", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows, "verdict": verdict, "bit_exact": exact,
        "screen": {"shipped_ms_step": round(tot_ship, 2), "route_ms_step": round(tot_route, 2),
                   "net_ms_step": round(net, 2), "net_s_design": round(net * 200 / 1e3, 3)},
        "tokens": I, "c_z": C_Z, "n_warm": NWARM, "n_rep": NREP, "host": "qb2", "card": 2,
        "tune_matmul": bool(M._TUNE_MATMUL),
        "census_shipped_ms_step": SHIPPED_MS_STEP,
        "read_roof_GB_s_measured": 390.0, "write_roof_GB_s_measured": 269.6,
    }, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
