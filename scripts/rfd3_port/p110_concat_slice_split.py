#!/usr/bin/env python3
"""p110 -- who owns the 25.584 ms/call the token encoder's concat row reports?

p109 (p46 re-run) puts `ttnn.concat[B,I,I,258]` top of the token encoder at 25.584 ms/call,
51.168 ms/step, 22.3 GB/s, 5.8 % of roof -- the largest single unlevered item in the model, and
with an oversync inflation of 0.95x it is not an isolated-timing artifact.

But that row brackets a REGION, not an op, and the region is two ops:

    dself = combined one-hot                    [B,I,I,160]
    wide  = concat(z[128], dself[160])          [B,I,I,288]   both pieces tile-aligned
    zcat  = slice(wide, ..., 258)               [B,I,I,258]   <- 258 is not a tile multiple

`_CONCAT_ALIGNED` was landed on a measurement of the CONCAT at 6.168 ms/call (5.08x better than
the 31.335 of the unaligned three-piece form). The slice is the op that fix introduced, and
nothing measured it on its own. `ttnn.slice` is never a view, so it is a full copy, and its
output width is 258 -- 8.06 tiles.

This times the four ops separately at the production shape, then asks whether the slice is
needed at all. It is there so `rms_norm` averages over 258 columns instead of 288. But the 30 pad
columns are exact zeros, so they add nothing to the sum of squares and only change the
denominator: rms over 288 is rms over 258 scaled by sqrt(258/288), exactly, apart from the
epsilon term. So the 288-wide route can reproduce the 258-wide answer with a rescaled norm weight
and a linear weight zero-padded by 30 rows -- and the slice disappears.

Both the cost and the numerics of that route are measured here. Nothing is claimed bit-exact:
the epsilon breaks the algebra at the 1e-6 level and the run reports the actual maxabs.
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

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p110/concat_slice.json")
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
I = int(sys.argv[3]) if len(sys.argv) > 3 else 685
C_Z, N_BINS = 128, 65
W258 = 2 * N_BINS + C_Z                 # 258, what rms_norm must average over
W288 = 288                              # 128 + 160, the tile-aligned concat width
ONEHOT = W288 - C_Z                     # 160
CALLS_PER_STEP, STEPS = 2, 200


def timeit(fn, dev, n=REPS, warm=2):
    for _ in range(warm):
        o = fn()
        if o is not None:
            ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3)
        if o is not None:
            ttnn.deallocate(o)
    return statistics.median(out)


def per_design(ms):
    return ms * CALLS_PER_STEP * STEPS / 1000.0


def main():
    dev = get_device()
    torch.manual_seed(0)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)

    z_h = torch.randn(1, I, I, C_Z) * 0.5
    # the one-hot: 130 real columns of exactly one 1.0 per row, padded to 160 with zeros
    oh_h = torch.zeros(1, I, I, ONEHOT)
    b1 = torch.randint(0, N_BINS, (1, I, I))
    b2 = torch.randint(0, N_BINS, (1, I, I))
    oh_h.scatter_(3, b1.unsqueeze(-1), 1.0)
    oh_h.scatter_(3, (N_BINS + b2).unsqueeze(-1), 1.0)
    w_n_h = torch.randn(W258) * 0.1 + 1.0
    w_l_h = torch.randn(W258, C_Z) * 0.05

    z = ttnn.from_torch(z_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    oh = ttnn.from_torch(oh_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    w_n = ttnn.from_torch(w_n_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    w_l = ttnn.from_torch(w_l_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    # 288-wide replacements: the norm weight carries the sqrt(258/288) correction and zeros in
    # the pad, the linear weight is zero-padded by 30 rows so the pad columns contribute nothing.
    # Which correction is right depends on whether ttnn.rms_norm divides by the LOGICAL last dim
    # (258) or by its tile padding (288). If the 258-wide tensor is already being averaged over
    # its padded 288, no correction is needed at all. That is a fact about the op, so it is
    # measured rather than argued: both candidates are built and the one that matches wins.
    w_l288_h = torch.cat([w_l_h, torch.zeros(W288 - W258, C_Z)])
    w_l288 = ttnn.from_torch(w_l288_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    CORRS = {"sqrt(258/288)": (W258 / W288) ** 0.5, "1.0": 1.0,
             "sqrt(288/258)": (W288 / W258) ** 0.5}
    w_n288_by = {}
    for name, c in CORRS.items():
        w_n288_by[name] = ttnn.from_torch(
            torch.cat([w_n_h, torch.zeros(W288 - W258)]) * c,
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    w_n288 = w_n288_by["sqrt(258/288)"]

    def do_concat():
        return ttnn.concat([z, oh], dim=-1)

    wide = do_concat()

    def do_slice():
        return ttnn.slice(wide, [0, 0, 0, 0], [1, I, I, W258])

    zcat = do_slice()

    t_concat = timeit(do_concat, dev)
    t_slice = timeit(do_slice, dev)
    t_norm258 = timeit(lambda: ttnn.rms_norm(zcat, weight=w_n, epsilon=1e-6,
                                             compute_kernel_config=ckc), dev)
    t_norm288 = timeit(lambda: ttnn.rms_norm(wide, weight=w_n288, epsilon=1e-6,
                                             compute_kernel_config=ckc), dev)
    n258 = ttnn.rms_norm(zcat, weight=w_n, epsilon=1e-6, compute_kernel_config=ckc)
    n288 = ttnn.rms_norm(wide, weight=w_n288, epsilon=1e-6, compute_kernel_config=ckc)
    t_lin258 = timeit(lambda: ttnn.linear(n258, w_l, compute_kernel_config=ckc,
                                          dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN), dev)
    t_lin288 = timeit(lambda: ttnn.linear(n288, w_l288, compute_kernel_config=ckc,
                                          dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN), dev)

    out258 = ttnn.to_torch(ttnn.linear(n258, w_l, compute_kernel_config=ckc,
                                       dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)).float()
    cands = {}
    for name, wn in w_n288_by.items():
        nn = ttnn.rms_norm(wide, weight=wn, epsilon=1e-6, compute_kernel_config=ckc)
        o = ttnn.to_torch(ttnn.linear(nn, w_l288, compute_kernel_config=ckc,
                                      dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)).float()
        dd = (out258 - o).abs()
        cands[name] = dict(corr=CORRS[name], bit_exact=bool(torch.equal(out258, o)),
                           maxabs=round(float(dd.max()), 8),
                           rel_max=round(float((dd / out258.abs().clamp(min=1e-6)).max()), 8),
                           rel_median=round(
                               float((dd / out258.abs().clamp(min=1e-6)).median()), 10))
        print("[p110] corr %-14s (%.6f): bit_exact=%-5s maxabs=%.3e rel_median=%.3e"
              % (name, CORRS[name], cands[name]["bit_exact"], cands[name]["maxabs"],
                 cands[name]["rel_median"]), flush=True)
    best = min(cands, key=lambda k: cands[k]["maxabs"])
    print("[p110] best correction: %s" % best, flush=True)
    d = torch.tensor(cands[best]["maxabs"])
    den = torch.tensor(1.0)
    exact = cands[best]["bit_exact"]

    shipped = t_concat + t_slice + t_norm258 + t_lin258
    proposed = t_concat + t_norm288 + t_lin288
    rows = dict(
        I=I, w258=W258, w288=W288, reps=REPS,
        concat_ms=round(t_concat, 4), slice_ms=round(t_slice, 4),
        rms_norm_258_ms=round(t_norm258, 4), rms_norm_288_ms=round(t_norm288, 4),
        linear_258_ms=round(t_lin258, 4), linear_288_ms=round(t_lin288, 4),
        shipped_chain_ms=round(shipped, 4), proposed_chain_ms=round(proposed, 4),
        shipped_s_per_design=round(per_design(shipped), 3),
        proposed_s_per_design=round(per_design(proposed), 3),
        prize_s_per_design=round(per_design(shipped - proposed), 3),
        slice_alone_s_per_design=round(per_design(t_slice), 3),
        bit_exact=exact, best_correction=best, corrections=cands,
        card=os.environ.get("TT_VISIBLE_DEVICES"), host=os.uname().nodename,
    )

    print("[p110] I=%d  concat(128+160 -> 288) %8.4f ms" % (I, t_concat), flush=True)
    print("[p110]        slice(288 -> 258)     %8.4f ms   <- %+.3f s/design on its own"
          % (t_slice, per_design(t_slice)), flush=True)
    print("[p110]        rms_norm  258 %8.4f   288 %8.4f" % (t_norm258, t_norm288), flush=True)
    print("[p110]        linear    258 %8.4f   288 %8.4f" % (t_lin258, t_lin288), flush=True)
    print("[p110] shipped chain  %8.4f ms/call -> %7.3f s/design"
          % (shipped, per_design(shipped)), flush=True)
    print("[p110] proposed chain %8.4f ms/call -> %7.3f s/design   prize %+.3f s/design"
          % (proposed, per_design(proposed), per_design(shipped - proposed)), flush=True)
    print("[p110] numerics: best=%s bit_exact=%s maxabs=%.3e"
          % (best, exact, cands[best]["maxabs"]), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
