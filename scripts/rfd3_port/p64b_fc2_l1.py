#!/usr/bin/env python3
"""p64b -- the remaining headroom in the p64 route: `fc2`'s output.

p64's arm C keeps `rms_norm`, `fc1` and the multiply in L1, but `fc2` goes through
`_tuned_linear`, which had no `memory_config` argument, so `b` still round-tripped DRAM:
493.9 MB written and read back per H=512 call. `_tuned_linear` now takes `mem=`, keyed into the
config cache because a calibrated config is only certified bitwise-equal against the default it
was timed against.

Arm C is p64's measured route (fc2 -> DRAM), arm C2 adds fc2 -> L1. The extra live buffer raises
the per-chunk L1 footprint, so the chunk height is re-swept rather than assumed.

    ~/.coworker/scripts/benchlock.sh rfd3-b8-irreducible-traffic -- env TT_VISIBLE_DEVICES=2 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-b8-irreducible-traffic PYTHONPATH=$PWD RFD3_TUNE_MATMUL=1 \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p64b_fc2_l1.py
"""
import json
import os
import pathlib
import sys

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN            # noqa: E402
from tt_bio.rfd3 import model as M                                   # noqa: E402
import scripts.rfd3_port.p64_pair_transition_l1 as p64               # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p64/fc2_l1.json")
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
I = p64.I


def swiglu(mod, x, mem, fc2_mem):
    xn = ttnn.rms_norm(x, weight=mod.norm_w, epsilon=1e-6,
                       compute_kernel_config=mod.compute_kernel_config, memory_config=mem)
    a = ttnn.linear(xn, mod.fc1_w, activation="silu",
                    compute_kernel_config=mod.compute_kernel_config, dtype=mod.dtype,
                    core_grid=M.BATCH_INVARIANT_GRID, memory_config=mem)
    b = M._tuned_linear(xn, mod.fc2_w, ckc=mod.compute_kernel_config, dtype=mod.dtype,
                        core_grid=M.BATCH_INVARIANT_GRID, mem=fc2_mem)
    ttnn.deallocate(xn)
    m = ttnn.multiply(a, b, memory_config=mem, output_tensor=a)
    ttnn.deallocate(b)
    out = M._tuned_linear(m, mod.fc3_w, ckc=mod.compute_kernel_config, dtype=mod.dtype,
                          core_grid=CORE_GRID_MAIN)
    ttnn.deallocate(m)
    return out


def chunked(mod, x, h, mem, fc2_mem):
    H = x.shape[1]
    parts = []
    for s in range(0, H, h):
        c = x[:, s:min(s + h, H)]
        parts.append(swiglu(mod, c, mem, fc2_mem))
        ttnn.deallocate(c)
    if len(parts) == 1:
        return parts[0]
    out = ttnn.concat(parts, dim=1)
    for p in parts:
        ttnn.deallocate(p)
    return out


def main():
    dev = get_device()
    rows, best = [], {}
    z = ttnn.from_torch(torch.randn(1, I, I, p64.C_Z, generator=torch.Generator().manual_seed(7)),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    mods = {h: p64.mk_transition(h, seed=100 + h) for h in (512, 256)}
    for hidden in (512, 256):
        p64.row(rows, "A  shipped whole-tensor", hidden, None,
                p64.timeit(lambda: mods[hidden](z), dev), "baseline")
        hs = os.environ.get("P64B_H_%d" % hidden)
        for h in ([int(v) for v in hs.split(",")] if hs else (32, 48, 64)):
            for tag, fc2_mem in (("C  fc2->DRAM", DRAM), ("C2 fc2->L1  ", L1)):
                try:
                    ms = p64.timeit(lambda: chunked(mods[hidden], z, h, L1, fc2_mem), dev)
                except Exception as e:
                    print("%s H=%3d h=%-3d DID NOT FIT: %s"
                          % (tag, hidden, h, str(e).splitlines()[0][:120]), flush=True)
                    rows.append({"arm": tag, "hidden": hidden, "h": h, "ms_median": None,
                                 "note": "L1 clash"})
                    continue
                p64.row(rows, tag, hidden, h, ms, "")
                k = (tag[:2], hidden)
                if k not in best or ms[0] < best[k][1]:
                    best[k] = (h, ms[0])
    # bit-exactness of the best C2 arm, against the shipped whole-tensor call
    exact = {}
    for hidden in (512, 256):
        if ("C2", hidden) not in best:
            continue
        h = best[("C2", hidden)][0]
        ref, got = mods[hidden](z), chunked(mods[hidden], z, h, L1, L1)
        exact[hidden] = {"h": h, "maxabs": M._mm_maxabs(got, ref),
                         "torch_equal": bool(torch.equal(ttnn.to_torch(got), ttnn.to_torch(ref)))}
        print("F  bit-exact C2 H=%3d h=%-3d maxabs=%.6e torch.equal=%s"
              % (hidden, h, exact[hidden]["maxabs"], exact[hidden]["torch_equal"]), flush=True)
        ttnn.deallocate(ref); ttnn.deallocate(got)

    ship = {h: min(r["ms_median"] for r in rows
                   if r["arm"].startswith("A") and r["hidden"] == h and r["ms_median"]) 
            for h in (512, 256)}
    for tag in ("C ", "C2"):
        tot = sum(4 * best[(tag, h)][1] for h in (512, 256) if (tag, h) in best)
        s = sum(4 * ship[h] for h in (512, 256))
        print("%s  shipped %.1f -> %.1f ms/step  net %+.1f ms/step = %+.2f s/design  (h=%s)"
              % (tag, s, tot, -(s - tot), -(s - tot) * 200 / 1e3,
                 {h: best[(tag, h)][0] for h in (512, 256) if (tag, h) in best}), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "bit_exact": exact,
                               "best": {f"{k[0]}|{k[1]}": v for k, v in best.items()},
                               "shipped_ms_call": ship, "tokens": I, "host": "qb2", "card": 2},
                              indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
