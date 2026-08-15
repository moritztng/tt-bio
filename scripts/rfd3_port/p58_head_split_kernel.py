"""Is there a one-kernel replacement for the DiT's head split, the last unbuilt lever?

p49 put the DiT's layout chain -- two reshapes, two permutes and a pad -- at 56.3 ms/step, and p51
showed it runs at 1.0-1.6 % of the read roof at every head geometry, so the cost is the op path, not
the shape. `_merge_heads` already uses `ttnn.experimental.nlp_concat_heads` for the inverse movement
and the code says that does it in one kernel. This asks whether the forward direction has the same
escape, and whether it accepts head_dim=24.

Arms at the production shape, warm, synced both sides:

  A  the shipped chain, 3x (reshape + permute) for q, k and v separately
  B  ttnn.experimental.nlp_create_qkv_heads on a fused [1, I, 1152], if it exists and accepts 24
  C  the same fused tensor through the shipped chain, to separate "one kernel" from "one call"

A bit-exactness check runs on whatever B produces; a layout op that reorders values differently is
not a candidate whatever it costs.

    ~/.coworker/scripts/benchlock.sh rfd3-page-gap-rootcause -- env TT_VISIBLE_DEVICES=0 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-page-gap-rootcause PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p58_head_split_kernel.py
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
from tt_bio.tenstorrent import get_device                              # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p58/head_split_kernel.json")
I, NH, HD = 685, 16, 24
NWARM, NREP = 3, 10


def timeit(fn, dev):
    for _ in range(NWARM):
        fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(NREP):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def main():
    dev = get_device()
    res = {"tokens": I, "n_head": NH, "head_dim": HD, "n_rep": NREP}

    qt, kt, vt = (torch.randn(1, I, NH * HD) for _ in range(3))
    q, k, v = (ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
               for t in (qt, kt, vt))
    fused_t = torch.cat([qt, kt, vt], dim=-1)
    fused = ttnn.from_torch(fused_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    def heads(x):
        return ttnn.permute(ttnn.reshape(x, (1, I, NH, HD)), (0, 2, 1, 3))

    def arm_a():
        return [heads(t) for t in (q, k, v)]

    ms_a = timeit(arm_a, dev)
    print("A shipped chain, 3x (reshape+permute)   %8.4f ms" % ms_a, flush=True)
    res["ms_shipped_chain"] = round(ms_a, 4)

    fn = getattr(getattr(ttnn, "experimental", None), "nlp_create_qkv_heads", None)
    res["nlp_create_qkv_heads_present"] = fn is not None
    if fn is None:
        print("B ttnn.experimental.nlp_create_qkv_heads is NOT present in this build", flush=True)
    else:
        try:
            out = fn(fused, num_heads=NH, num_kv_heads=NH, transpose_k_heads=False,
                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ref = ttnn.to_torch(heads(q)).float()
            got = ttnn.to_torch(out[0]).float()
            exact = bool(torch.equal(got, ref))
            ms_b = timeit(lambda: fn(fused, num_heads=NH, num_kv_heads=NH,
                                     transpose_k_heads=False,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG), dev)
            print("B nlp_create_qkv_heads                  %8.4f ms   bit-exact vs A: %s"
                  % (ms_b, exact), flush=True)
            res["ms_nlp_create_qkv_heads"] = round(ms_b, 4)
            res["nlp_bit_exact_vs_shipped"] = exact
            res["ratio_vs_shipped"] = round(ms_a / ms_b, 3)
        except Exception as e:
            print("B nlp_create_qkv_heads REJECTED: %s" % str(e).split("\n")[0][:200], flush=True)
            res["nlp_error"] = str(e).split("\n")[0][:400]

    def arm_c():
        return heads(fused)
    try:
        ms_c = timeit(lambda: ttnn.permute(ttnn.reshape(fused, (1, I, 3 * NH, HD)), (0, 2, 1, 3)),
                      dev)
        print("C one chain on the fused [1,I,1152]     %8.4f ms" % ms_c, flush=True)
        res["ms_fused_single_chain"] = round(ms_c, 4)
    except Exception as e:
        print("C fused chain REJECTED: %s" % str(e).split("\n")[0][:160], flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
