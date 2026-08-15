"""Can the two 65-wide one-hots become one tile-aligned piece, so the concat takes its fast path?

p52 established that `ttnn.concat` is 15-20x slow if ANY input piece is narrower than a tile, and
that neither the output width nor the piece offset matters. The encoder concatenates z (128) with
two 65-wide one-hots, so both one-hots are on the slow side of that cliff.

There is a bit-exact way out that p52 did not consider. Instead of two one-hots, gather ONE combined
one-hot from a table indexed by (d_bin * 65 + self_bin), whose rows are the two one-hots side by
side in the first 130 columns and zero out to 160 -- 5 tiles. Then:

    zcat320 = concat([z(128), dself(160)])   both pieces tile multiples -> the fast path
    zcat    = slice(zcat320, 0:258)          the padding is contiguous at the END, so this is exact

Every value is 0.0 or 1.0 or a copy, so the result is bitwise what ships today, with no change to the
rms_norm that consumes it. The whole route turns on one unmeasured cost: `ttnn.embedding` with a
4225 x 160 table instead of a 65 x 65 one. This measures that, and the slice, before anything is
built.

Kill gate, written first: the route replaces two 65-wide embeddings plus the 258 concat (about 32.0
ms/call) with one wide embedding plus a 320 concat plus a slice. If that sum is not under 20 ms/call
the route is NO-GO and the compensated non-bit-exact form is the only way to the 50.7 ms/step.

    ~/.coworker/scripts/benchlock.sh rfd3-page-gap-rootcause -- env TT_VISIBLE_DEVICES=0 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-page-gap-rootcause PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p53_combined_onehot.py
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

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p53/combined_onehot.json")
I, NBINS = 685, 65
NWARM, NREP = 2, 6


def timeit(fn, dev):
    for _ in range(NWARM):
        t = fn()
        ttnn.deallocate(t)
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(NREP):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        t = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(t)
    return statistics.median(ts) * 1e3


def onehot(idx, table, width, dev):
    oh = ttnn.embedding(idx, table, layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    oh = ttnn.reshape(oh, (1, I, I, width))
    return ttnn.to_layout(oh, ttnn.TILE_LAYOUT)


def main():
    dev = get_device()
    rows = []
    bins_d = torch.randint(0, NBINS, (1, I, I), dtype=torch.int32)
    bins_s = torch.randint(0, NBINS, (1, I, I), dtype=torch.int32)
    # the model uploads the index flat (_tt_idx), and ttnn.embedding wants it that way
    idx_d = ttnn.from_torch(bins_d.reshape(1, -1), dtype=ttnn.uint32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    idx_c = ttnn.from_torch((bins_d * NBINS + bins_s).reshape(1, -1), dtype=ttnn.uint32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

    eye65 = ttnn.from_torch(torch.eye(NBINS), dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    comb = torch.zeros(NBINS * NBINS, 160)
    ar = torch.arange(NBINS)
    comb[ar.repeat_interleave(NBINS) * NBINS + ar.repeat(NBINS), ar.repeat_interleave(NBINS)] = 1.0
    comb[ar.repeat_interleave(NBINS) * NBINS + ar.repeat(NBINS), NBINS + ar.repeat(NBINS)] = 1.0
    tab160 = ttnn.from_torch(comb, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

    ms_e65 = timeit(lambda: onehot(idx_d, eye65, NBINS, dev), dev)
    ms_e160 = timeit(lambda: onehot(idx_c, tab160, 160, dev), dev)
    print("embedding 65-wide  (x2 today) %8.4f ms each" % ms_e65, flush=True)
    print("embedding 160-wide (x1 new)   %8.4f ms" % ms_e160, flush=True)

    z = ttnn.from_torch(torch.randn(1, I, I, 128), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev)
    dself = ttnn.from_torch(torch.randn(1, I, I, 160), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev)
    ms_cat320 = timeit(lambda: ttnn.concat([z, dself], dim=-1), dev)
    cat320 = ttnn.concat([z, dself], dim=-1)
    ms_slice = timeit(lambda: ttnn.slice(cat320, [0, 0, 0, 0], [1, I, I, 258]), dev)
    print("concat 128+160 -> 320         %8.4f ms" % ms_cat320, flush=True)
    print("slice  320 -> 258             %8.4f ms" % ms_slice, flush=True)

    shipped = 2 * ms_e65 + 25.35
    proposed = ms_e160 + ms_cat320 + ms_slice
    print("\nshipped  2 embeddings + concat258 = %.3f ms/call" % shipped)
    print("proposed 1 embedding + concat320 + slice = %.3f ms/call" % proposed)
    print("ratio %.2fx, %.1f ms/step at 2 calls/step  [gate: proposed < 20 ms]"
          % (shipped / proposed, 2 * (shipped - proposed)))

    rows = {"ms_embedding_65": round(ms_e65, 4), "ms_embedding_160": round(ms_e160, 4),
            "ms_concat_128_160": round(ms_cat320, 4), "ms_slice_320_258": round(ms_slice, 4),
            "ms_shipped_per_call": round(shipped, 3), "ms_proposed_per_call": round(proposed, 3),
            "ms_saved_per_step": round(2 * (shipped - proposed), 2),
            "concat258_ms_from_p52": 25.35, "tokens": I, "n_rep": NREP,
            "host": "qb2", "card": 0, "ttnn": "0.68.0"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
