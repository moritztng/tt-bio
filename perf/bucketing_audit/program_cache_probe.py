"""What a bucket actually buys: program-cache entries, and what one entry costs to build.

The bucketing task has always been justified on "fewer kernel compilations", and the size of that
win depends on one fact nobody had measured: does the ttnn program cache key on the LOGICAL shape or
on the PADDED one? `ttnn.TILE_LAYOUT` already pads physically to 32, so if the key were the padded
shape then a bucket multiple of 32 would collapse nothing and every multiple above 32 would be
buying variant count with real compute. It is the logical shape, so the opposite holds.

Run from the repo root on a single card (qb2, blackhole p300 needs the mesh-graph descriptor or
`ttnn.open_device` aborts with "Custom fabric mesh graph descriptor path must be specified"):

    TT_MESH_GRAPH_DESC_PATH=$(python3 -c 'import ttnn,pathlib;print(pathlib.Path(ttnn.__file__).parent.parent.parent/"tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto")') \
    TT_METAL_CACHE=$(mktemp -d) TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
      python3 perf/bucketing_audit/program_cache_probe.py

`TT_METAL_CACHE` at a fresh directory is what makes the timing arm honest: the on-disk kernel cache
is shared across runs, so a warm cache hides exactly the cost the bucket removes.
"""
import time

import torch
import ttnn


def _open():
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    return dev


def keying(dev):
    """Same physical tile extent, different logical length. +1 entry means logical keying."""
    def up(n, w):
        return ttnn.from_torch(torch.randn(1, 1, n, w, dtype=torch.bfloat16),
                               layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    def run(label, n, w, op):
        before = dev.num_program_cache_entries()
        op(up(n, w), w)
        ttnn.synchronize_device(dev)
        after = dev.num_program_cache_entries()
        print("%-34s logical=[1,1,%d,%d] padded_w=%d  entries %d->%d (+%d)"
              % (label, n, w, (w + 31) // 32 * 32, before, after, after - before))

    softmax = lambda x, w: ttnn.softmax(x, dim=-1)
    matmul = lambda x, w: ttnn.matmul(x, up(w, 128))

    print("== does the program cache key on the logical shape? ==")
    run("softmax W=128 (aligned, 1st)", 128, 128, softmax)
    run("softmax W=98  (same pad 128)", 128, 98, softmax)
    run("softmax W=100 (same pad 128)", 128, 100, softmax)
    run("softmax W=128 again (hit?)", 128, 128, softmax)
    run("softmax W=160 (new pad)", 128, 160, softmax)
    run("matmul  K=128 (aligned, 1st)", 128, 128, matmul)
    run("matmul  K=98  (same pad 128)", 128, 98, matmul)
    run("matmul  K=100 (same pad 128)", 128, 100, matmul)


def cost(dev, n_widths=16):
    """What the entries the bucket removes cost to build. Needs a cold TT_METAL_CACHE."""
    def up(n, w):
        return ttnn.from_torch(torch.randn(1, 1, n, w, dtype=torch.bfloat16),
                               layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    def arm(label, widths):
        e0, t0 = dev.num_program_cache_entries(), time.perf_counter()
        for w in widths:
            x = up(128, w)
            ttnn.softmax(x, dim=-1)
            ttnn.matmul(x, up(w, 128))
            ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t0
        de = dev.num_program_cache_entries() - e0
        print("ARM %-30s %d calls  +%d programs  %7.3f s  %6.3f s/program"
              % (label, len(widths), de, dt, dt / max(de, 1)))
        return de, dt

    print("\n== what one program build costs (cold TT_METAL_CACHE) ==")
    widths = [98 + i for i in range(n_widths)]        # all physically 128
    a_e, a_t = arm("A ragged, %d logical widths" % n_widths, widths)
    b_e, b_t = arm("B bucketed, 1 logical width", [128] * n_widths)
    print("RESULT bucketing %d..%d -> 128 removes %d program builds and %.3f s of %.3f s"
          % (widths[0], widths[-1], a_e - b_e, a_t - b_t, a_t))


if __name__ == "__main__":
    d = _open()
    try:
        keying(d)
        cost(d)
    finally:
        ttnn.close_device(d)
