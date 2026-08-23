#!/usr/bin/env python3
"""p82 -- re-price the atom attention honestly, per op, and with per-row distinct keys.

Why this exists. p2's 28.506 s/design headline (perf/p77/exactness_bound.json, arm
`sparse_K128`) built its gathered arm as

    kg = [1, H, DH, K]          ttnn.matmul(q[1,H,L,DH], kg) -> [1,H,L,K]

which gives every one of the 6051 query rows the SAME 128 keys. That is an ordinary small dense
matmul, not gathered attention: the real thing needs 128 DISTINCT keys per row. So 0.4291 ms/call
prices work the model would not be doing, and the 16.2657 - 0.4291 subtraction that produced the
only prize large enough to reach the bar is not a bound on gathered attention.

This measures two things instead:

  A  where the shipped dense chain's 16.2657 ms/call actually goes, op by op. That says what a
     fused kernel would be deleting, rather than assuming it deletes all of it.
  B  an HONEST per-row-gathered arm: K and V pre-gathered on the host into
     [1, H*L, K, DH], so every row has its own 128 keys, timed as a batched matmul. The host
     gather is outside the timing, so this is a bound on "gathered operands already in DRAM" --
     pessimistic against a fused kernel that keeps K/V L1-resident, optimistic against anything
     that has to build the gathered operands with a device op.
"""
import json, os, pathlib, statistics, sys, time
import torch, ttnn
sys.path.insert(0, os.getcwd())
from tt_bio import softmax_generic                                       # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p82/honest_bound.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
H, L, NK, DH, K = 4, 6051, 6080, 32, 128
CALLS, STEPS = 9, 200
SCALE = DH ** -0.5


def timeit(fn, dev, n=N, warm=2):
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


def main():
    dev = get_device()
    torch.manual_seed(42)
    rows = []

    q = ttnn.from_torch(torch.randn(1, H, L, DH), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev)
    kt = ttnn.from_torch(torch.randn(1, H, DH, NK), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    v = ttnn.from_torch(torch.randn(1, H, NK, DH), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev)
    bias32 = ttnn.from_torch(torch.randn(1, H, L, NK), dtype=ttnn.float32,
                             layout=ttnn.TILE_LAYOUT, device=dev)

    # ---- A: the shipped dense chain, whole and then op by op ---------------------------------
    def dense_whole():
        sc = ttnn.matmul(q, kt)
        sc32 = ttnn.typecast(sc, ttnn.float32); ttnn.deallocate(sc)
        sc32 = ttnn.multiply(sc32, SCALE)
        sc32 = ttnn.add(sc32, bias32)
        a = softmax_generic.softmax_bf16(sc32, ttnn.bfloat16); ttnn.deallocate(sc32)
        o = ttnn.matmul(a, v); ttnn.deallocate(a)
        return o

    t_dense = timeit(dense_whole, dev)
    print("[p82] dense chain whole            %9.4f ms/call" % t_dense, flush=True)
    rows.append(dict(arm="dense_whole", ms=round(t_dense, 4)))

    sc_bf = ttnn.matmul(q, kt)
    sc_f32 = ttnn.typecast(sc_bf, ttnn.float32)
    sc_biased = ttnn.add(ttnn.multiply(sc_f32, SCALE), bias32)
    attn = softmax_generic.softmax_bf16(sc_biased, ttnn.bfloat16)

    parts = [
        ("qk matmul  [L,32]@[32,6080]", lambda: ttnn.matmul(q, kt)),
        ("typecast   bf16->fp32 588MB", lambda: ttnn.typecast(sc_bf, ttnn.float32)),
        ("multiply   scale fp32", lambda: ttnn.multiply(sc_f32, SCALE)),
        ("add        + bias fp32", lambda: ttnn.add(sc_f32, bias32)),
        ("softmax    fp32->bf16", lambda: softmax_generic.softmax_bf16(sc_biased, ttnn.bfloat16)),
        ("pv matmul  [L,6080]@[6080,32]", lambda: ttnn.matmul(attn, v)),
    ]
    print("\n%-32s %12s %8s" % ("dense chain, op by op", "ms/call", "% chain"), flush=True)
    tot = 0.0
    for name, fn in parts:
        try:
            t = timeit(fn, dev)
            tot += t
            print("%-32s %12.4f %7.1f%%" % (name, t, 100.0 * t / t_dense), flush=True)
            rows.append(dict(arm="dense_part", part=name, ms=round(t, 4),
                             pct_of_chain=round(100.0 * t / t_dense, 2)))
        except Exception as e:
            print("%-32s  EXC %s" % (name, str(e)[:70]), flush=True)
    print("%-32s %12.4f %7.1f%%  (sum of parts)" % ("", tot, 100.0 * tot / t_dense), flush=True)
    rows.append(dict(arm="dense_parts_sum", ms=round(tot, 4)))
    for t in (sc_bf, sc_f32, sc_biased, attn):
        ttnn.deallocate(t)

    # ---- B: p2's arm, reproduced, and then the honest per-row-gathered arm --------------------
    kg = ttnn.from_torch(torch.randn(1, H, DH, K), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    vg = ttnn.from_torch(torch.randn(1, H, K, DH), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    biasg = ttnn.from_torch(torch.randn(1, H, L, K), dtype=ttnn.float32,
                            layout=ttnn.TILE_LAYOUT, device=dev)

    def p2_sparse():
        sc = ttnn.matmul(q, kg)
        sc32 = ttnn.typecast(sc, ttnn.float32); ttnn.deallocate(sc)
        sc32 = ttnn.add(ttnn.multiply(sc32, SCALE), biasg)
        a = softmax_generic.softmax_bf16(sc32, ttnn.bfloat16); ttnn.deallocate(sc32)
        o = ttnn.matmul(a, vg); ttnn.deallocate(a)
        return o

    t_p2 = timeit(p2_sparse, dev)
    print("\n[p82] p2 arm, ONE shared key block %9.4f ms/call  (reproduces 0.4291)" % t_p2,
          flush=True)
    rows.append(dict(arm="p2_shared_keys", ms=round(t_p2, 4),
                     note="every row attends the same 128 keys -- not gathered attention"))

    # Honest arm: per-row distinct keys, pre-gathered on host, batched over H*L rows.
    # q_b [1, H*L, 1, DH] @ kg_b [1, H*L, DH, K] -> [1, H*L, 1, K]
    try:
        qb = ttnn.from_torch(torch.randn(1, H * L, 1, DH), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=dev)
        kgb = ttnn.from_torch(torch.randn(1, H * L, DH, K), dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=dev)
        vgb = ttnn.from_torch(torch.randn(1, H * L, K, DH), dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=dev)
        bgb = ttnn.from_torch(torch.randn(1, H * L, 1, K), dtype=ttnn.float32,
                              layout=ttnn.TILE_LAYOUT, device=dev)

        def honest():
            sc = ttnn.matmul(qb, kgb)
            sc32 = ttnn.typecast(sc, ttnn.float32); ttnn.deallocate(sc)
            sc32 = ttnn.add(ttnn.multiply(sc32, SCALE), bgb)
            a = softmax_generic.softmax_bf16(sc32, ttnn.bfloat16); ttnn.deallocate(sc32)
            o = ttnn.matmul(a, vgb); ttnn.deallocate(a)
            return o

        t_hon = timeit(honest, dev, n=3)
        print("[p82] honest per-row gathered      %9.4f ms/call  (operands already in DRAM)"
              % t_hon, flush=True)
        rows.append(dict(arm="honest_per_row_gathered", ms=round(t_hon, 4),
                         note="K,V pre-gathered on host into [1,H*L,K,DH]; host gather NOT timed"))
    except Exception as e:
        t_hon = None
        print("[p82] honest per-row gathered      EXC %s" % str(e)[:110], flush=True)
        rows.append(dict(arm="honest_per_row_gathered", exc=str(e)[:400]))

    def prize(t):
        return None if t is None else round((t_dense - t) * CALLS * STEPS / 1000.0, 3)

    print("\n%-38s %12s %14s" % ("arm", "ms/call", "s/design prize"), flush=True)
    for nm, t in (("dense shipped", t_dense), ("p2 shared-key arm (28.506 came from here)", t_p2),
                  ("honest per-row gathered", t_hon)):
        print("%-38s %12s %14s"
              % (nm, "n/a" if t is None else "%.4f" % t, prize(t)), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows, "H": H, "L": L, "n_key": NK, "head_dim": DH, "k_sparse": K,
        "calls_per_step": CALLS, "steps": STEPS,
        "dense_ms": round(t_dense, 4), "p2_shared_key_ms": round(t_p2, 4),
        "honest_gathered_ms": None if t_hon is None else round(t_hon, 4),
        "p2_prize_s_per_design": prize(t_p2), "honest_prize_s_per_design": prize(t_hon),
        "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
    }, indent=2) + "\n")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
