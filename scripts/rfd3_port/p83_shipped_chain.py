#!/usr/bin/env python3
"""p83 -- what does the atom attention ACTUALLY cost as shipped?

p77's `dense_shipped` arm (16.2657 ms/call), which every prize in p2 and p3 is subtracted from,
reconstructs the chain as

    matmul -> typecast fp32 -> multiply scale -> add dense_bias -> softmax -> matmul

But the shipped sparse path has `_PAIRBIAS_FUSED=1` by default and runs L6b instead: ONE kernel,
`rfd3_bias.fused_scores_bias_fp32`, replaces the mask template, the bias scatter, both widens and
the scaled add (tt_bio/rfd3/model.py, "8.5 ms/call of traffic becomes 1.67"). So p77 priced a
route production does not take, and it overstates the baseline that the 28.506 s/design headline
was subtracted from.

This measures the real thing: the four ops the shipped sparse atom attention issues per call, at
the page fixture's shape, individually and as a chain. The result is the honest ceiling on ALL
atom-attention work -- what deleting the entire site would buy.
"""
import json, os, pathlib, statistics, sys, time
import torch, ttnn
sys.path.insert(0, os.getcwd())
from tt_bio import softmax_generic                                       # noqa: E402
from tt_bio import rfd3_bias                                                 # noqa: E402
from tt_bio.rfd3 import model as M                                          # noqa: E402
from tt_bio.tenstorrent import get_device, attn_value_matmul                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p83/shipped_chain.json")
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
    print("[p83] fused_enabled=%s" % rfd3_bias.fused_enabled(), flush=True)

    qq = ttnn.from_torch(torch.randn(1, H, L, DH), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    kkt = ttnn.from_torch(torch.randn(1, H, DH, NK), dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=dev)
    vv = ttnn.from_torch(torch.randn(1, H, NK, DH), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    # The compact pair bias and the ROW_MAJOR neighbour index the shipped kernel consumes.
    idx = torch.stack([torch.randperm(L)[:K].sort().values for _ in range(L)]).unsqueeze(0)
    attn_idx_rm = M._sparse_attn_index_rm(idx, dev)
    pair_bias = ttnn.from_torch(torch.randn(1, H, L, K), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev)

    scores_bf = ttnn.matmul(qq, kkt)
    scores_f32 = rfd3_bias.fused_scores_bias_fp32(scores_bf, pair_bias, attn_idx_rm, SCALE)
    attn = softmax_generic.softmax_bf16(scores_f32, ttnn.bfloat16)

    parts = [
        ("1 qk matmul", lambda: ttnn.matmul(qq, kkt)),
        ("2 fused_scores_bias_fp32 (L6b)",
         lambda: rfd3_bias.fused_scores_bias_fp32(scores_bf, pair_bias, attn_idx_rm, SCALE)),
        ("3 softmax fp32->bf16", lambda: softmax_generic.softmax_bf16(scores_f32, ttnn.bfloat16)),
        ("4 pv matmul", lambda: attn_value_matmul(attn, vv, None, ttnn.bfloat16)),
    ]
    print("\n%-34s %11s %9s" % ("shipped sparse atom attention", "ms/call", "% chain"), flush=True)
    tot, measured = 0.0, []
    for name, fn in parts:
        try:
            t = timeit(fn, dev)
            tot += t
            measured.append((name, t))
        except Exception as e:
            print("%-34s EXC %s" % (name, str(e)[:80]), flush=True)
            rows.append(dict(part=name, exc=str(e)[:300]))
    for name, t in measured:
        print("%-34s %11.4f %8.1f%%" % (name, t, 100.0 * t / tot), flush=True)
        rows.append(dict(part=name, ms=round(t, 4), pct=round(100.0 * t / tot, 2)))
    print("%-34s %11.4f" % ("SHIPPED CHAIN TOTAL", tot), flush=True)

    def whole():
        s = ttnn.matmul(qq, kkt)
        sf = rfd3_bias.fused_scores_bias_fp32(s, pair_bias, attn_idx_rm, SCALE)
        ttnn.deallocate(s)
        a = softmax_generic.softmax_bf16(sf, ttnn.bfloat16)
        ttnn.deallocate(sf)
        o = attn_value_matmul(a, vv, None, ttnn.bfloat16)
        ttnn.deallocate(a)
        return o

    t_whole = timeit(whole, dev)
    print("%-34s %11.4f  (end to end)" % ("SHIPPED CHAIN MEASURED", t_whole), flush=True)
    rows.append(dict(part="shipped_chain_whole", ms=round(t_whole, 4)))

    site = t_whole * CALLS * STEPS / 1000.0
    p77 = 16.2657
    print("\n  p77 'dense_shipped' baseline      %8.4f ms/call -> %7.3f s/design"
          % (p77, p77 * CALLS * STEPS / 1000.0), flush=True)
    print("  ACTUAL shipped chain              %8.4f ms/call -> %7.3f s/design"
          % (t_whole, site), flush=True)
    print("  p77 overstates the site by        %8.4f ms/call -> %7.3f s/design"
          % (p77 - t_whole, (p77 - t_whole) * CALLS * STEPS / 1000.0), flush=True)
    print("\n  Deleting the ENTIRE atom attention buys at most %.3f s/design." % site, flush=True)
    print("  The two matmuls alone are irreducible, so the real ceiling is lower still.",
          flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows, "H": H, "L": L, "n_key": NK, "head_dim": DH, "k_sparse": K,
        "calls_per_step": CALLS, "steps": STEPS,
        "shipped_chain_ms": round(t_whole, 4),
        "shipped_parts_sum_ms": round(tot, 4),
        "whole_site_s_per_design": round(site, 3),
        "p77_dense_shipped_ms": p77,
        "p77_overstatement_s_per_design": round((p77 - t_whole) * CALLS * STEPS / 1000.0, 3),
        "fused_enabled": bool(rfd3_bias.fused_enabled()),
        "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
    }, indent=2) + "\n")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
