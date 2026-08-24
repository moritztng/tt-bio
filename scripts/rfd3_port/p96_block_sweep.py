#!/usr/bin/env python3
"""p96 -- block-sparse atom attention, with the two ends p95 lost on actually addressed.

p95 measured the block-sparse chain at Q=32 and it came out 3.4x SLOWER than the shipped
dense one, with the core 2.1x FASTER. All of the loss is in two places, and neither is a
property of block sparsity:

    kv gather   21.987 ms   embedding plus a permute and two relayouts on 35.8 M elements
    pv matmul    8.547 ms   760 matmuls of [32,1472]@[1472,32], the tiny-M penalty

Two fixes, measured here:

1. **Head-major gather, no permutes.** `kk` arrives `[1,H,NK,DH]`, which is already
   head-major contiguous, so reshaping it to `[H*NK, DH]` and giving the index a head
   offset `h*NK` lands the gathered rows directly in `[H, nb, U, DH]`. p95 permuted the
   source into head-last, gathered, then permuted back, and both permutes were on the big
   tensor. The pad slots need no zero row: their bias is -1e4, so their softmax weight is
   exactly 0 and whatever V they read is multiplied away.
2. **Bigger blocks.** Gather cost scales with `nb * U` and the union grows much slower than
   the block count falls: p94 measured mean 488 at Q=32 and 829 at Q=128, for 4x fewer
   blocks. Same lever fixes the pv matmul, whose M is Q.

Both embedding layouts are timed: ROW_MAJOR then `to_layout`, and straight to TILE. The
memory says the sign of that choice flips with the shape, so it is measured, not picked.
"""
import json, os, pathlib, statistics, sys, time
import torch, ttnn
sys.path.insert(0, os.getcwd())
from tt_bio import softmax_generic                                       # noqa: E402
from tt_bio import rfd3_bias                                             # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.tenstorrent import get_device, attn_value_matmul             # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p96/block_sweep.json")
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TAG = sys.argv[3] if len(sys.argv) > 3 else "early"
QSWEEP = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "32,64,128,256").split(",")]
# "worst": size the key axis by the widest union over EVERY schedule point in indices.pt, which
# is what production must do because U is a compile-time constant and a per-step U recompiles.
# "tag": size it by this tag alone, which is what p95-p100 did and which under-sizes it.
UMODE = sys.argv[5] if len(sys.argv) > 5 else "tag"
IDX_PT = pathlib.Path("perf/p94/indices.pt")
H, L, NK, DH, K = 4, 6051, 6080, 32, 128
CALLS, STEPS = 9, 200
SCALE = DH ** -0.5


def tile(n):
    return ((n + 31) // 32) * 32


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


def build(idx_pad, nb, Q, U):
    """Head-offset gather index [H*nb*U] and the block-local column of every neighbour."""
    gather = torch.zeros(nb, U, dtype=torch.int64)      # pad slots read row 0; weight is 0
    pos = torch.zeros(nb * Q, K, dtype=torch.int64)
    widths = []
    for b in range(nb):
        blk = idx_pad[b * Q:(b + 1) * Q]
        u = torch.unique(blk)
        widths.append(int(u.numel()))
        gather[b, :u.numel()] = u
        pos[b * Q:(b + 1) * Q] = torch.searchsorted(u, blk)
    heads = (torch.arange(H, dtype=torch.int64) * NK).view(H, 1, 1)
    return (gather.unsqueeze(0) + heads).reshape(-1), pos, widths


def main():
    dev = get_device()
    torch.manual_seed(42)
    idx = torch.load(IDX_PT)[TAG].long()
    nb_rows = tile(L)
    idx_pad = torch.cat([idx, idx[-1:].expand(nb_rows - L, K)], 0)

    u_worst = {}
    if UMODE != "tag":
        allidx = torch.load(IDX_PT)
        for Q in QSWEEP:
            nb = nb_rows // Q
            if nb * Q != nb_rows:
                continue
            w = 0
            for t in sorted(allidx.keys()):
                it = allidx[t].long()
                ip = torch.cat([it, it[-1:].expand(nb_rows - it.shape[0], K)], 0)
                w = max(w, max(int(torch.unique(ip[b * Q:(b + 1) * Q]).numel())
                               for b in range(nb)))
            u_worst[Q] = w
        print("[p96] UMODE=worst, key axis sized by the widest union over %d schedule points: %s"
              % (len(allidx), {q: tile(v) for q, v in u_worst.items()}), flush=True)

    qq = ttnn.from_torch(torch.randn(1, H, nb_rows, DH), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    kk = ttnn.from_torch(torch.randn(1, H, NK, DH), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    vv = ttnn.from_torch(torch.randn(1, H, NK, DH), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    pair_bias = ttnn.from_torch(torch.randn(1, H, nb_rows, K), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev)

    # --- shipped dense control, same process, same operands -------------------------
    kkt = ttnn.permute(kk, (0, 1, 3, 2))
    attn_idx_rm = M._sparse_attn_index_rm(idx_pad.unsqueeze(0), dev)

    def dense_whole():
        s = ttnn.matmul(qq, kkt)
        sf = rfd3_bias.fused_scores_bias_fp32(s, pair_bias, attn_idx_rm, SCALE)
        ttnn.deallocate(s)
        a = softmax_generic.softmax_bf16(sf, ttnn.bfloat16)
        ttnn.deallocate(sf)
        o = attn_value_matmul(a, vv, None, ttnn.bfloat16)
        ttnn.deallocate(a)
        return o

    t_dense = timeit(dense_whole, dev)
    t_dense_perm = timeit(lambda: ttnn.permute(kk, (0, 1, 3, 2)), dev)
    print("[p96] tag=%s  SHIPPED dense chain %.4f ms/call -> %.3f s/design"
          % (TAG, t_dense, t_dense * CALLS * STEPS / 1000.0), flush=True)
    print("[p96] dense K permute %.4f ms/call (production does it per call, model.py:1732)"
          % t_dense_perm, flush=True)
    print("[p96] fused_enabled=%s\n" % rfd3_bias.fused_enabled(), flush=True)

    results = []
    for Q in QSWEEP:
        nb = nb_rows // Q
        if nb * Q != nb_rows:
            print("[p96] Q=%d does not divide %d, skipped" % (Q, nb_rows), flush=True)
            continue
        u_max = max(int(torch.unique(idx_pad[b * Q:(b + 1) * Q]).numel()) for b in range(nb))
        U = tile(u_max if UMODE == "tag" else max(u_max, u_worst[Q]))
        gflat, pos, widths = build(idx_pad, nb, Q, U)
        gdev = ttnn.from_torch(gflat.reshape(1, -1).to(torch.int32),
                               layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
        pos_rm = M._sparse_attn_index_rm(pos.unsqueeze(0), dev)
        qb = ttnn.reshape(qq, (H, nb, Q, DH))

        # head-major source rows, no permute: [1,H,NK,DH] is already [H*NK, DH]
        def src(x):
            return ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (H * NK, DH))

        k_src, v_src = src(kk), src(vv)

        def gath(s, tiled):
            g = ttnn.embedding(gdev, s,
                               layout=(ttnn.TILE_LAYOUT if tiled else ttnn.ROW_MAJOR_LAYOUT),
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
            g = ttnn.reshape(g, (H, nb, U, DH))
            return g if tiled else ttnn.to_layout(g, ttnn.TILE_LAYOUT)

        try:
            t_g_rm = timeit(lambda: gath(k_src, False), dev)
        except Exception as e:
            t_g_rm, _ = float("nan"), print("   gather RM  EXC %s" % str(e)[:90], flush=True)
        try:
            t_g_ti = timeit(lambda: gath(k_src, True), dev)
        except Exception as e:
            t_g_ti, _ = float("nan"), print("   gather TILE EXC %s" % str(e)[:90], flush=True)
        tiled = not (t_g_ti != t_g_ti) and not (t_g_rm == t_g_rm and t_g_rm < t_g_ti)
        t_g = t_g_ti if tiled else t_g_rm

        def whole():
            kg = gath(k_src, tiled)
            vg = gath(v_src, tiled)
            s = ttnn.matmul(qb, ttnn.permute(kg, (0, 1, 3, 2)))
            ttnn.deallocate(kg)
            s = ttnn.reshape(s, (1, H, nb_rows, U))
            sf = rfd3_bias.fused_scores_bias_fp32(s, pair_bias, pos_rm, SCALE)
            ttnn.deallocate(s)
            a = softmax_generic.softmax_bf16(sf, ttnn.bfloat16)
            ttnn.deallocate(sf)
            a = ttnn.reshape(a, (H, nb, Q, U))
            o = ttnn.matmul(a, vg)
            ttnn.deallocate(a)
            ttnn.deallocate(vg)
            return ttnn.reshape(o, (1, H, nb_rows, DH))

        # parts, on live operands
        kg = gath(k_src, tiled)
        vg = gath(v_src, tiled)
        kgt = ttnn.permute(kg, (0, 1, 3, 2))
        s_b = ttnn.reshape(ttnn.matmul(qb, kgt), (1, H, nb_rows, U))
        sf_b = rfd3_bias.fused_scores_bias_fp32(s_b, pair_bias, pos_rm, SCALE)
        a_b = ttnn.reshape(softmax_generic.softmax_bf16(sf_b, ttnn.bfloat16), (H, nb, Q, U))
        t_qk = timeit(lambda: ttnn.matmul(qb, kgt), dev)
        t_bias = timeit(lambda: rfd3_bias.fused_scores_bias_fp32(s_b, pair_bias, pos_rm, SCALE), dev)
        t_sm = timeit(lambda: softmax_generic.softmax_bf16(sf_b, ttnn.bfloat16), dev)
        t_pv = timeit(lambda: ttnn.matmul(a_b, vg), dev)
        t_perm = timeit(lambda: ttnn.permute(kg, (0, 1, 3, 2)), dev)
        load0 = os.getloadavg()[0]
        t_d_pre = timeit(dense_whole, dev)
        t_whole = timeit(whole, dev)
        t_d_post = timeit(dense_whole, dev)
        t_d = (t_d_pre + t_d_post) / 2.0
        load1 = os.getloadavg()[0]

        core = t_qk + t_bias + t_sm
        parts = 2 * t_g + t_perm + core + t_pv
        print("Q=%3d  nb=%3d  U=%4d (max %4d, mean %5.1f)  key axis %.2fx narrower"
              % (Q, nb, U, u_max, sum(widths) / len(widths), NK / U), flush=True)
        print("       gather  RM %7.4f  TILE %7.4f  -> using %s (%.4f, x2 = %.4f)"
              % (t_g_rm, t_g_ti, "TILE" if tiled else "RM", t_g, 2 * t_g), flush=True)
        print("       qk %7.4f  bias %7.4f  softmax %7.4f  = core %7.4f"
              % (t_qk, t_bias, t_sm, core), flush=True)
        print("       kg permute %7.4f  (%d gather rows, %.2fx the %d dense K rows)"
              % (t_perm, nb * U, nb * U / NK, NK), flush=True)
        print("       parts %7.4f  unattributed %+7.4f" % (parts, t_whole - parts), flush=True)
        print("       dense control this round %.4f / %.4f (drift %+.1f%%, load %.1f->%.1f)"
              % (t_d_pre, t_d_post, 100.0 * (t_d_post - t_d_pre) / t_d_pre, load0, load1),
              flush=True)
        print("       pv %7.4f    chain %8.4f ms/call -> %7.3f s/design   vs dense %+7.3f (%.2fx)"
              % (t_pv, t_whole, t_whole * CALLS * STEPS / 1000.0,
                 (t_d - t_whole) * CALLS * STEPS / 1000.0,
                 t_d / t_whole if t_whole else float("nan")), flush=True)
        results.append(dict(Q=Q, n_blocks=nb, u_width=U, u_max=u_max,
                            union_mean=round(sum(widths) / len(widths), 1),
                            gather_rm_ms=round(t_g_rm, 4), gather_tile_ms=round(t_g_ti, 4),
                            gather_used=("TILE" if tiled else "RM"),
                            qk_ms=round(t_qk, 4), bias_ms=round(t_bias, 4),
                            softmax_ms=round(t_sm, 4), core_ms=round(core, 4),
                            pv_ms=round(t_pv, 4), perm_ms=round(t_perm, 4),
                            gather_rows=nb * U, parts_ms=round(parts, 4),
                            unattributed_ms=round(t_whole - parts, 4),
                            chain_ms=round(t_whole, 4),
                            dense_pre_ms=round(t_d_pre, 4), dense_post_ms=round(t_d_post, 4),
                            dense_round_ms=round(t_d, 4),
                            load_before=round(load0, 2), load_after=round(load1, 2),
                            s_per_design=round(t_whole * CALLS * STEPS / 1000.0, 3),
                            prize_vs_dense_s=round((t_d - t_whole) * CALLS * STEPS / 1000.0, 3)))
        for t in (kg, vg, s_b, sf_b, a_b):
            try:
                ttnn.deallocate(t)
            except Exception:
                pass

    best = max((r for r in results), key=lambda r: r["prize_vs_dense_s"], default=None)
    if best:
        print("\nbest Q=%d: %+.3f s/design isolated against the shipped chain"
              % (best["Q"], best["prize_vs_dense_s"]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        tag=TAG, u_mode=UMODE, dense_chain_ms=round(t_dense, 4),
        dense_perm_ms=round(t_dense_perm, 4),
        dense_s_per_design=round(t_dense * CALLS * STEPS / 1000.0, 3),
        H=H, L=L, n_key=NK, head_dim=DH, k_sparse=K, reps=REPS,
        calls_per_step=CALLS, steps=STEPS, sweep=results,
        host=os.uname().nodename, card=os.environ.get("TT_VISIBLE_DEVICES"),
    ), indent=2) + "\n")
    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
