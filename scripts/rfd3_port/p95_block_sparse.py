#!/usr/bin/env python3
"""p95 -- the BLOCK-sparse atom attention chain, at the production shape, against the shipped one.

p94 measured that the 128-neighbour index is block-sparse: the union of the key sets of 32
consecutive query rows is 307-524 keys on average and never exceeded 1462 at any schedule
point. So the site does not need a per-row gather (p82: 7.2x slower than dense) and does not
need `ttnn.gather` (silently wrong above 1920). It needs a per-BLOCK gather and a batched
dense matmul over [H, n_blocks, Q, U].

The chain this builds, all stock ops plus the shipped L6b bias kernel:

    kv gather   ttnn.embedding, per block, over the union rows        -> [H, nb, U, DH]
    qk          matmul [H,nb,Q,DH] @ [H,nb,DH,U]                      -> [H, nb, Q, U]
    bias        rfd3_bias.fused_scores_bias_fp32, key axis U not 6080 -> fp32
    softmax     over U                                                -> bf16
    pv          matmul [H,nb,Q,U] @ [H,nb,U,DH]                       -> [H, nb, Q, DH]

The L6b kernel needs no change: it scatters K bias values into a key axis of width
`scores.shape[3]` at positions given by `idx_rm`, and the block form just gives it a smaller
axis and block-local positions.

Both arms run in ONE process on the same operands, so the comparison is a control, not a
quote from another day's artifact. The dense arm is re-measured here rather than taken from
p83. Correctness is checked by value against the dense arm, which is what decides whether
this needs the accuracy envelope at all.
"""
import json, os, pathlib, statistics, sys, time
import torch, ttnn
sys.path.insert(0, os.getcwd())
from tt_bio import softmax_generic                                       # noqa: E402
from tt_bio import rfd3_bias                                             # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.tenstorrent import get_device, attn_value_matmul             # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p95/block_sparse.json")
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
IDX_PT = pathlib.Path("perf/p94/indices.pt")
TAG = sys.argv[3] if len(sys.argv) > 3 else "early"       # early is the widest union p94 saw
QSWEEP = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "32,64,128,256").split(",")]
H, L, NK, DH, K = 4, 6051, 6080, 32, 128
Q = 32
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


def build_blocks(idx, nb, u_width):
    """Per-block key union and each row's position inside it.

    Returns gather_rows [nb, u_width] int64 (pad slots point at row NK, a zero row) and
    pos [L_pad, K] int64, the block-local column of every neighbour.
    """
    l_pad = nb * Q
    gather = torch.full((nb, u_width), NK, dtype=torch.int64)
    pos = torch.zeros(l_pad, K, dtype=torch.int64)
    widths = []
    for b in range(nb):
        r0, r1 = b * Q, min((b + 1) * Q, idx.shape[0])
        blk = idx[r0:r1]
        u = torch.unique(blk)                       # sorted ascending
        widths.append(int(u.numel()))
        gather[b, :u.numel()] = u
        # searchsorted maps each neighbour id to its slot in the block union
        pos[r0:r1] = torch.searchsorted(u, blk)
        if r1 < (b + 1) * Q:                        # tail rows of the padded block
            pos[r1:(b + 1) * Q] = 0
    return gather, pos, widths


def gather_kv(x_bhnd, gather_flat_dev, nb, u_width, dev):
    """[1,H,NK,DH] -> [H, nb, U, DH], gathering whole rows per block via ttnn.embedding.

    ttnn.embedding is the row-gather this codebase already runs at production scale
    (_pack_atoms_dev_core). ttnn.gather is not usable here and is not needed: the gather is
    over the ROW axis, which embedding does natively.
    """
    x = ttnn.permute(x_bhnd, (0, 2, 1, 3))                       # [1, NK, H, DH]
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (NK, H * DH))
    x = ttnn.pad(x, [[0, 1], [0, 0]], 0.0)                       # row NK == zeros, the pad slot
    return gather_kv_from_rm(x, gather_flat_dev, nb, u_width)


def gather_kv_from_rm(x_rm, gather_flat_dev, nb, u_width):
    """The half that repeats per call: the source rows are a per-STEP constant, so the
    permute/relayout/pad above is hoistable out of the 9 calls a step makes and only the
    embedding and the reshape back are really per call."""
    g = ttnn.embedding(gather_flat_dev, x_rm, layout=ttnn.ROW_MAJOR_LAYOUT,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    g = ttnn.reshape(g, (nb, u_width, H, DH))
    g = ttnn.to_layout(g, ttnn.TILE_LAYOUT)
    return ttnn.permute(g, (2, 0, 1, 3))                         # [H, nb, U, DH]


def to_rm_rows(x_bhnd):
    x = ttnn.permute(x_bhnd, (0, 2, 1, 3))
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (NK, H * DH))
    return ttnn.pad(x, [[0, 1], [0, 0]], 0.0)


def main():
    dev = get_device()
    torch.manual_seed(42)
    if not IDX_PT.exists():
        print("[p95] %s missing -- run p94 first" % IDX_PT, flush=True)
        return
    idx = torch.load(IDX_PT)[TAG].long()                         # [L, K], sorted ascending
    assert idx.shape == (L, K), idx.shape
    nb = (NK + Q - 1) // Q
    l_pad = nb * Q
    idx_pad = torch.cat([idx, idx[-1:].expand(l_pad - L, K)], 0)  # pad rows reuse the last index

    u_max = max(int(torch.unique(idx_pad[b * Q:(b + 1) * Q]).numel()) for b in range(nb))
    U = tile(u_max)
    gather, pos, widths = build_blocks(idx_pad, nb, U)
    print("[p95] tag=%s  nb=%d  U_max=%d -> U=%d  (union mean %.1f median %.1f)"
          % (TAG, nb, u_max, U, sum(widths) / len(widths),
             statistics.median(widths)), flush=True)
    print("[p95] fused_enabled=%s  dense key axis %d -> block key axis %d (%.2fx narrower)"
          % (rfd3_bias.fused_enabled(), NK, U, NK / U), flush=True)

    qq = ttnn.from_torch(torch.randn(1, H, l_pad, DH), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    kk = ttnn.from_torch(torch.randn(1, H, NK, DH), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    vv = ttnn.from_torch(torch.randn(1, H, NK, DH), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    pair_bias = ttnn.from_torch(torch.randn(1, H, l_pad, K), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev)

    # --- dense (shipped) arm ---------------------------------------------------------
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

    # --- block-sparse arm ------------------------------------------------------------
    gather_dev = ttnn.from_torch(gather.reshape(1, -1).to(torch.int32),
                                 layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
    pos_rm = M._sparse_attn_index_rm(pos.unsqueeze(0), dev)
    qb = ttnn.reshape(qq, (H, nb, Q, DH))

    def block_whole():
        kg = gather_kv(kk, gather_dev, nb, U, dev)               # [H, nb, U, DH]
        vg = gather_kv(vv, gather_dev, nb, U, dev)
        s = ttnn.matmul(qb, ttnn.permute(kg, (0, 1, 3, 2)))      # [H, nb, Q, U]
        ttnn.deallocate(kg)
        s = ttnn.reshape(s, (1, H, l_pad, U))
        sf = rfd3_bias.fused_scores_bias_fp32(s, pair_bias, pos_rm, SCALE)
        ttnn.deallocate(s)
        a = softmax_generic.softmax_bf16(sf, ttnn.bfloat16)
        ttnn.deallocate(sf)
        a = ttnn.reshape(a, (H, nb, Q, U))
        o = ttnn.matmul(a, vg)                                   # [H, nb, Q, DH]
        ttnn.deallocate(a)
        ttnn.deallocate(vg)
        return ttnn.reshape(o, (1, H, l_pad, DH))

    # --- correctness, before any timing ----------------------------------------------
    od = ttnn.to_torch(dense_whole()).float()
    ob = ttnn.to_torch(block_whole()).float()
    exact = bool(torch.equal(od, ob))
    diff = (od - ob).abs()
    rel = diff.max().item() / max(1e-12, od.abs().max().item())
    print("\n[p95] CORRECTNESS  bit-exact=%s  maxabs=%.3e  relmax=%.3e  meanabs=%.3e"
          % (exact, diff.max().item(), rel, diff.mean().item()), flush=True)

    # --- timing -----------------------------------------------------------------------
    parts = [
        ("block 1 kv gather x2", lambda: (gather_kv(kk, gather_dev, nb, U, dev),
                                          gather_kv(vv, gather_dev, nb, U, dev))[1]),
        ("block 2 qk matmul", None),
        ("block 3 fused bias", None),
        ("block 4 softmax", None),
        ("block 5 pv matmul", None),
    ]
    kg = gather_kv(kk, gather_dev, nb, U, dev)
    vg = gather_kv(vv, gather_dev, nb, U, dev)
    kgt = ttnn.permute(kg, (0, 1, 3, 2))
    s_b = ttnn.reshape(ttnn.matmul(qb, kgt), (1, H, l_pad, U))
    sf_b = rfd3_bias.fused_scores_bias_fp32(s_b, pair_bias, pos_rm, SCALE)
    a_b = ttnn.reshape(softmax_generic.softmax_bf16(sf_b, ttnn.bfloat16), (H, nb, Q, U))
    parts[1] = ("block 2 qk matmul", lambda: ttnn.matmul(qb, kgt))
    parts[2] = ("block 3 fused bias",
                lambda: rfd3_bias.fused_scores_bias_fp32(s_b, pair_bias, pos_rm, SCALE))
    parts[3] = ("block 4 softmax", lambda: softmax_generic.softmax_bf16(sf_b, ttnn.bfloat16))
    parts[4] = ("block 5 pv matmul", lambda: ttnn.matmul(a_b, vg))

    rows, tot = [], 0.0
    print("\n%-26s %11s" % ("block-sparse parts", "ms/call"), flush=True)
    for name, fn in parts:
        try:
            t = timeit(fn, dev)
            tot += t
            print("%-26s %11.4f" % (name, t), flush=True)
            rows.append(dict(part=name, ms=round(t, 4)))
        except Exception as e:
            print("%-26s EXC %s" % (name, str(e)[:100]), flush=True)
            rows.append(dict(part=name, exc=str(e)[:300]))
    print("%-26s %11.4f" % ("parts sum", tot), flush=True)

    t_dense = timeit(dense_whole, dev)
    t_block = timeit(block_whole, dev)
    print("\n%-26s %11.4f ms/call  ->  %7.3f s/design" % (
        "SHIPPED dense chain", t_dense, t_dense * CALLS * STEPS / 1000.0), flush=True)
    print("%-26s %11.4f ms/call  ->  %7.3f s/design" % (
        "BLOCK-SPARSE chain", t_block, t_block * CALLS * STEPS / 1000.0), flush=True)
    prize = (t_dense - t_block) * CALLS * STEPS / 1000.0
    print("%-26s %11.4f ms/call  ->  %+7.3f s/design  (%.2fx)" % (
        "delta", t_dense - t_block, prize,
        t_dense / t_block if t_block else float("nan")), flush=True)
    print("\n  isolated timing over-prices this site; E5.2 measured the atom-site factor.\n"
          "  Treat %+.3f s/design as an ISOLATED screen, not a fold number." % prize, flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        tag=TAG, H=H, L=L, n_key=NK, head_dim=DH, k_sparse=K, q_block=Q, n_blocks=nb,
        u_width=U, u_max=u_max, union_mean=round(sum(widths) / len(widths), 1),
        union_median=statistics.median(widths), key_axis_narrowing=round(NK / U, 3),
        bit_exact=exact, maxabs=diff.max().item(), relmax=rel, meanabs=diff.mean().item(),
        parts=rows, parts_sum_ms=round(tot, 4),
        dense_chain_ms=round(t_dense, 4), block_chain_ms=round(t_block, 4),
        dense_s_per_design=round(t_dense * CALLS * STEPS / 1000.0, 3),
        block_s_per_design=round(t_block * CALLS * STEPS / 1000.0, 3),
        isolated_prize_s_per_design=round(prize, 3),
        calls_per_step=CALLS, steps=STEPS, reps=REPS,
        fused_enabled=bool(rfd3_bias.fused_enabled()),
        host=os.uname().nodename, card=os.environ.get("TT_VISIBLE_DEVICES"),
    ), indent=2) + "\n")
    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
