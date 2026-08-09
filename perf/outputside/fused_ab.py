#!/usr/bin/env python3
"""A/B the fused trimul output op against the ttnn permute + running concat.

Both arms in one process on the same device and allocator, amortized issues per
synchronize..synchronize region. Parity is torch.equal against the ttnn chain: the fused op
only moves bytes, so bit-exactness is the bar, not PCC.
"""
import argparse, json, statistics, sys, time
from pathlib import Path
import torch, ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device            # noqa: E402
from tt_bio import trimul_out_fused as F             # noqa: E402

L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG


def timed(dev, fn, issues=4, reps=5, warm=2):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(issues):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3 / issues)
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--c", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--gx", type=int, default=0)   # 0 = whatever this card reports
    ap.add_argument("--gy", type=int, default=0)
    ap.add_argument("--dst", default="dram", choices=["dram", "l1"])
    ap.add_argument("--out", default="fused_ab.json")
    a = ap.parse_args()
    N, C, HID = a.n, a.c, a.hidden
    npairs = HID // C
    dev = get_device()
    dg = dev.compute_with_storage_grid_size()
    # Kernels on a dispatch core are rejected outright ("Kernels cannot be placed on dispatch
    # cores"), and the card grid is not the same on every card: card 0 reports 11x10.
    grid = (a.gx or dg.x, a.gy or dg.y)
    dmc = DRAM if a.dst == "dram" else L1
    res = {"n": N, "c": C, "hidden": HID, "npairs": npairs, "grid": f"{grid[0]}x{grid[1]}",
           "card_grid": f"{dg.x}x{dg.y}", "dst": a.dst, "arms": {}}
    print(f"card grid {dg.x}x{dg.y}, using {grid[0]}x{grid[1]}; N={N} C={C} hidden={HID} "
          f"npairs={npairs} dst={a.dst}", flush=True)

    torch.manual_seed(0)
    src_t = [torch.randn(1, C, N, N) for _ in range(npairs)]
    src = [ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=L1) for t in src_t]

    # ---- reference: what production does today ----
    def prod_chain():
        x = None
        for i, s in enumerate(src):
            xc = ttnn.permute(s, (0, 2, 3, 1), memory_config=L1)
            if i == 0:
                x = ttnn.clone(xc, memory_config=DRAM)
                ttnn.deallocate(xc)
            else:
                xo = x
                x = ttnn.concat([xo, xc], dim=-1)
                ttnn.deallocate(xo)
                ttnn.deallocate(xc)
        return x

    ref_dev = prod_chain()
    ref = ttnn.to_torch(ref_dev)
    ttnn.deallocate(ref_dev)

    # ---- fused ----
    shape = ttnn.Shape([1, N, N, HID])
    dst = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, dmc)
    for i, s in enumerate(src):
        F.fused_output(s, dst, i, grid=grid)
    got = ttnn.to_torch(dst)
    exact = bool(torch.equal(ref, got))
    res["bit_exact_vs_ttnn_chain"] = exact
    print(f"bit-exact vs ttnn permute+concat: {exact}", flush=True)
    if not exact:
        d = (ref.float() - got.float()).abs()
        res["max_abs_diff"] = float(d.max())
        res["frac_wrong"] = float((d > 0).float().mean())
        print(f"  max|diff| {d.max():.4g}  fraction wrong {(d>0).float().mean():.4f}", flush=True)

    chunk_B = C * N * N * 2
    full_B = HID * N * N * 2

    def rec(name, ms, rw, note=""):
        res["arms"][name] = {"ms": ms, "gbs": rw / (ms * 1e-3) / 1e9, "bytes": rw, "note": note}
        print(f"{name:34s} {ms:9.4f} ms  {rw/(ms*1e-3)/1e9:8.1f} GB/s  {note}", flush=True)

    rw_prod = npairs * 2 * chunk_B + 2 * chunk_B + sum(2 * (k * chunk_B) + chunk_B
                                                       for k in range(1, npairs))

    def prod_arm():
        ttnn.deallocate(prod_chain())
    rec("production permute+running concat", timed(dev, prod_arm, issues=2, reps=5), rw_prod,
        "the chain being replaced")

    def fused_arm():
        for i, s in enumerate(src):
            F.fused_output(s, dst, i, grid=grid)
    rec("fused output op", timed(dev, fused_arm, issues=2, reps=5), 2 * full_B,
        "reads each chunk once, writes the stripe")

    # Where the whole-tile half of the op loses to a clone of the same bytes. Both controls
    # produce WRONG data: they swap the real page index for a sequential one, so a big drop
    # means the real index pattern is hitting a bank the sequential one spreads over.
    for lbl, kw in (("noex", dict(no_exchange=True)),
                    ("noex + seq write pages", dict(no_exchange=True, seq_write=True)),
                    ("noex + seq read pages", dict(no_exchange=True, seq_read=True)),
                    ("noex + seq both", dict(no_exchange=True, seq_read=True, seq_write=True)),
                    ("full + seq write pages", dict(seq_write=True)),
                    ("noex + write stride 2", dict(no_exchange=True, seq_write=True, seq_stride=2)),
                    ("noex + write stride 4", dict(no_exchange=True, seq_write=True, seq_stride=4)),
                    ("noex + write stride 8", dict(no_exchange=True, seq_write=True, seq_stride=8)),
                    ("noex + write stride 16", dict(no_exchange=True, seq_write=True, seq_stride=16))):
        def arm(kw=kw):
            for i, s in enumerate(src):
                F.fused_output(s, dst, i, grid=grid, **kw)
        try:
            rec(f"  {lbl} (WRONG data)", timed(dev, arm, issues=2, reps=5), 2 * full_B)
        except Exception as e:                                            # noqa: BLE001
            print(f"  {lbl} ERR {str(e)[:80]}", flush=True)

    # RBATCH: tiles per dst-register acquire. DEPTH: groups of CB, i.e. whether the reader,
    # compute kernel and writer overlap at all.
    best = None
    for depth in (1, 2):
        for rb in (1, 2, 4, 8):
            def arm(rb=rb, depth=depth):
                for i, s in enumerate(src):
                    F.fused_output(s, dst, i, grid=grid, rbatch=rb, depth=depth)
            try:
                ms = timed(dev, arm, issues=2, reps=5)
            except Exception as e:                                        # noqa: BLE001
                print(f"  rbatch={rb} depth={depth} ERR {str(e)[:80]}", flush=True)
                continue
            rec(f"  rbatch={rb} depth={depth}", ms, 2 * full_B)
            if best is None or ms < best[0]:
                best = (ms, rb, depth)
    if best:
        res["best_config"] = {"ms": best[0], "rbatch": best[1], "depth": best[2]}
        # parity at the winning configuration, not just at the default
        for i, s in enumerate(src):
            F.fused_output(s, dst, i, grid=grid, rbatch=best[1], depth=best[2])
        ok = bool(torch.equal(ref, ttnn.to_torch(dst)))
        res["best_config"]["bit_exact"] = ok
        print(f"best rbatch={best[1]} depth={best[2]} {best[0]:.4f} ms, bit-exact {ok}", flush=True)

    # Splits the op: same whole-tile NOC traffic and the same transpose, exchange skipped.
    # Produces wrong data on purpose, so it runs after the parity check and never feeds one.
    def noex_arm():
        for i, s in enumerate(src):
            F.fused_output(s, dst, i, grid=grid, no_exchange=True)
    rec("fused, exchange skipped (WRONG data)", timed(dev, noex_arm, issues=2, reps=5),
        2 * full_B, "the op's whole-tile floor")

    # the byte floor: same traffic, no reordering
    perm = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                           dtype=ttnn.bfloat16, memory_config=L1)
    rec("BOUND clone x%d L1->%s" % (npairs, a.dst.upper()),
        timed(dev, lambda: [ttnn.deallocate(ttnn.clone(perm, memory_config=dmc))
                            for _ in range(npairs)], issues=2, reps=5),
        2 * full_B, "the fused op's byte floor")
    ttnn.deallocate(perm)

    # DRAM->DRAM copy roof on this card, for the qkv chain
    big = ttnn.from_torch(torch.randn(1, N, N, HID), layout=ttnn.TILE_LAYOUT, device=dev,
                          dtype=ttnn.bfloat16, memory_config=DRAM)
    rec("roof clone DRAM->DRAM", timed(dev, lambda: ttnn.deallocate(
        ttnn.clone(big, memory_config=DRAM)), issues=4, reps=5), 2 * full_B, "copy roof")
    ttnn.deallocate(big)

    print(json.dumps(res, indent=1), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


main()
