#!/usr/bin/env python3
"""How much of the fused trimul output op's gap to the copy roof is DRAM bank spread.

D6 showed the op is at 31.9% of the L1->DRAM clone roof because one channel chunk's
destination pages reach 2 of this card's 8 DRAM banks. It priced the fix off a synthetic
stride walk. This probe measures the real thing: CTSPREAD keeps every group's 32-write
pattern exactly as it is and only walks the groups across more column stripes, so the grid
reaches 2/4/8 banks with nothing else changed. The data is wrong on purpose (the chunks
overwrite each other); this is a timing control, and it prices the 2-chunk and 4-chunk
variants before either is built.
"""
import argparse, json, statistics, sys, time
from pathlib import Path
import torch, ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device            # noqa: E402
from tt_bio import trimul_out_fused as F             # noqa: E402

L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG


def timed(dev, fn, issues=2, reps=5, warm=2):
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
    ap.add_argument("--out", default="perf/fourbank/spread.json")
    a = ap.parse_args()
    N, C, HID = a.n, a.c, a.hidden
    npairs = HID // C
    dev = get_device()
    dg = dev.compute_with_storage_grid_size()
    grid = (dg.x, dg.y)
    res = {"n": N, "c": C, "hidden": HID, "npairs": npairs,
           "grid": f"{grid[0]}x{grid[1]}", "arms": {}}
    print(f"grid {grid[0]}x{grid[1]}  N={N} C={C} hidden={HID} npairs={npairs}", flush=True)

    torch.manual_seed(0)
    src = [ttnn.from_torch(torch.randn(1, C, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                           dtype=ttnn.bfloat16, memory_config=L1) for _ in range(npairs)]
    dst = ttnn.allocate_tensor_on_device(ttnn.Shape([1, N, N, HID]), ttnn.bfloat16,
                                         ttnn.TILE_LAYOUT, dev, DRAM)

    # parity of the unmodified op first, so a CTSPREAD regression in the shared kernel
    # source cannot pass unnoticed
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
    for i, s in enumerate(src):
        F.fused_output(s, dst, i, grid=grid)
    res["bit_exact_ct_spread_1"] = bool(torch.equal(ref, ttnn.to_torch(dst)))
    print(f"bit-exact, ct_spread=1: {res['bit_exact_ct_spread_1']}", flush=True)

    full_B = HID * N * N * 2
    rw = 2 * full_B

    def rec(name, ms, note=""):
        res["arms"][name] = {"ms": ms, "gbs": rw / (ms * 1e-3) / 1e9}
        print(f"{name:38s} {ms:9.4f} ms  {rw/(ms*1e-3)/1e9:8.1f} GB/s  {note}", flush=True)

    # exch: 32 = production, 16 = HALFEXCH (half the exchange, wrong data), 0 = none.
    # The 32/16/0 triple at each bank count says whether the exchange is linear in the work it
    # does once the bank fix stops the slow write from hiding it, which is what decides whether
    # splitting it across both dataflow RISCs is worth a kernel change.
    for spread in (1, 2, 4):
        for exch in (32, 16, 0):
            def arm(spread=spread, exch=exch):
                for i, s in enumerate(src):
                    F.fused_output(s, dst, i, grid=grid, ct_spread=spread,
                                   no_exchange=(exch == 0), half_exch=(exch == 16))
            tag = {32: "", 16: " half_exch", 0: " noex"}[exch]
            lbl = f"ct_spread={spread} ({spread*(C//32)} banks){tag}"
            try:
                rec(lbl, timed(dev, arm),
                    "REAL DATA" if (spread == 1 and exch == 32) else "wrong data, timing only")
            except Exception as e:                                        # noqa: BLE001
                print(f"  {lbl} ERR {str(e)[:100]}", flush=True)

    # The real thing: two chunks per launch, correct data, 4 banks live. Must land within
    # ~5% of the ct_spread=2 control or the group->chunk mapping is not producing the spread.
    def pair_arm(**kw):
        for i in range(0, npairs - 1, 2):
            F.fused_output_pair(src[i], src[i + 1], dst, i, i + 1, grid=grid, **kw)
        if npairs % 2:
            F.fused_output(src[-1], dst, npairs - 1, grid=grid, **kw)

    pair_arm()
    res["bit_exact_pair"] = bool(torch.equal(ref, ttnn.to_torch(dst)))
    print(f"bit-exact, paired: {res['bit_exact_pair']}", flush=True)
    rec("PAIR (4 banks)", timed(dev, pair_arm), "REAL DATA")
    rec("PAIR (4 banks) half_exch", timed(dev, lambda: pair_arm(half_exch=True)),
        "wrong data, timing only")
    rec("PAIR (4 banks) noex", timed(dev, lambda: pair_arm(no_exchange=True)),
        "wrong data, timing only")

    # Pairing does two things at once: it doubles the bank spread AND halves the launch
    # count. Sending both chunks of a pair to the SAME column stripe keeps the halved launch
    # count with the 2-bank spread of today, so the difference to PAIR is the banks alone.
    def pair_same_stripe():
        for i in range(0, npairs - 1, 2):
            F.fused_output_pair(src[i], src[i + 1], dst, i, i, grid=grid)
    rec("PAIR (2 banks, one stripe)", timed(dev, pair_same_stripe),
        "wrong data, timing only")

    perm = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                           dtype=ttnn.bfloat16, memory_config=L1)
    rec(f"ROOF clone x{npairs} L1->DRAM",
        timed(dev, lambda: [ttnn.deallocate(ttnn.clone(perm, memory_config=DRAM))
                            for _ in range(npairs)]), "same bytes, no reordering")
    ttnn.deallocate(perm)

    print(json.dumps(res, indent=1), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


main()
