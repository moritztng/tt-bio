#!/usr/bin/env python3
"""D6 -- the trimul output-side chain and the tri-attention qkv chain, measured.

The chain under test is tenstorrent.py:1043-1075: the triangle matmul emits [1,C,N,N] in L1, a
permute(0,2,3,1) brings the channel chunk back to the last axis, and a running concat glues the
chunks into [1,N,N,hidden] in DRAM. Everything here is amortized (pipe issues per
synchronize..synchronize region) because W4 measured 0.02-0.05 ms of per-region overhead, which is
20%+ of a 13 MB op.

Roofs are measured on THIS card, never inherited.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path
import torch, ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device  # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
MC = {"dram": DRAM, "l1": L1}


def timed(dev, fn, warm=4, pipe=8, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / pipe)
    return st.median(out)


def mk(dev, shape, mc, seed=0):
    torch.manual_seed(seed)
    return ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--hidden", type=int, default=256)   # protenix-v2 c_hidden; opendde 384
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--heads", type=int, default=8)      # protenix-v2 tri-att heads; opendde 12
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    N, C, HID = a.n, a.chunk, a.hidden
    npairs = HID // C
    dev = get_device()
    dg = dev.compute_with_storage_grid_size()
    res = {"card_grid": f"{dg.x}x{dg.y}", "ttnn": getattr(ttnn, "__version__", "?"),
           "n": N, "chunk": C, "hidden": HID, "npairs": npairs, "heads": a.heads}
    print(f"grid={dg.x}x{dg.y} ttnn={res[ttnn]} N={N} C={C} hidden={HID} npairs={npairs}", flush=True)

    chunk_B = N * N * C * 2
    full_B = N * N * HID * 2

    # ---------------- roofs on this card ----------------
    print("\n=== roofs (ttnn.clone, this card) ===", flush=True)
    roofs = {}
    for lbl, shape, src, dst in (
            ("chunk_l1_to_l1", (1, N, N, C), L1, L1),
            ("chunk_l1_to_dram", (1, N, N, C), L1, DRAM),
            ("chunk_dram_to_dram", (1, N, N, C), DRAM, DRAM),
            ("full_dram_to_dram", (1, N, N, HID), DRAM, DRAM),
            ("full_dram_to_l1", (1, N, N, HID), DRAM, L1)):
        nb = 1
        for d in shape:
            nb *= d
        nb *= 2
        try:
            x = mk(dev, shape, src)
            s = timed(dev, lambda: ttnn.deallocate(ttnn.clone(x, memory_config=dst)))
            ttnn.deallocate(x)
            roofs[lbl] = {"ms": round(s * 1e3, 4), "MB": round(nb / 1e6, 2),
                          "GBs_rw": round(2 * nb / s / 1e9, 1)}
            print(f"  {lbl:20s} {s*1e3:8.4f} ms  {2*nb/s/1e9:7.1f} GB/s (read+write)", flush=True)
        except Exception as e:                                            # noqa: BLE001
            roofs[lbl] = {"err": str(e)[:120]}
            print(f"  {lbl:20s} ERR {str(e)[:90]}", flush=True)
    res["roofs"] = roofs

    # ---------------- the production output-side chain ----------------
    print("\n=== production output-side chain (permute + running concat) ===", flush=True)
    src = [mk(dev, (1, C, N, N), L1, seed=i) for i in range(npairs)]

    def prod():
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
        ttnn.deallocate(x)

    t_prod = timed(dev, prod, warm=3, pipe=4, reps=5)
    # bytes the chain moves: npairs permutes (r+w chunk), one clone (r+w chunk),
    # and the running concat (read acc + chunk, write acc+chunk) for i=1..npairs-1
    permute_B = npairs * 2 * chunk_B
    clone_B = 2 * chunk_B
    concat_B = sum(2 * (i + 1) * chunk_B for i in range(1, npairs))
    chain_B = permute_B + clone_B + concat_B
    res["prod_chain"] = {"ms": round(t_prod * 1e3, 4), "moved_MB": round(chain_B / 1e6, 2),
                         "GBs": round(chain_B / t_prod / 1e9, 1)}
    print(f"  whole chain            {t_prod*1e3:8.4f} ms  moves {chain_B/1e6:7.2f} MB"
          f"  {chain_B/t_prod/1e9:7.1f} GB/s", flush=True)

    legs = {}
    def leg(name, fn, nbytes, warm=3, pipe=4, reps=5):
        try:
            s = timed(dev, fn, warm=warm, pipe=pipe, reps=reps)
        except Exception as e:                                            # noqa: BLE001
            legs[name] = {"err": str(e)[:140]}
            print(f"  {name:22s} ERR {str(e)[:90]}", flush=True)
            return None
        legs[name] = {"ms": round(s * 1e3, 4), "MB": round(nbytes / 1e6, 2),
                      "GBs": round(nbytes / s / 1e9, 1)}
        print(f"  {name:22s} {s*1e3:8.4f} ms  moves {nbytes/1e6:7.2f} MB  {nbytes/s/1e9:7.1f} GB/s",
              flush=True)
        return s

    leg("permute x%d ->L1" % npairs,
        lambda: [ttnn.deallocate(ttnn.permute(s, (0, 2, 3, 1), memory_config=L1)) for s in src],
        permute_B)
    leg("permute x%d ->DRAM" % npairs,
        lambda: [ttnn.deallocate(ttnn.permute(s, (0, 2, 3, 1), memory_config=DRAM)) for s in src],
        permute_B)

    def decomp(mc):
        for s in src:
            t = ttnn.transpose(s, 1, 2, memory_config=mc)
            u = ttnn.transpose(t, 2, 3, memory_config=mc)
            ttnn.deallocate(t)
            ttnn.deallocate(u)
    leg("transpose12+23 ->L1", lambda: decomp(L1), permute_B)
    leg("transpose12+23 ->DRAM", lambda: decomp(DRAM), permute_B)

    # the byte floor a fused permute-into-the-destination would pay: read each chunk from L1
    # once, write it once into DRAM. No intermediate, no concat.
    perm = [mk(dev, (1, N, N, C), L1, seed=100 + i) for i in range(npairs)]
    leg("BOUND clone x%d L1->DRAM" % npairs,
        lambda: [ttnn.deallocate(ttnn.clone(p, memory_config=DRAM)) for p in perm], 2 * npairs * chunk_B)
    leg("BOUND clone x%d L1->L1" % npairs,
        lambda: [ttnn.deallocate(ttnn.clone(p, memory_config=L1)) for p in perm], 2 * npairs * chunk_B)

    # concat legs, given already-permuted chunks
    acc0 = ttnn.clone(perm[0], memory_config=DRAM)
    def running_concat():
        x = ttnn.clone(acc0, memory_config=DRAM)
        for i in range(1, npairs):
            xo = x
            x = ttnn.concat([xo, perm[i]], dim=-1)
            ttnn.deallocate(xo)
        ttnn.deallocate(x)
    leg("running concat (+clone)", running_concat, clone_B + concat_B)
    leg("single concat all %d" % npairs,
        lambda: ttnn.deallocate(ttnn.concat(list(perm), dim=-1)), 2 * full_B)
    ttnn.deallocate(acc0)
    for p in perm:
        ttnn.deallocate(p)
    for s in src:
        ttnn.deallocate(s)
    res["legs"] = legs

    # ---------------- tri-attention qkv chain ----------------
    print("\n=== tri-attention nlp_create_qkv_heads ===", flush=True)
    qkv_ch = 3 * a.heads * 32
    qkv = mk(dev, (N, 1, N, qkv_ch), DRAM, seed=7)
    qkv_B = N * N * qkv_ch * 2

    def split():
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=a.heads, num_kv_heads=a.heads, transpose_k_heads=False,
            memory_config=DRAM)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
    t_split = timed(dev, split, warm=2, pipe=4, reps=5)
    res["qkv"] = {"ms": round(t_split * 1e3, 4), "moved_MB": round(2 * qkv_B / 1e6, 2),
                  "GBs": round(2 * qkv_B / t_split / 1e9, 1), "in_MB": round(qkv_B / 1e6, 2)}
    print(f"  nlp_create_qkv_heads   {t_split*1e3:8.4f} ms  moves {2*qkv_B/1e6:7.2f} MB"
          f"  {2*qkv_B/t_split/1e9:7.1f} GB/s", flush=True)
    # control: a clone of the same bytes, DRAM->DRAM. Same traffic, no reordering.
    t_cl = timed(dev, lambda: ttnn.deallocate(ttnn.clone(qkv, memory_config=DRAM)), warm=2, pipe=4, reps=5)
    res["qkv_clone_control"] = {"ms": round(t_cl * 1e3, 4), "GBs": round(2 * qkv_B / t_cl / 1e9, 1)}
    print(f"  clone control          {t_cl*1e3:8.4f} ms  {2*qkv_B/t_cl/1e9:7.1f} GB/s", flush=True)
    ttnn.deallocate(qkv)

    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"\nwrote {a.out}", flush=True)


main()
