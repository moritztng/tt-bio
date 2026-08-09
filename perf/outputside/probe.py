#!/usr/bin/env python3
"""D6 output-side probe: the trimul's permute(0,2,3,1) + concat, and tri_att's qkv chain.

Every arm is amortized: ISSUES launches between two synchronize_device calls, median of
REPS regions, so per-region fixed overhead (0.02-0.05 ms, W4) cannot masquerade as op time.
GB/s counts read + write bytes.

    TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=... python3 probe.py --n 320 --c 32
"""
import argparse, json, statistics, time
import torch, ttnn
from tt_bio.tenstorrent import get_device

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG


def timed(fn, issues, reps=5):
    dev = get_device()
    # warm / compile
    for _ in range(2):
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
    ap.add_argument("--c", type=int, default=32)
    ap.add_argument("--cz", type=int, default=256)
    ap.add_argument("--out", default="probe.json")
    a = ap.parse_args()
    N, C, CZ = a.n, a.c, a.cz
    dev = get_device()
    n_pairs = CZ // C
    res = {"n": N, "c": C, "cz": CZ, "n_pairs": n_pairs, "arms": {}}

    def rec(name, ms, rw_bytes, note=""):
        res["arms"][name] = {"ms": ms, "gbs": rw_bytes / (ms * 1e-3) / 1e9, "bytes": rw_bytes, "note": note}
        print(f"{name:32s} {ms:9.4f} ms  {rw_bytes/(ms*1e-3)/1e9:8.1f} GB/s  {note}", flush=True)

    # ---- the matmul-output shape: [1, C, N, N] in L1 ----
    xc_t = torch.randn(1, C, N, N, dtype=torch.float32)
    xc = ttnn.from_torch(xc_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=L1)
    B = C * N * N * 2                      # payload bytes of one channel chunk
    print(f"chunk payload {B/2**20:.2f} MiB, n_pairs {n_pairs}", flush=True)

    # roofs on THIS card, same bytes
    rec("roof_clone_L1->L1", timed(lambda: ttnn.clone(xc, memory_config=L1), 8), 2 * B, "copy roof, L1 dest")
    rec("roof_clone_L1->DRAM", timed(lambda: ttnn.clone(xc, memory_config=DRAM), 8), 2 * B, "copy roof, DRAM dest")

    # the production output-side permute
    rec("permute(0,2,3,1)->L1", timed(lambda: ttnn.permute(xc, (0, 2, 3, 1), memory_config=L1), 8), 2 * B)
    rec("permute(0,2,3,1)->DRAM", timed(lambda: ttnn.permute(xc, (0, 2, 3, 1), memory_config=DRAM), 8), 2 * B)

    def two_transpose(mc):
        t = ttnn.transpose(xc, 1, 2, memory_config=mc)
        u = ttnn.transpose(t, 2, 3, memory_config=mc)
        ttnn.deallocate(t)
        ttnn.deallocate(u)
    rec("transpose(1,2)+(2,3)->L1", timed(lambda: two_transpose(L1), 8), 4 * B, "two passes")
    rec("transpose(1,2)+(2,3)->DRAM", timed(lambda: two_transpose(DRAM), 8), 4 * B, "two passes")

    # parity of the alternatives against the single permute
    ref = ttnn.to_torch(ttnn.permute(xc, (0, 2, 3, 1), memory_config=L1))
    t = ttnn.transpose(xc, 1, 2, memory_config=L1)
    alt = ttnn.to_torch(ttnn.transpose(t, 2, 3, memory_config=L1))
    res["transpose2_bit_exact_vs_permute"] = bool(torch.equal(ref, alt))
    print("transpose2 bit-exact vs permute:", res["transpose2_bit_exact_vs_permute"], flush=True)
    del t, alt

    # ---- the concat half ----
    perm = ttnn.permute(xc, (0, 2, 3, 1), memory_config=L1)   # [1,N,N,C]
    PB = C * N * N * 2

    def running_concat():
        acc = ttnn.clone(perm, memory_config=DRAM)
        for _ in range(n_pairs - 1):
            old = acc
            acc = ttnn.concat([old, perm], dim=-1)
            ttnn.deallocate(old)
        ttnn.deallocate(acc)
    rw_run = 2 * PB + sum(2 * (k * PB) + PB for k in range(1, n_pairs))
    rec("concat_running (production)", timed(running_concat, 2, 3), rw_run, f"{n_pairs} chunks, O(n^2)")

    parts = [ttnn.clone(perm, memory_config=L1) for _ in range(n_pairs)]

    def single_concat():
        o = ttnn.concat(parts, dim=-1)
        ttnn.deallocate(o)
    rec("concat_single (W2)", timed(single_concat, 4, 3), 2 * n_pairs * PB, f"{n_pairs} chunks, O(n)")
    for p in parts:
        ttnn.deallocate(p)
    ttnn.deallocate(perm)

    # ---- tri_att qkv chain at the real shape ----
    x = ttnn.from_torch(torch.randn(1, N * N, CZ), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    w = ttnn.from_torch(torch.randn(CZ, 3 * 8 * 32), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    qkv_bytes = N * N * CZ * 2 + N * N * 768 * 2

    def proj():
        o = ttnn.experimental.minimal_matmul(input_tensor=x, weight_tensor=w, dtype=ttnn.bfloat16)
        ttnn.deallocate(o)
    rec("qkv minimal_matmul", timed(proj, 4, 3), qkv_bytes, "[102400,256]@[256,768] DRAM")

    qkv = ttnn.experimental.minimal_matmul(input_tensor=x, weight_tensor=w, dtype=ttnn.bfloat16)
    qkv_u = ttnn.unsqueeze(qkv, 1)

    def split():
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv_u, num_heads=8, num_kv_heads=8, transpose_k_heads=False,
            memory_config=qkv_u.memory_config())
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
    rec("nlp_create_qkv_heads", timed(split, 4, 3), 2 * N * N * 768 * 2, "read+rewrite 157MB")

    print(json.dumps(res, indent=1))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


main()
