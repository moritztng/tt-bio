# Relayed artifacts — evidence produced on qb2, which cannot push

qb2 has no push credentials, so a child running there can leave its only evidence uncommitted in a
worktree. `c12-kblock-unlock/`'s sweep was `?? perf/c12_kblock/` (untracked, not even a local
commit) when the parent checked at 2026-09-17 10:39Z, and the whole of C12's Axis-B measurement
rests on it. Copied here so it survives; the child still owns the path in its own branch, and if it
commits and relays its own copy later, delete this one rather than keeping two.

`c12_kblock/sweep_512_qb2c1.json` — 6 keys x up to 12 configs, qb2 node 1, pinned and
during-sampled 1350 MHz, arms interleaved rep by rep, every arm scored against a float64 reference
of its own bf16 operands. The two rows this parent used:

    pair_tr_fc1  tenstorrent.py:8211  act=silu  [1,16,512,128]x[128,512] L1->L1  8960 calls
                 shipped 0.1201 ms/call   8.94 TFLOP/s   88.4 GB/s
    pair_tr_fc2  tenstorrent.py:8222  act=None  same shape, same config, same counts
                 shipped 0.0314 ms/call  34.21 TFLOP/s  338.2 GB/s

Byte-for-byte the same operation (1,073,741,824 FLOPs, 10,616,832 B, derived config 1d/bw2/obh3/
obw16 on both), so the 0.0887 ms/call gap is the silu epilogue and nothing else. Over 8,960 calls
that is 0.7947 s / 1,073 Mcycles, 5.3 % of the 14.881 s fold. Row `c12-unfused-silu-bh` owns it.
