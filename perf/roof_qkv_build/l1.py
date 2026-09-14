#!/usr/bin/env python3
"""What the fold costs L1, at every size the gate would take it, without a device.

The row's third kill criterion is the one-size defect: a lever that fits at 512 aa and not at
298 aa is not a lever. `sdpa_generic.cb_bytes` prices the whole CB surface exactly (ten measured
refusals, to the byte), so this is decidable on the host.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ttnn  # noqa: E402

from tt_bio import sdpa_generic as SG  # noqa: E402

H, DH = 4, 32
C = H * DH
CORES = int(os.environ.get("ROOF_CORES", "130"))


def row(seq, q_chunk, k_chunk, xbf):
    p = SG.plan_for_shape(seq, H, DH, q_chunk, k_chunk, grid=(CORES, 1),
                          split=(CORES // H, H, 1))
    kvbf = p["k_num_chunks"]
    p = SG.plan(SG._ShapeOnly([seq, H, seq, DH]), SG._ShapeOnly([seq, H, seq, DH]),
                SG._ShapeOnly([seq, H, seq, DH]), SG._ShapeOnly([1, H, seq, seq]),
                SG._ShapeOnly([seq, H, seq, DH]), q_chunk, k_chunk, (CORES, 1),
                (ttnn.MathFidelity.HiFi2, True, False, False), 1.0, (CORES // H, H, 1), kvbf)
    pers = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
    Ct = C // 32
    base = SG.cb_bytes(p, mask_cb_tiles=pers)
    extra = SG.fuse_qkv_cbs(p, Ct, xbf)
    add = sum(n * page for _i, n, page, _f in extra)
    budget = SG.L1_PER_CORE - SG.PROGRAM_RESERVE
    return dict(seq=seq, q=q_chunk, k=k_chunk, xbf=xbf, kvbf=kvbf,
                Sq_chunk_t=p["Sq_chunk_t"], Skt=p["Skt"], k_chunks=p["k_num_chunks"],
                base=base, add=add, total=base + add, free=budget - base - add,
                sub_h=SG.proj_subblock_h(p, Ct), fits=base + add <= budget)


def main():
    print(f"cores={CORES}  budget={SG.L1_PER_CORE - SG.PROGRAM_RESERVE} B per core "
          f"(L1 {SG.L1_PER_CORE} - program reserve {SG.PROGRAM_RESERVE})")
    print(f"{'seq':>5} {'q':>5} {'k':>5} {'xbuf':>4} {'kvbuf':>5} {'sub_h':>5} "
          f"{'base B':>9} {'fold B':>8} {'total B':>9} {'free B':>9}  fits")
    for seq in (320, 512):
        for k_chunk in SG.chunk_divisors(seq):
            if k_chunk > seq or seq % k_chunk:
                continue
            for xbf in (2, 1):
                r = row(seq, seq, k_chunk, xbf)
                print(f"{r['seq']:>5} {r['q']:>5} {r['k']:>5} {r['xbf']:>4} {r['kvbf']:>5} "
                      f"{r['sub_h']:>5} {r['base']:>9} {r['add']:>8} {r['total']:>9} "
                      f"{r['free']:>9}  {'yes' if r['fits'] else 'NO'}")
            if k_chunk == 256 and seq == 512:
                pass
        print()


if __name__ == "__main__":
    main()
