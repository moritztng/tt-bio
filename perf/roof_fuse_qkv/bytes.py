#!/usr/bin/env python3
"""What the qkv -> SDPA fold actually does to the DRAM ledger, counted per core.

FUSION_PAIRS.md ranks this pair at 805.3 MB by counting the six intermediates' write+read. That
count is right and it is not the fold's byte delta: a fused consumer does not stop reading, it
reads the PRODUCER'S OPERAND instead. Here the operand is the pre-projection pair tensor x, the
projection contracts 128 channels to 32 per head, and the SDPA is split one head per core -- so
every head re-reads the whole of x. This prices that.

No device. Shapes come from the committed buffer-keyed trace
perf/b2z2_byte_floor/out/trace_512_wh_c10.json.gz, the same trace FUSION_PAIRS.md ranked.
"""
from __future__ import annotations

import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TRACE = os.path.join(ROOT, "perf", "b2z2_byte_floor", "out", "trace_512_wh_c10.json.gz")
MB = 1e6                 # decimal MB, the unit FUSION_PAIRS.md and CENSUS.md use


def trace_shapes():
    """(x_bytes, qkv_bytes, w_bytes, n_pairs, B, H, S, d, C) off the committed trace."""
    d = json.load(gzip.open(TRACE, "rt"))
    rows = d["rows"]
    pairs = []
    for i, r in enumerate(rows):
        if r["owner"] != "sdpa_generic.py:463:sdpa":
            continue
        q_id = r["in"][0]
        # the program that wrote q, in program order before this sdpa
        prod = [j for j, p in enumerate(rows[:i])
                if p["owner"] == "mm_generic.py:359:generic_minimal_matmul" and q_id in p["in"]]
        assert prod, f"no producer for {q_id}"
        p = rows[prod[-1]]
        pairs.append((p, r))
    assert pairs, "no sdpa rows"
    p, r = pairs[0]
    x_b = p["in_b"][0]
    w_b = p["in_b"][1]
    B, H, S, dh = (int(v) for v in r["in_s"][0].split("x"))
    C = int(p["in_s"][0].split("x")[-1])
    qkv_b = r["in_b"][0]
    assert all(pp["in_b"][0] == x_b and rr["in_b"][0] == qkv_b for pp, rr in pairs)
    return dict(x_b=x_b, qkv_b=qkv_b, w_b=w_b, n_pairs=len(pairs), B=B, H=H, S=S, d=dh, C=C,
                block_mb=d.get("block_mb"))


def ledger(B, H, S, d, C, q_pf, elem=2):
    """DRAM bytes the qkv projection + SDPA move, for each fold arm. One triangle attention.

    q_pf is the number of q chunks a (batch, head) is split into: the SDPA re-reads k and v once
    per q chunk, which is the term that decides the sign.
    """
    x = B * S * C * elem
    t = B * H * S * d * elem                     # one of q, k, v
    assert t == x, (t, x)                        # H*d == C here, so they are the same size
    today = dict(
        mm_read_x=x,
        mm_write_qkv=3 * t,
        sdpa_read_q=t,
        sdpa_read_kv=2 * q_pf * t,
    )
    # Fold Q only: the mm still makes k and v, the SDPA reads x rows for its own q chunk. Every
    # head reads all C channels of those rows, so the read is H times x's q-chunk slice, and the
    # q chunks of one (batch, head) tile the sequence exactly: H * x.
    q_only = dict(
        mm_read_x=x,
        mm_write_qkv=2 * t,
        sdpa_read_x_for_q=H * x,
        sdpa_read_kv=2 * q_pf * t,
    )
    # Fold q, k and v: the projection program disappears. A core owns (batch subset, one head,
    # one q chunk) and needs the whole sequence's x for k and v, which already contains its q
    # rows -- so it reads x[b] once per (head, q chunk) it owns: H * q_pf * x.
    full = dict(sdpa_read_x_all=H * q_pf * x)
    # Same, with the H cores that share a (batch, q chunk) reading x once and multicasting.
    full_mcast = dict(sdpa_read_x_mcast=q_pf * x)
    return {k: v for k, v in
            dict(today=today, q_only=q_only, full=full, full_mcast=full_mcast).items()}


def main():
    s = trace_shapes()
    B, H, S, d, C = s["B"], s["H"], s["S"], s["d"], s["C"]
    print(f"trace: {TRACE}")
    print(f"  B={B} H={H} S={S} head_dim={d} channels={C}   x={s['x_b']/MB:.1f} MB  "
          f"q=k=v={s['qkv_b']/MB:.1f} MB  W={s['w_b']/1024:.0f} KB  "
          f"{s['n_pairs']} triangle attentions in the block")
    pair_prize = s["n_pairs"] * 3 * 2 * s["qkv_b"]
    print(f"  FUSION_PAIRS.md prize (6 allocations, write+read): {pair_prize/MB:.1f} MB")
    for q_pf in (int(x) for x in sys.argv[1:] or ["1", "2"]):
        L = ledger(B, H, S, d, C, q_pf)
        tot = {k: sum(v.values()) for k, v in L.items()}
        print(f"\n  q chunks per (batch, head) = {q_pf}      [one triangle attention]")
        for arm in ("today", "q_only", "full", "full_mcast"):
            terms = "  ".join(f"{k}={v/MB:.1f}" for k, v in L[arm].items())
            delta = tot[arm] - tot["today"]
            sign = "" if arm == "today" else f"   delta {delta/MB:+.1f} MB"
            print(f"    {arm:11s} {tot[arm]/MB:8.1f} MB   {terms}{sign}")
        print(f"    per block ({s['n_pairs']} attentions): " + ", ".join(
            f"{a} {(tot[a]-tot['today'])*s['n_pairs']/MB:+.1f} MB"
            for a in ("q_only", "full", "full_mcast")))


if __name__ == "__main__":
    main()
