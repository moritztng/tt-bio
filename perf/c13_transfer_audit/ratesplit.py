#!/usr/bin/env python3
"""Within `count`: does it matter whether the RATE was measured on the same kernel?

MEASURED-RATE: the multiplier came from a capture or probe of this very kernel/link.
BORROWED-RATE: the multiplier is a constant from elsewhere -- a per-program launch floor, a
               two-clock 'clock-immune' term, an assumed ns/tile or us/call.
"""
import json, math, random, statistics as st
random.seed(1)
rows=json.load(open("transfer_table.json"))
BORROWED = {
 ("c10-trace-lever","DiT trace replay"),                                   # 13.878 us/call off a two-clock F
 ("c12-host-decomp","F = 3.9830 s read as deletable host"),                # the same F, read as host
 ("c12-cond-hoist-block-timing","cond-hoist at the block"),                # 5.5968 us per-program launch floor
 ("b2z2-step-binaryng-fusion","2-chain BinaryNg fusion (P2)"),             # programs removed x 9.76 us
 ("c10-grid-sweep","host issue per matmul call"),                          # assumed 15-45 us
 ("b2z2-datum-rate-floor","bare move-only kernel ns/tile"),                # assumed 45-60 ns/tile
 ("b2z2-dst-resident-fusion","delete the mul_cb L1 round trip"),           # the campaign's 71.3 ns/tile
 ("b2z2-sharded-sampler","token-axis sampler shard (P3)"),                 # published 10.7 us link fit; latency-bound in fact
 ("b2z2-dual-chip-fold","sharded dual-chip fold"),                         # same published fit
 ("c10-core-grid","core_grid on bare sites (own prereg)"),                 # source read, no rate at all
 ("b2z2-step-fusion-next-sites","unpadded head split (P2)"),
 ("b2z2-step-adaln-sdpa","norm-of-a parallelism (P2)"),
 ("b2z2-l1-sharded-residency","L1LOCAL / DRAM per-tile cost"),
 ("b2z-work-removal","MSA depth ladder 1024 -> 64"),
 ("b2z2-twopass-loop-bh","hoist the two-pass loop init (P2)"),
 ("b2z2-step-matmul-group","N-stacking the s-projections"),
}
# note: c12-cond-hoist's block row is not in the table under that name; it is folded into
# c12-compose-fold's op_ab row. Kept in the set so the membership test is explicit.
sel=[r for r in rows if r["kind"]=="count" and r.get("scored") is not False]
for r in sel:
    r["rate"]="borrowed" if (r["slug"],r["lever"]) in BORROWED else "measured"
for grp in ("measured","borrowed"):
    g=[r for r in sel if r["rate"]==grp and r["outcome"]=="HELD"]
    tot=[r for r in sel if r["rate"]==grp]
    bad=sum(1 for r in tot if r["outcome"]!="HELD")
    lg=[math.log10(r["Rv"]) for r in g]
    geo=10**st.fmean(lg); sc=10**st.stdev(lg)
    bs=sorted(10**st.fmean(random.choices(lg,k=len(lg))) for _ in range(20000))
    print(f"count / {grp.upper():9s} rate   n={len(tot):2d}  geomean R {geo:.3f}  95% CI "
          f"[{bs[500]:.3f}, {bs[19500]:.3f}]  scatter {sc:.2f}x  wrong-sign-or-nulled {bad}/{len(tot)}"
          f" = {100*bad/len(tot):.0f} %")
    print("   " + ", ".join(f"{r['slug']}:{r['Rv']:.3f}" for r in sorted(tot,key=lambda x:x['Rv'])))
    print()
json.dump(rows,open("transfer_table.json","w"),indent=1)
