# BH shipped set (D) vs TT_BIO_LEVERS=none, 7 seeds, unpaired seed-cluster bootstrap

Read 2026-11-02 20:0xZ. Base `grade/bh-ee0..7` (8 runs), test `grade/bh-bD0..6` (7 runs), 8 designs per run,
`perf/spd/bg_unpaired.py` (20000 draws, seeds resampled whole because a numeric lever re-rolls every trajectory).
D is the set that main `d8a8285de` ships on Blackhole after the trunk group was dropped.

| metric | mean diff vs none | 95 % interval |
|---|---|---|
| design_iptm | -0.0206 | [-0.0415, +0.0003] |
| design_ptm | +0.0183 | [+0.0013, +0.0378] |
| complex_plddt | +0.0058 | [+0.0021, +0.0094] |

Per-run design_iptm: none 0.223 0.210 0.193 0.202 0.172 0.236 0.175 0.159; D 0.178 0.154 0.196 0.156 0.171 0.201 0.175.

Verdict: PASS on the charter's normal bar (pooled interval covers 0), and ptm and plddt move the right way with
intervals that exclude 0. This was pre-registered at 7 seeds after the 5-seed read (-0.024 [-0.047, -0.0005]) just
missed 0; the consequence of a miss was to drop the diffusion group as well, and that does not trigger.

Honest caveat: the point estimate is -0.021, larger than grade.py's 0.005 confidence tolerance, and the interval
only just covers 0. Per-design design_iptm sd is 0.098 at 8 designs a seed, so no design-run grade resolves better
than ~0.02 for sane money. The earlier arms on the same base: T (trunk group) -0.029 [-0.048, -0.010] is the one
arm whose interval is wholly worse, and it is already out of both sets; L (base + host levers only) is
-0.007 [-0.034, +0.023].
