# Lever census, main vs 840c5f844 (TT_BIO_TRIATT_DIVIDING_K on)

qb2 card 0 (p300c), 2026-09-27 07:35-07:51Z, `scripts/lever_census.py` around the size-ladder
arm's own fold command for each model. Main is `0ebcaae1f`.

| cell | levers that differ between main and 840c5f844 |
|---|---|
| openbind 256 | none |
| esmfold2 256 | none |
| nesso1 256 | none |
| nesso1 1024 | none |
| openbind 1152 | none |

The size-ladder FAILs on 840c5f844 (qb1 p150a, `gate_dividingk.json`) name SDPA_WIDE_K,
TRIATT_SDPA_HIFI, TRIATT_PERSISTENT_MASK, SDPA_FUSED_LARGE_S, SPLIT_SWIGLU and others. Every one
reads the same on main: esmfold2 SPLIT_SWIGLU is 578/157 (frac 0.786) on both. The recorded
Blackhole baselines predate main's SDPA_WIDE_K default and the per-site TRIATT_SDPA_HIFI
flag, so main fails this arm without the candidate.
