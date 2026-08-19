# Size-ladder gate arm — scouting censuses

Lever censuses behind the size-ladder release-gate arm (`scripts/release_gate.py --model
size-ladder`). Produced 2026-08-19 on pc card 0 (Blackhole p150a), `main` a99844f96, ttnn as
installed in `/home/moritz/tt-bio/env`, with:

    python3 scripts/lever_census.py --tt-bio <python> --label b2-<N> --out <file> -- \
        -m tt_bio.main predict perf/size512/fixtures/cdk2x2_<N>.yaml --model boltz2 \
        --single_sequence --sampling_steps 6 --diffusion_samples 1 --seed 0

Two things these files record.

**The ladder** (`128`, `512`, `768`). Seven default-ON levers serve every call at 512 aa and zero at
128 aa. K2 (`TRIATT_PERSISTENT_MASK`) reads 0/560 at 128, 560/0 at 512 and **560/560 at 768** — half
its calls declined at the top rung while its flag still reads True. Fold time over 512->768 scales
as N^3.48 (`runtime_s` 19.37 mean -> 79.3), the cliff first measured on 2026-08-13.

**The RED injection** (`640_divk_on` vs `640_divk_off`). `TT_BIO_SDPA_DIV_K=0` is the pre-K3 state:
`k_chunk` stops dividing the padded sequence, `fill_preconditions` rejects, and the fused K1/K2
kernel declines silently. K2 goes 560/0 -> 0/1680 and no other lever moves. This is the arm's
proof-of-catch condition; it needs no code edit.

Timings are pc card 0 figures and are not comparable to a qb1/qb2 absolute. pc card 0 also
miscomputes some matmuls, so nothing here is a digest or parity claim — these are fired/dark counts
and one timing ratio.
