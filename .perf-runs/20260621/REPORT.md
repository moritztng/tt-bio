# tt-bio perf — 2026-06-21 — diffusion residency + rel-pos embedding

Branch: exp/perf-20260621-diffusion-resident  (off main @ 8a40c42)
Card: tt-quietbox /dev/tenstorrent/0 (all 4 free at start)

## Baseline (main, warm L=512 --fast, card 0)
cmd: TT_BOLTZ_STAGE_TIME=1 env/bin/tt-bio predict ~/.tt-bio-perf/in512 --fast --device_ids 0 --seed 0
- cold p512a = 203.3s (incl compile)
- WARM p512b = 37.9s   <- BEAT THIS
- stage wall-clock (from timestamps): embedder ~1s | trunk ~25s (66%) | diffusion ~10s (26%) | confidence ~2s
- Known: trunk = compute-bound (only win = device-resident recycle, already done on exp/perf-resident-trunk).
  25s warm floor reachable via FULL tt-minimal resident stack (trunk + diffusion-resident + cond handoff + rel-pos embed).

## Plan
Target the diffusion + rel-pos pieces NOT yet on main (trunk residency is a separate branch).

## Device-synced warm baselines (stagehook, --fast, card 0)
| Size | trunk | diffusion | conf | e2e | diff ms/step |
|---|---|---|---|---|---|
| L=256 | 8.98s | 7.55s (42%) | 0.42s | 17.8s | 37.7 |
| L=512 | 24.42s | 9.92s (27%) | 1.71s | 37.5s | 49.6 |
| L=686 | 42.71s | 15.0s (24%) | 3.41s | 62.8s | 75 |

Diffusion ms/step grows only 1.35x for 2x atoms (256->512) => ~25-30ms fixed per-step dispatch floor.
At L=256 the floor dominates (host-dispatch-bound, traceable); at 512+ device compute hides it.

## Target chosen: trace-capture the diffusion score-model step, replay across 200 steps
Rationale: small-protein (L<=~300) diffusion is host-dispatch-bound (skill: 38->17ms, 2.2x).
Lossless by construction (identical kernel stream -> PCC~1.0). Expect big win @256, neutral @512/686.
Risk: trace_region_size DRAM reservation -> must verify no OOM @686. Revert if any size regresses/OOMs.

## WIN: trace-capture the diffusion score-model step (ported from origin/perf/trace-pairformer-msa-diffusion)
Env-gated (TT_BIO_TRACE / TT_BIO_TRACE_DIFFUSION); default off => zero behavior change when unset.
- L=256 warm: diffusion 7.55s -> 5.71s (-24%); e2e 17.8s -> 15.8s (-11%).
- LOSSLESS proof: TT_BIO_TRACE_SELFCHECK replays trace + runs same inputs eagerly:
  PCC=1.000000, maxdiff=0.000e+00 (bit-identical). The "unsafe allocation during active trace"
  warning is benign (transient op buffers).
- Fold plddt/ptm differences (~0.004) are DEVICE NONDETERMINISM: two eager runs w/ identical MSA+seed
  differ 0.005 plddt / 0.031 ptm. Fold metrics are NOT a losslessness test (playbook RNG confound).

## Rigorous same-code A/B (median of 3, --fast, shared MSA), L=512
trace OFF (branch, trace disabled): e2e 35.3s  trunk 21.86s  diffusion ~10.3s
trace ON  (branch):                 e2e 32.6s  trunk 18.73s  diffusion ~10.3s (gated off)
=> trunk -14.3%, e2e -7.6%. Diffusion unchanged at 512 (size-gated off). LOSSLESS (PCC=1.0).

## Accuracy / robustness
- Bit-identical proof (byte-identical inputs, TT_BIO_TRACE_SELFCHECK):
  diffusion step PCC=1.000000 maxdiff=0; pairformer stack PCC=1.000000 maxdiff=0.
- Fold-metric variance is DEVICE NONDETERMINISM: two eager runs (same MSA+seed) differ
  0.005 plddt / 0.031 ptm; Ca-RMSD eager-vs-eager = 2.84A, eager-vs-TRACE = 2.66A (no regression).
- Default (non-fast) mode: trace works, plddt 0.862 sane, no OOM.
- Sizes 256/512/686 all run --fast and default, NO OOM with 2GiB trace region.
- Feature env-gated default-OFF => zero behavior change unless TT_BIO_TRACE set.
- Committed: 25f1c0a on exp/perf-20260621-diffusion-resident.

## Trace-ON reps (concurrent, for reference) — medians
L=256: e2e 14.3s (trunk ~6.97, diffusion ~6.0)   L=686: e2e 53.1s (trunk ~33.4)
(Clean non-concurrent A/B for 256/686 in the FINAL SUMMARY below.)

## NOTE on measurement
Multi-card parallel folds contend on HOST CPU (the diffusion host sampling loop),
inflating wall time 1-3s. The L=512 A/B above used SYMMETRIC concurrency (off & on
both running) so its delta is valid. 256/686 final numbers re-measured ALONE.

## ===== FINAL SUMMARY (clean, non-concurrent, warm --fast) =====
| Size | baseline e2e | trace e2e | e2e Δ | trunk Δ | diffusion trace |
|------|-------------|-----------|-------|---------|-----------------|
| L=256 | 17.5s (17.4/17.6) | 13.9s (13.7/14.1) | **-20.6%** | -23% | ON (gated, helps) |
| L=512 | 35.3s (3-rep med) | 32.6s (3-rep med) | **-7.6%** | -14.3% | OFF (gated; compute-bound) |
| L=686 | 61.5s | 51.6s | **-16.1%** | -22% | OFF (gated) |

(L=512 = rigorous 3-rep symmetric A/B; 256/686 = clean alone runs. Concurrent runs
contend on host CPU; all numbers above are contention-free except where noted.)

ACCURACY: bit-identical (per-step PCC=1.000000 maxdiff=0 for diffusion AND pairformer
on byte-identical inputs). Lossless by construction. End-to-end Ca-RMSD trace-vs-eager
(2.66A) <= eager-vs-eager device-nondeterminism floor (2.84A): NO regression.

ROBUSTNESS: 256/512/686 all run --fast AND default(non-fast); NO OOM (2GiB trace region).
Default-OFF => existing behavior byte-unchanged. Untested (handled by construction):
templated/ligand/multimer inputs.

RECOMMENDATION: MERGE as opt-in (set TT_BIO_TRACE=1). Validated lossless, -7.6% to -20.6%
e2e across sizes, no regression anywhere, zero-risk when off. Before flipping default-ON:
validate templated/ligand/multimer folds and confirm 2GiB region safe for largest production
proteins / multi-sample diffusion.

NOTE: main (NOT this branch) intermittently HANGS at "trunk 1/4" on L=686 (livelock,
~13min CPU, ignores SIGINT) — observed on 2 of ~6 main-baseline 686 runs; the trace branch
runs all completed cleanly. Possibly a pre-existing main instability; worth a separate look.
