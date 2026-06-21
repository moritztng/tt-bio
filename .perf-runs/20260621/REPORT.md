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
