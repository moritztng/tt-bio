# Protenix-v2 --fast baseline (warm per-sample), tt-quietbox BH, n_step=200
Measured 2026-07-08, single protein (trunk COLD), diffusion samples 1=cold 2,3=WARM.

| L    | trunk(cold) | edm_sample warm/sample | e2e(nsample=3) |
|------|-------------|------------------------|----------------|
| 256  | 5.88s       | 7.49s (8.69 cold)      | 34.1s          |
| 512  | 48.0s(cold) | 8.13s (24.2 cold)      | 141.4s         |

KEY: warm diffusion per-sample ~FLAT in L (7.5s@256, 8.1s@512) despite 2x atoms
=> Protenix diffusion warm is per-step FIXED-OVERHEAD / host-dispatch bound (denoise does
host torch fourier/r_noisy/EDM-precond + a device->host transfer EVERY step), NOT
device-compute-bound like Boltz-2/ESMFold2. Batching B samples into ONE 200-step loop
should amortize that overhead -> near-Bx diffusion throughput. Diffusion dominates e2e at
n_sample>1 (n_sample=3: ~79% @256). LEVER = batch the serial edm_sample loop (denoise B>1).

## HOST-BOUND CONFIRMED (2026-07-08)
warm ms/step nearly L-INDEPENDENT: NT=38 35.2ms, L=256 37.5ms, L=512 40.6ms.
=> per-step fixed-overhead bound (host loop + per-step device->host to_torch), NOT compute.
=> batching B samples into one loop should give ~Bx diffusion throughput. GO.
compute_random_augmentation(multiplicity) already batches. AdaLN/AttentionPairBias broadcast batch (Boltz-2 multiplicity uses them).

## RESULT: single-device sample-batching is a DEAD END (measured 2026-07-08)
L=256 n=3: serial 23.9s (8.69+7.56+7.69) vs BATCHED 31.8s  -> 1.4x SLOWER
L=512 n=3: serial 26.4s (9.63+8.41+8.38) vs BATCHED 42.7s  -> 1.6x SLOWER (worse at large L)
Batch dim B=3 is tile-misaligned; the windowed atom-attn (folds B into B*nb blocks) and the
head_dim=48 DiT do 3x the device work LESS efficiently than 3 sequential B=1-optimal passes.
atxE/atxD batch bit-exactly (B=1 == serial maxdiff 0); DiT has a B>1 buffer-aliasing bug
(block0 clean, block1+ diverge from identical inputs; apb itself proven correct at B=3).
CONCLUSION: reverted. The real Protenix lever is per-step DISPATCH reduction (diffusion warm is
L-independent ~35-41ms/step = dispatch-bound), which helps EVERY fold incl. default n_sample=1:
ttnn TRACE of the denoise device stream, or a device-resident coordinate loop.
Multi-card sample-parallel would give ~linear n_sample>1 throughput but the per-step device<->host
round-trip (r_update to_torch each step) is the same tax that made mesh-diffusion a dead-end (06-24).
