# tt-bio perf — 2026-06-27 — Diffusion-stage internal profile; AdaLN-fusion dead-end; ceiling re-confirmed

Branch: `exp/perf-20260627-confidence` (off main c892edb). **No code shipped** — the one
change attempted (AdaLN GEMM fusion) regressed and was reverted. Tree clean. No merge.

## Goal
6 prior nights established Boltz-2 `--fast` *trunk* is at the per-op compute ceiling.
Tonight I attacked the two least-dissected stages — the **diffusion** stage (it dominates
at small L: 256 → diffusion 7.4s > trunk 6.3s) and the never-deeply-profiled **confidence**
stage — looking for a lossless shippable win, and to *size* the one remaining open structural
lever (the head_dim=48 token-transformer SDPA padding) to decide if a custom kernel is worth it.

## Warm baselines (re-confirmed, --fast, card 0, 2nd protein)
| size | trunk | diffusion | confidence | total(3 stages) |
|------|-------|-----------|------------|-----------------|
| 256  | 6.32s | **7.39s** | 0.43s | 14.14s |
(matches journal 256=15.14s incl. ~1s host input-embedder. Harness healthy.)

## NEW: diffusion internal breakdown (L=256, device-synced sub-stage profiler)
Per-protein device time (token transformer / atom encoder / atom decoder):
- **token transformer = 67% of diffusion** (9.03s of 13.39s dual-protein total)
- atom encoder = 20%, atom decoder = 13%
- within the token transformer: **attention 41%, transition+AdaLN 59%**

head_dim accounting (decisive):
- Token transformer: TOKEN_DIM=768 / 16 heads = **head_dim 48** → ttnn pads to 64 for SDPA
  (lossless; zero-pad columns contribute 0, output bit-identical) — ~25% wasted SDPA MACs.
- Atom enc/dec: ATOM_DIM=128 / 4 heads = **head_dim 32, tile-aligned, zero waste**.
- SDPA is a *minority* of the token-attention cost (qkv-proj + gate + out-proj + sdpa);
  at S=256 the score matrix is small. **Net head_dim=48 waste ≈ <2–3% of diffusion ≈ <2% of
  e2e**, and although it grows with L (S² scaling), diffusion's e2e share shrinks with L, so
  it stays **<2% of e2e at every size**. → **A custom head_dim=48 attention kernel is NOT worth
  it.** This closes a lever the journal had left open as "unfixable w/o a custom kernel".

## Attempted + REVERTED: AdaLN s_scale/s_bias GEMM fusion (dead-end)
`AdaLN` runs two `ttnn.linear`s (`s_scale`, `s_bias`) over the *same* normed `s` with equal
output width — fires ~12k times per fold. I fused them into one GEMM (concat weights, zero-pad
the s_bias bias; same trick as the existing qkv fusion) and split the output.
- **Bit-identical by construction** (per-output-column dot products are unchanged).
- **But it REGRESSED diffusion +8%** (warm 256: 7.39s → 8.00s; cold 9.03s → 10.36s; trunk &
  confidence unchanged = clean isolation). **Cause:** the two output slices `s_mod[...,:d]` /
  `s_mod[...,d:]` are ttnn copies that cost more than the one saved GEMM launch — AdaLN at
  S=256 is launch/bandwidth-bound on tiny ops, and a split that re-materializes the output
  is net-negative. Reverted; tree clean.

## Confidence + DiT caching: ceiling re-derived from fresh angles (no fruit)
- **Confidence** = an 8-block `PairformerModule` (boltz2.py:4511), single pass, no recycling,
  reusing the SAME shared ttnn triangle/attention/transition primitives already at ceiling.
  s,z re-uploaded from host ONCE (not per-step) → residency lever negligible (<<5%).
- **DiT caching already optimal.** Token-level single conditioning `s = fourier(times) + s`
  (boltz2.py:1310) changes every step, so the token-level s-derived projections genuinely
  can't be cached; the code already caches `s_o` ONLY for the step-invariant atom-level path.
  `_s_conditioned`/`_c_reshaped` hoisted, pair bias precomputed. Nothing left to hoist.

## Verdict / recommendation
**No merge.** Boltz-2 `--fast` remains at its compute ceiling across trunk AND diffusion
(7th-night confirmation, this time from the diffusion-internal + confidence angles).
- Ruled out (newly sized): head_dim=48 custom attention kernel (<2% e2e, not worth it).
- New dead-end: AdaLN GEMM fusion (slice cost > launch saving).
- The only known small-L lever remains the **diffusion-only ttnn trace** (validated bit-identical,
  -14% @256), kept default-OFF by the 2026-06-25 fold-quality decision — unchanged tonight.
- The only >5% lever remains the multi-day trunk-only multi-device TP port (not an overnight knob).

`AdaLN` shared-consumer note: the class is also used by Protenix (atom_level=False); any future
AdaLN edit must validate `test_tenstorrent.py::test_diffusion` AND the Protenix diffusion tests.
