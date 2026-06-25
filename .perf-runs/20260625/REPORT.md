# Perf run 2026-06-25 — Harden the ttnn-trace win to merge-ready (full-input validation)

Branch: exp/perf-20260625-attnpairbias (off main 8a40c42)
Engineer: autonomous overnight run.

## Target & rationale
Boltz-2 --fast per-op kernels are at ceiling (see LEARNINGS: trimul subblock, SDPA chunk,
exp_approx, bf8 weights, HiFi2 all dead ends; AttentionPairBias 18.3% is genuine per-layer
compute; diffusion_samples=1 default so no sample-batching fruit; diffusion token-transformer
pair bias already precomputed). The single best UNMERGED win is the ttnn TRACE capture/replay
of Pairformer+MSA+Diffusion (branch exp/perf-20260621-diffusion-resident): -7.6% e2e @512,
lossless by construction. BUT it was only ever validated at ONE point (L=512, --fast).
Moritz's gate is "works for ALL inputs." Tonight: validate across the FULL matrix
(L=256/512/686 x --fast/default x no-OOM x bit-identical) and make a merge recommendation.

Trace impl: master switch TT_BIO_TRACE (+ per-module TT_BIO_TRACE_{PF,MSA,DIFFUSION}),
diffusion-trace auto-gated off above seq_len 384 (TT_BIO_TRACE_DIFFUSION_MAX_SEQ), default OFF.

## Measurements (warm, A = trace OFF, B = trace ON; same code, env-gated)
(filled in incrementally below)

