# 2026-07-08 — Protenix-v2 --fast: diffusion dispatch-bound; LANDED lossless denoise-trace win

Branch `exp/bioperf-largeseq`. NOT pushed, main untouched.

## Profiling (new): Protenix diffusion is per-step DISPATCH-bound
Warm per-sample `edm_sample` (200 steps) is ~L-independent:
NT=38 35.2ms/step, L=256 37.5ms, L=512 40.6ms. ~400 device op launches/step x ~85us
~= 34ms, size-independent => dispatch-bound, NOT device-compute-bound (unlike
Boltz-2/ESMFold2). Diffusion dominates e2e at n_sample>1 (trunk once; diffusion+confidence
per sample). L=256 trunk 5.88s, warm diffusion 7.49s/sample.

## Attempt 1: single-device sample-dim batching (mission lever #2) -> DEAD END, reverted
denoise batched over B>1 (fold sample dim into windowed atom-attn block dim; batched DiT).
atxE/atxD bit-exact at B=1. MEASURED SLOWER: L256 n=3 serial 23.9s -> batched 31.8s (1.4x);
L512 26.4 -> 42.7s (1.6x, worse at large L). B=3 is tile-misaligned; wide windowed-attn +
head_dim=48 DiT do 3x device work less efficiently than 3 B=1-optimal passes. Reverted.

## Attempt 2: ttnn TRACE of the denoise device stream -> LANDED (lossless)
Two commits (b0c6450 profiling+dead-end doc; 1147a82 the win):
1. AtomTransformer._adaln now CACHES the AdaLN module per prefix (was reconstructing it +
   re-uploading 4 weights via ttnn.from_torch every call = ~9600 redundant device writes/fold
   in the diffusion enc/decoder). Bit-identical; always on; prerequisite for trace capture.
2. fold(trace=True): capture the ~400-op denoise device stream once/fold, replay it,
   collapsing per-step dispatch. Per-step host inputs (fourier(t_hat), scaled coords) staged
   into fixed device buffers; fold-fixed cond stays resident. Open the device with a trace
   region first: get_device(trace_region_size=1<<30) (default 0 => layout unchanged).

### Warm diffusion (edm_sample, 200 steps, --fast) — CLEAN no-contention, card 0 alone
| L    | untraced (AdaLN cache) | trace  | trace speedup |
|------|------------------------|--------|---------------|
| 256  | 6.134s                 | 4.776s | **-22.1%**    |
| 512  | 7.746s                 | 7.728s | -0.2% (neutral) |
e2e @L256 n_sample=1 ~ trunk 5.88 + diff + conf 0.4: 12.4s -> 11.06s (~-11% e2e), and the
win grows with n_sample (diffusion share grows: n=5 e2e ~ -19%).

### Accuracy — LOSSLESS (proven)
trace vs untraced, SAME seed, golden (N=275, 200 steps): coord maxdiff=0.000e+00,
Kabsch RMSD=0.0000 A; untraced-vs-ref 7.1034 A == trace-vs-ref 7.1034 A. The trace replays
the identical kernel stream and Protenix denoise is deterministic (serial x2 maxdiff 0).
Tests: test_protenix_{atomtx,diffusion,ife,fold,diffusion_cond,atomfeat} 8/8 pass.

### Scope / ceiling
Trace win is largest where dispatch > compute (small-mid L, e.g. L256 diffusion compute floor
~26.7ms/step) and shrinks as L grows (L512 compute ~38ms/step ~= dispatch => neutral). So it
helps small-mid proteins and the default n_sample=1 path; large-L (512/1024) diffusion is
compute-bound (same ceiling as Boltz-2). AdaLN caching is a small bit-identical win at all L.
