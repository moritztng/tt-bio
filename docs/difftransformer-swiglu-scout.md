# Kernel scout: fused SwiGLU for the diffusion transition block (no-go)

## Result

The diffusion transformer's `ConditionedTransitionBlock` (`tt_bio/tenstorrent.py`, shared by
Boltz-2 and BoltzGen; Protenix-v2 has its own) is a **token-width** SwiGLU block — the same
"linear in sequence length" shape as ESMC's per-token FFN, not the quadratic pair transition
that won 1.14-1.18x in ESMFold2 (`docs/kernel-scout-next.md`). This is the ESMC case
(`docs/esmc-swiglu-fusion-scout.md`), not the ESMFold2 case.

Two independent reasons make this a no-go:

1. The `fuse_swiglu` epilogue kernel **is not present in the current fleet ttnn build**. It is
   ttnn-build-gated and this build predates it, so the fusion cannot be enabled or measured on
   qb2 and would be inert even if wired in.
2. The one buildable piece of the fusion — merging the block's two `swish`/`gates` matmuls into a
   single packed matmul — is a genuine but **marginal ~1.6%** on the diffusion denoiser, below
   the ESMFold2 fusion's bar, and it introduced a run-to-run instability spike the baseline never
   showed.

Runtime code is unchanged; the profiling and A/B instrumentation was reverted after measuring.

## Activation width: ESMC-like, not pair-transition-like

`ConditionedTransitionBlock` is SwiGLU (`swish_gate` -> `silu(gates) * swish`) plus an extra
`a_to_b` gate. Its widest intermediate is the `swish_gate` output:

| Path | dim | dim_inner | swish_gate out | intermediate `[bm, n, .]` |
|---|---:|---:|---:|---|
| token transformer (24 layers) | 768 | 1536 | 3072 | `[bm, n_tok, 3072]` |
| atom encoder/decoder (3+3)     | 128 |  256 |  512 | `[bm, n_atom, 512]` |

Both are **linear in sequence length**, exactly like ESMC's FFN (`[B,L,d]`) and unlike the pair
transition's quadratic `[B,L,L,c]`. At prot.yaml (117 residues) the token intermediate is
~1.5 MiB — well below the ESMC-6B `[1,2048,10240]` (~42 MiB) tensor that *still* showed no win.
The fusion's benefit (eliding the wide intermediate's DRAM round trip) scales with that tensor,
so the a-priori was a no-win, and the measurements agree.

## Share in a real warm diffusion loop

`pc`-style methodology on qb2 (physical card 2, one Blackhole P150a), real Boltz-2 checkpoint,
`examples/prot.yaml` (117 residues), `--single_sequence`, 200 sampling steps, seed 0. Steps
2-200 run warm off the program cache.

Clean per-step diffusion device wall (timed around the denoiser `score_model` call, whose
`to_torch` readback is the only synchronization — no per-op sync distortion), 200 steps:

| Run | score_model total | per step |
|---|---:|---:|
| baseline | 3.348 / 3.371 / 3.355 / 3.372 s | ~16.8 ms |

The transition block's share of the two diffusion transformers, from synchronized per-call
accounting (an **upper bound** — the per-call syncs inflate the transition relative to its true
compute): **25.3%** of the token transformer, **21.6%** of the atom transformers. Call counts
per fold confirm the wiring: 4800 token-transition calls (200 steps x 24 layers), 1200
atom-transition (200 x (3+3)). The 24-layer token transformer is the diffusion bulk, and the
`swish_gate` matmuls the fusion targets are only part of the transition's ~25%.

## The fusion is not in the fleet build

`SwiGLUFFN` guards the fusion with `"fuse_swiglu" in minimal_matmul.__doc__`. On the qb2 build
that string is absent, and calling `ttnn.experimental.minimal_matmul(..., fuse_swiglu=True)`
raises `TypeError: incompatible function arguments` — the op exposes only
`bias_tensor / fused_activation / config / memory_config / dtype / compute_kernel_config`. So the
exact epilogue fusion cannot be evaluated here, and the merged ESMFold2 call site is correctly
inert on this build.

## Buildable proxy A/B: packed swish/gates matmul

The available part of the fusion is packing the block's two separate `swish` and `gates` linears
(`[dim, dim_inner]` each) into one `[dim, 2*dim_inner]` matmul, then `chunk` + `silu`-gate (the
`SwiGLUFFN` unfused path). This removes one matmul dispatch per call but, unlike the real fused
kernel, still materializes the wide `[.,2*dim_inner]` intermediate and adds a `chunk`.

| Run | score_model total (200 steps) |
|---|---:|
| baseline (4 runs) | 3.348 / 3.371 / 3.355 / 3.372 s -> mean **3.361 s** |
| packed  (4 runs) | **3.998** / 3.304 / 3.300 / 3.313 s -> warm mean **3.306 s** |

The three warm packed runs are a consistent **~1.6%** below the baseline mean, but the first
packed run spiked to 3.998 s (+19%), a swing the tight baseline never produced — the wide packed
intermediate occasionally hits a slower allocation/memory path.

Accuracy: TT diffusion is not bit-deterministic at fixed seed, so structures are compared against
the baseline's own run-to-run floor. Same-seed Kabsch RMSD base-vs-base = 2.21 / 2.41 Å;
base-vs-packed = 2.03 Å — **within the baseline noise floor**, so the packed matmul does not
change accuracy. (The packed path is the same math in the same bf16; only the matmul tiling over
`2*dim_inner` vs `dim_inner` columns differs.)

## Verdict

No-go. The block is the ESMC (linear-width) case, and the ESMFold2 `fuse_swiglu` kernel that won
there is absent from the fleet build. The only buildable proxy nets ~1.6% on the diffusion
denoiser (~1% on a warm BoltzGen batch, where diffusion is ~63% of runtime per
`docs/permodel-kernel-scout.md`) with a startup instability spike — below the ESMFold2 fusion's
14-18% bar and not worth the risk. Nothing landed. If a future ttnn build ships `fuse_swiglu`,
the real epilogue fusion (which elides the wide intermediate the packed proxy still writes) could
be re-measured here, but the small intermediate size makes an ESMC-style flat result the
expected outcome.

## Reproduce

```bash
# baseline vs packed swish/gates matmul, clean per-step diffusion timing
WT=<worktree>; PY=~/tt-bio-dev/env/bin/python3
for tag in base pack; do
  [ $tag = pack ] && P=TT_CTB_PACK=1 || P=
  env TT_VISIBLE_DEVICES=2 $P TT_DIFFTIME=/tmp/dt_$tag.json PYTHONPATH=$WT $PY \
    -m tt_bio.main predict $WT/examples/prot.yaml --out_dir /tmp/o_$tag \
    --model boltz2 --num_devices 1 --single_sequence --sampling_steps 200 --seed 0
  cat /tmp/dt_$tag.json   # {"score_model_s": ..., "steps": 200}
done
```

(`TT_DIFFTIME` and `TT_CTB_PACK` were temporary env-guarded scout instrumentation, reverted with
this doc.)
