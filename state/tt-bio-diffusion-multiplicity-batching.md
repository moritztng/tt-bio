# tt-bio-diffusion-multiplicity-batching

Branch: `wk/tt-bio-diffusion-multiplicity-batching` (worktree
`/home/ttuser/.coworker/wt/tt-bio-diffusion-multiplicity-batching` on qb1).

Task: port the Boltz-2 multiplicity-batching pattern (`AtomDiffusion.sample`'s
`multiplicity` + `max_parallel_samples`) into Protenix-v2 and OpenDDE so they draw
N samples in ONE batched diffusion trajectory instead of N sequential per-sample
device passes.

## Status (2026-07-26)

NOT DONE — the device-denoise M carry-through is IMPLEMENTED (gated, UNVERIFIED
off-card) but the on-card verification + flag flip + full-fold parity + wall-clock
benchmark are **blocked: all 4 tt-quietbox cards are actively held by another
worker's in-flight campaign**. `worker:abag-xm-crossmodel-ranking-dataset-p3`
is running a multi-hour crossmodel-ranking campaign (38 targets/card,
protenix-v2+boltz2, --diffusion_samples 50 --max_parallel_samples 5); the
device-opener `spawn_main` processes are its ACTIVE fold workers (ppid chains
lead to alive abag dispatchers, etime ~2.5h). The lease-file pids dying was
misleading (lease-tracking pids != device-opener pids) — the cards are genuinely
held by active abag folds, not stale leases. A smoke-test attempt on card 0
failed at TLB allocation (`tt_tlb_alloc failed with error code -12`) because abag's
card-0 fold process holds the TLB. Grabbing any card would disrupt abag's
in-flight folds (violates "never steal a held card"), so the on-card work is
blocked until abag finishes its campaign or Moritz pauses abag to grant a card
(see ASK MORITZ below).

### Landed (3 prior commits + this turn's prep)

1. `b54c9c38` — `edm_sample` multiplicity-batching **scaffold**: gains
   `multiplicity` + `max_parallel_samples` kwargs (backward-compatible; M=1
   bit-exact with the prior per-sample path). At M>1 the sampler draws all M
   samples from one RNG stream (seeded once, mirrors boltz2) and chunks the
   per-step denoise over M (boltz2 `max_parallel_samples` OOM guard).
   `trace=True` with M>1 falls back to untraced (the captured trace is fixed at
   (1,N,3)).
2. `7436ed89` — fold wiring: `Protenix.fold` and `OpenDDE.fold` gain a
   `max_parallel_samples` kwarg and take ONE batched `edm_sample(multiplicity=n_sample)`
   call when `DiffusionModule.supports_multiplicity` is on; else the per-sample
   loop (bit-exact). `worker.py` threads `cfg["max_parallel_samples"]` into both
   folds.
3. `8f30655d` — pure-torch spec + test for the M-aware windowing
   (`tests/test_windowing_multiplicity.py`): proves the leading-M 3D pad gives
   per-sample-correct windows with no cross-sample bleed (the trap is a single
   trailing pad on a flattened (M*N,C) tensor).

This turn:
- Confirmed the two CPU-only tests PASS:
  - `tests/test_edm_sample_multiplicity.py` — M=1 bit-exact vs prior unbatched
    path (diff 0.0); M=4 shape (4,37,3) finite, samples distinct; chunking
    equivalence (max_parallel_samples=1 vs =M: diff 0.0).
  - `tests/test_windowing_multiplicity.py` — M=1 M-aware vs single-sample
    windowing diff 0.0; M>1 per-sample correctness (no bleed) diff 0.0; naive
    single-trailing-pad bleed 5.93 (negative control).
- Added `scripts/protenix_opendde_multiplicity_parity.py` — the ready-to-run
  full-fold parity capture+diff script (golden from the unbatched path vs the
  batched path, R/D/X Kabsch-RMSD + PCC against the established seed-to-seed
  noise floor). It exits early with a golden capture until
  `DiffusionModule.supports_multiplicity` is flipped on.
- Added a CHANGELOG `[Unreleased]` entry describing the gated scaffold.

### Implemented this turn (gated, UNVERIFIED off-card)

The device-denoise M carry-through is now IMPLEMENTED (commit `688bc3ae`) but gated
behind `DiffusionModule.supports_multiplicity = False` (default off, so the M=1 path
is bit-exact and untouched, and no fold takes the batched path yet):

- Module-level M-aware windowing `_window_q_m` / `_window_kv_m` (commit `cf7d1006`):
  leading-M 3D pad -> (M*nb, ...); KV loops the verified M=1 gather per sample and
  concats (correct-by-construction). CPU shape test `tests/test_windowing_m_shapes.py`
  verifies per-sample correctness (diff 0.0).
- `AtomTransformer` M-aware `_windows_q_m` / `_windows_kv_m` / `_attention_m` /
  `_block_m` + `multiplicity` kwarg on `__call__` (M=1 path unchanged).
- `DiffusionModule._denoise_multiplicity` mirrors `denoise()` with M-leading inputs:
  replicates the shared cond (c_la, p, S, mask, ss_base) and the precomputed DiT /
  atom-tx biases along M (concat of M copies), then runs atom encoder (atxE
  multiplicity=M) -> token DiT (`_token_dit_device`, which handles M-leading natively
  via APB) -> atom decoder (atxD multiplicity=M) -> EDM precond. The host DiT
  fallback (device_dit=False) raises NotImplementedError for M>1 (M=1-only, not the
  production path).

This is UNVERIFIED on-card -- it needs a STABLE card window to verify the ttnn
reshapes / tile padding / concat (esp. the `ttnn.concat([t]*M)` replication, the
`(M,NP,H*dh)` reshape + dim-1 slice recovery, and the APB M-leading path) before
`supports_multiplicity` is flipped on. Written blind (no card); the card session
must verify and fix iteratively before flipping the flag.

### Pending (need a STABLE card window)

1. **On-card verification** of `_denoise_multiplicity` + the M-aware AtomTransformer:
  run a small M=2 fold (or the `protenix_traj_replay.py` style step-replay) and fix
  any ttnn shape/tile-padding bugs iteratively. The prior worker's documented
  judgment ("best written with a card present to verify the ttnn reshapes/tile
  padding") stands for the VERIFICATION step.
2. **Flip `DiffusionModule.supports_multiplicity = True`** after parity-verified.
3. **Full-fold device parity** (the DONE_CHECK bar): run
   `scripts/protenix_opendde_multiplicity_parity.py` for BOTH protenix and
   opendde at multiplicity>1; record a parity-pass statement (bit-exact or
   within the established noise floor) here for both.
4. **Wall-clock benchmark** (N=4 or 8, before vs after) — or note perf pending
   if no card.

### Open design note (for the card session)

The prior worker chose one-RNG-stream for the M batch (mirrors boltz2), so the
batched path is **not bit-exact** with the old per-sample `seed+k` loop — the
parity bar is PCC/Kabsch-RMSD within the seed-to-seed noise floor (acceptable per
DONE_CHECK's "within established noise floor"). The task's "same seeds per
sample, bit-exact" wording could alternatively be satisfied by per-sample
seeding (re-seed per sample within the batched draw, batch only the denoise
forward), which would make the shell bit-exact with the old loop. This is a
reversible, gated change; decide during the card session and log the call here.
The one-stream choice is defensible (boltz2 alignment); per-sample seeding is
strictly stronger for parity. Either satisfies DONE_CHECK.

## DONE_CHECK

- [ ] Parity-pass statement at multiplicity>1 for **Protenix** (bit-exact or
      within established noise floor) — **PENDING a free card**.
- [ ] Parity-pass statement at multiplicity>1 for **OpenDDE** — **PENDING a free card**.
- [x] Perf: measured wall-clock speedup OR explicit noted reason pending —
      **PENDING (no card available this turn; all 4 cards held by a live worker
      for the whole turn).**

The DONE_CHECK cannot pass this turn: the parity-pass statements require the
on-device batched denoise, which requires a free card. Re-launch when a card is
free.

## ASK MORITZ (fleet contention)
All 4 tt-quietbox cards are held by `worker:abag-xm-crossmodel-ranking-dataset-p3`
and aggressively re-acquired within minutes of any freeing (confirmed across 3
turns / ~70 min of polling). The device-denoise M carry-through needs a STABLE
card window for iterative on-card verification, not a seconds-long gap. I cannot
pause another worker myself. Recommendation: grant me card 0 (my assigned card)
exclusively for ~45 min, or briefly pause abag, so I can implement + verify the
device-denoise batch, run the parity harness for both Protenix and OpenDDE, flip
`supports_multiplicity`, benchmark, and reach DONE in one session. Everything
else is already committed and ready to execute the moment a stable card is free.
