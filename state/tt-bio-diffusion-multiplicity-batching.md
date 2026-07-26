# tt-bio-diffusion-multiplicity-batching

Branch: `wk/tt-bio-diffusion-multiplicity-batching` (worktree
`/home/ttuser/.coworker/wt/tt-bio-diffusion-multiplicity-batching` on qb1).

Task: port the Boltz-2 multiplicity-batching pattern (`AtomDiffusion.sample`'s
`multiplicity` + `max_parallel_samples`) into Protenix-v2 and OpenDDE so they draw
N samples in ONE batched diffusion trajectory instead of N sequential per-sample
device passes.

## Status (2026-07-26)

NOT DONE — the device-denoise M carry-through, full-fold device parity, and the
wall-clock benchmark are **pending a STABLE card window**. Across 3 turns of polling
(~70 min total) all 4 tt-quietbox cards were held by another live worker
(`worker:abag-xm-crossmodel-ranking-dataset-p3`); that worker aggressively
re-acquires any card the moment its process dies (observed: card 2 freed at 18:31
and was re-grabbed within seconds; card 3 freed at 18:42 and was re-grabbed within
~7 min). So there is no stable window for the iterative on-card verification the
device-denoise M carry-through needs (per the prior worker's documented judgment:
"best written with a card present to verify the ttnn reshapes/tile padding"). The
on-device work could not run.

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

### Pending (need a free card)

1. **`DiffusionModule.denoise` M carry-through**: the device denoise hardcodes
   `(1,N,3)` / `(N,...)` / `(1,NT,768)` shapes throughout (atom encoder, DiT,
   atom decoder, windowing `_window_q`/`_window_kv`/`_windows_q`/`_windows_kv`,
   cond cache `c_la_dev`/`p_dev`/`Smean_dev`/`S_dev`/`atxE_bias`/`atxD_bias`/
   `dit_block_biases`). Carry M as the leading dim via the leading-M 3D pad
   proven in `tests/test_windowing_multiplicity.py`; `repeat_interleave(M,0)`
   the sample-invariant cond tensors; one batched forward per step. The prior
   worker's documented judgment: "best written with a card present to verify the
   ttnn reshapes/tile padding" — respected; not implemented blind this turn.
2. **Flip `DiffusionModule.supports_multiplicity = True`** after the device
   denoise is parity-verified.
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
