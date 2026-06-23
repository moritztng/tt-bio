# ESMFold2 perf run — 2026-06-23 (branch exp/perf-20260623-esmfold2)

Target: ESMFold2 / esmfold2-fast — NEVER profiled before (journals only cover Boltz-2 + Protenix-v2).
Pipeline: ESMC-6B LM (80 blk) -> resident folding trunk (48/24 blk, reuses Boltz trimul) -> diffusion structure head -> confidence head.
Env fix: dev env built with `pip install -e . --no-deps` lacked declared dep `transformers==4.57.6`; installed --no-deps.

## Warm baseline (2nd protein, same size, --fast, tt-quietbox BH card 0)
### esmfold2-fast (24 trunk blocks), L=256
Cold fold1: prep=56.66 (incl compile+ESMC load; LM 12.38) trunk=7.60 diffusion=21.23 confidence=0.48
WARM fold2: prep=1.49 (LM 0.23)  trunk=2.40  diffusion=4.77  confidence=2.07  => e2e ~10.7s
Shares (warm): diffusion 44% DOMINANT | trunk 22% | confidence 19% | prep 14% (LM only ~2%)
NOTE: LM(ESMC-6B) warm is only 0.23s — NOT the bottleneck. Diffusion + confidence dominate at L=256.

### esmfold2-fast, L=512  (--fast, warm)
WARM fold2: prep=4.50 (LM 0.37)  trunk=9.30  diffusion=6.80  confidence=3.42  => e2e ~24.0s
Shares (warm): trunk 39% | diffusion 28% | confidence 14% | prep 19% (LM ~1.5%)
=> Same shape as Boltz: trunk (O(L^3) trimul) grows with L, diffusion share shrinks.
   prep has ~4s NON-LM work (lm_encoder FoldingTrunk + InputsEmbedder run ONCE, in 'prep').
Config: sampling_steps=200 (CLI default, inherited from Boltz; model's own default is 20),
        diffusion_samples=1. Diffusion = 200 denoise calls + 200 host Kabsch aligns.

## Key facts
- Trunk / lm_encoder / confidence all reuse the SAME ttnn TriangleMultiplication kernel as
  Boltz-2 (at ceiling per prior journals). LM(ESMC-6B) warm is negligible (0.2-0.4s).
- NO trace infra on main (the Boltz trace lives only on exp/perf-20260621-diffusion-resident).
- Resident trunk loop already ported to ESMFold2 (_install_resident_trunk_loop).

## Diffusion bound-type (warm, host vs device, last/warm trajectory only)
- L=256: 134 denoise calls | device_denoise=3.50s  host_kabsch_align=0.07s  host_center_aug=0.07s
- L=512: 134 denoise calls | device_denoise=5.60s  host_kabsch_align=0.10s  host_center_aug=0.10s
=> Diffusion is ~96% DEVICE (host Kabsch+augment only ~3-4%). 26ms/call @256, 42ms/call @512.
   The reverse-diffusion sampler runs the score model over ALL ATOMS (~1984 @ L=256): an
   atom encoder + 24 token-DiT blocks + atom decoder. Atom-dense => compute-bound even at L=256
   (unlike Boltz-2's token-level diffusion, which is host-dispatch-bound at small L).
- sampling_steps=200 is the REFERENCE default (esm processor.fold + config.structure_head.
  inference_num_steps), NOT a Boltz leak — reducing it changes output, so it is not a lossless lever.

## Experiments tried (both LOSSLESS by construction, both DEAD ENDS — reverted)
### 1. ttnn trace capture/replay of the FoldingTrunk  [TT_BIO_TRACE_FOLDTRUNK]
   Shape-keyed capture of FoldingTrunkModel (24/48 PairUpdateBlocks); reused by trunk loop
   (num_loops+1), lm_encoder, confidence. Self-check: PCC=1.000000 maxdiff=0 (bit-identical).
   Warm trunk: L=256 2.40->2.38s (0%); L=512 9.30->9.32s (0%). NET 0 / slight regress.
   WHY: ESMFold2's PairUpdateBlock is LEAN — 2 triangle-mults + 1 transition, NO triangle
   ATTENTION (Boltz-2's pairformer block has 5 ops incl. 2 tri-attentions). Far fewer op
   launches => host dispatch already negligible => trace reclaims nothing. The ~3.9x (not 8x)
   trunk scaling 256->512 is O(L^2) device work (layer norms / transitions / permutes), not a
   dispatch floor. 1.5 GiB trace region also regressed diffusion ~+1.0s (DRAM pressure next to 6B).

### 2. ttnn trace capture/replay of the diffusion step  [TT_BIO_TRACE_DIFFUSION]
   Captured DiffusionModuleModel.step (cond_single + atom enc + 24 DiT + atom dec) once per
   trajectory (recaptured each fold; ~132/134 steps replay). Self-check: PCC=1.000000 maxdiff=0.
   Warm diffusion @L=256: 4.77 -> 5.14s (+8% REGRESS). NET regress.
   WHY: the score model is compute-bound at L>=256 (atom-dense, ~96% device), so replay saves
   little dispatch, while the 768 MiB trace region degrades the denoise memory configs. (Boltz-2's
   diffusion trace only won for small proteins because it is host-dispatch-bound there; ESMFold2's
   atom-level score model is not.)

## VERDICT
No >=5% lossless win found. ESMFold2's neural cost is dominated by (a) the SAME ttnn
TriangleMultiplication kernel as Boltz-2 (trunk + lm_encoder + confidence) — already at its
compute-bound ceiling per prior journals — and (b) an atom-dense diffusion score model that is
also compute-bound at all representative sizes. The ESMC-6B LM frontend is negligible warm
(0.2-0.4s). ttnn trace (which only hides host op-dispatch) reclaims nothing because ESMFold2 has
little dispatch to hide. Remaining headroom is algorithmic / multi-device (same conclusion as
Boltz-2), not per-op knobs or trace.

## Branch / robustness
Branch exp/perf-20260623-esmfold2 — code reverted to main (no diff); only this report added.
All experiments validated bit-identical (per-step PCC=1.0, maxdiff=0) before being measured and
reverted. Env: installed declared dep transformers==4.57.6 (--no-deps) into ~/tt-bio-dev/env,
which `pip install -e . --no-deps` had skipped (needed to run ESMFold2 at all).
RECOMMENDATION: nothing to merge. Record the dead-ends so they are not retried.
