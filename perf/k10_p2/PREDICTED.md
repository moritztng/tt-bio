# Predicted before the CPU reference arms ran

Written 2026-09-14, before `fold_cpu_ref.py` executed once. The question: the CA-lDDT deficit
against upstream survives shared draws (`perf/roof_shared`: -0.00586 at 298 aa, -0.01487 and
-0.01462 on the two 512 aa pseudo-domains). Which of the three layers between the stacks carries
it, our torch code, the bf16 dtype boundary, or ttnn?

What is already ruled out on the static side, before any fold:

* `tt_bio/data/featurizer.py` against `boltz/data/feature/featurizerv2.py` (2352 vs 2354 lines):
  the only difference is the import block. Byte-identical featurization.
* `tt_bio/data/tokenize.py` against `boltz/data/tokenize/boltz2.py`: the only difference is the
  import block plus the `Tokenizer` ABC being inlined rather than imported.
* Diffusion process args. `Boltz2DiffusionParams` upstream is gamma_0 0.8, gamma_min 1.0,
  noise_scale 1.003, rho 7, step_scale 1.5, sigma_min 1e-4, sigma_max 160, sigma_data 16; the
  values in `tt_bio/main.py:3226` are the same, and `--step_scale` defaults to 1.5 on both.
* Steering. Upstream `predict` sets `fk_steering = physical_guidance_update = use_potentials`
  and leaves `contact_guidance_update` at its `True` default; `tt_bio/main.py:3247` does the
  same, so "potentials off" means the same thing on both sides.
* MSA module dropout (0.15 / 0.25 in tt-bio, 0.0 / 0.0 upstream at predict) is inert:
  `get_dropout_mask` multiplies the rate by `training`, which is 0 under `eval()`.

So the featurization and the sampler configuration are not the mechanism. The predictions:

1. **`ttcpufp32` lands very close to `gpurefshared-s0`**: worst per-pseudo-domain all-atom
   0.05 to 0.25 A at 298 aa. Two fp32 torch implementations of one checkpoint on the same noise
   differ only by op ordering and CPU-vs-CUDA fp32 reduction order.
2. **CA-lDDT of `ttcpufp32` against 1HCL is within 0.003 of the reference's 0.98351**, i.e. the
   port-code term of the deficit is under a quarter of the -0.00586 seen at 298 aa.
3. **The deficit therefore lives below the torch layer**, and `ttshared` (the device arm) keeps
   essentially all of it: d(ttshared, ttcpufp32) >= 0.8x d(ttshared, gpurefshared).
4. **`ttcpubf16` reads small**, 0.15 to 0.5 A at 298 aa with CA-lDDT within 0.005 of `ttcpufp32`.
   `torch.autocast` is mixed precision, keeping normalisations and accumulations in fp32, the
   same reason the anchor's GPU `bf16-mixed` control cost only 0.01993 A. If that holds, the
   deficit is specifically ttnn's all-bf16 arithmetic, not "bf16" in the abstract.
5. Confidence in 1 to 3: moderate. The alternative is that our torch model code itself drifted
   from upstream somewhere the featurizer diff cannot see (a layer, a norm placement, a
   checkpoint key mapped differently), in which case `ttcpufp32` misses the reference by a lot
   more than 0.25 A and prediction 1 fails loudly rather than quietly. That is the outcome worth
   having.

---

# Predicted before the 512 aa arms were scored

Written 2026-09-14 17:55Z. Both 512 aa CPU folds were still running on pc, neither had been
scored, and nothing about them was known beyond the 298 aa result they follow. At 298 aa the
split came out: `ttcpufp32` 0.03420 A from `gpurefshared-s0` with CA-lDDT 1.00000 against it, and
-0.00011 of the -0.00586 lDDT deficit, so 1.9 % is our torch code. At 512 aa the deficit is 2.5x
larger, -0.01487 on copy 1 and -0.01462 on copy 2, and the device arm sits 0.68922 A out.

6. **`ttcpufp32` at 512 aa lands 0.05 to 0.20 A from `gpurefshared-s0`** (worst per-pseudo-domain
   all-atom), above the 298 aa 0.03420 A because the trajectory is longer and the chimeric
   fixture's inter-domain hinge amplifies any small displacement. CA-lDDT against upstream
   >= 0.998.
7. **The port-code share of the deficit stays small**: |CA-lDDT(`ttcpufp32`) -
   CA-lDDT(`gpurefshared`)| against 1HCL <= 0.003 on each pseudo-domain, i.e. under 20 % of
   -0.0147. If the 298 aa 1.9 % were an artefact of the smaller size, this is where it breaks.
8. **No third direction at 512 aa either**: d(`ttshared`, `ttcpufp32`) within 15 % of
   d(`ttshared`, `gpurefshared`) = 0.68922 A.
9. **`ttcpubf16` at 512 aa is worse than its 298 aa 1.21511 A**: 1.5 to 3.5 A, CA-lDDT against
   1HCL down by more than 0.15 from `ttcpufp32`. Prediction 4 was already refuted at 298 aa (CPU
   autocast is not the GPU `bf16-mixed` the anchor controlled with), so this is the corrected
   form of it, and the point it carries is the same: the device arm's 0.00586 / 0.0147 is at the
   good end of the bf16 class, not the bad end.
10. Confidence in 6 to 9: high for 8, moderate for 6 and 7, low for the magnitude in 9. The
    outcome that would change the conclusion is 7 failing, i.e. our torch fp32 code carrying most
    of the 512 aa deficit while carrying 1.9 % of the 298 aa one. That would mean a
    size-dependent divergence in our port rather than a precision floor.
