# The rented-GPU OpenFold3 reference folds

Ten folds, `cdk2x2_298` and `cdk2x2_512` at seeds 0-4, official openfold3 0.4.4 on a rented
H200 (vast.ai instance 47558256, offer 39605001, $3.945/hr, destroyed 2026-08-12T15:52Z).
Config is the TT campaign's config: 3 recycles, 200 sampling steps, 1 diffusion sample,
templates off, MSA `perf/size512/fixtures/cdk2x2_<size>.a3m` (35 rows), checkpoint
`of3-p2-155k.pt`.

`ref_<size>_seed<n>.json` carries the provenance of its CIF: package versions, GPU, checkpoint
and a3m hashes, the openfold3 argv, and the cuEquivariance kernel counters. All ten folds ran
the cueq fast path (`triangle.triangle_attention` 700 calls each, i.e. per single fold) and all
ten passed `gpu5_accuracy_gate.py` (`gate_<size>_seed<n>.txt`).

One process per fold, so every structure is a fresh seed-anchored first draw. This is not how
the retained `512_refH200.cif` / `512_refB200.cif` in `../cifs/` were produced -- those came
from a 4-folds-per-process timing run (`n_timed_calls: 4`, predictions fold0-fold3 all under
`seed_0`), where folds 1-3 continue the RNG stream rather than restarting it. See section 4 of
`~/.coworker/state/openfold3-fused-sdpa-gpu-reference-check.md`.
