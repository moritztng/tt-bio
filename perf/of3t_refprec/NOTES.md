# of3t-refprec — concluded 2026-09-20

## The question, and the answer

Every gradient figure in this campaign is measured against a float64 reference and nobody trains
in float64. This row measured the precision floor of the training recipe itself: upstream
OpenFold3 0.4.3, its own weights, its own frozen batch, its own draws, run in single precision
and in bf16 autocast, scored against `grads_f64_043.pt`.

**Upstream's own fp32 reads mass-weighted 8.158418e-05 over all 4,170 tensors, 245x inside the
2.0e-02 bar.** The bar is not measuring single precision against float64, so the device arm's
7.5692 is the device's own error. On the one named tensor
`diffusion_module.diffusion_transformer.blocks.8.attention_pair_bias.layer_norm_a.layer_norm_s.weight`
(8.05416 % of the model), fp32 reads 2.929034e-05 where the device reads 18.504.

The full write-up with every figure is `~/.coworker/state/of3t-refprec.md`. The numbers live in
`REFPREC.json` (all sets, all arms) and `sidecar/per_tensor_<arm>.json` (one row per parameter
with `rel_l2`, `r` and `cos`, so any of it can be re-scored under a later denominator).
`REFPREC_REPORT.txt` is that file rendered.

## The arms, and where they are

`/home/ttuser/of3t_refprec/run/<arm>/`, built by `run_arm.sh <name> <dtype> <autocast> [draws]`.

| arm | dtype | `--autocast` | reads vs f64 |
|---|---|---|---|
| `arm2_f32_upstream` | float32 | `upstream` | 8.158418e-05 |
| `arm2b_f32_upstream_omp7` | float32 | `upstream` | 8.107441e-05 |
| `arm3_f32_removed` | float32 | `removed` | 8.158418e-05, byte-identical to arm 2 |
| `arm4_bf16_autocast` | fp32 params | `bf16` | 1.105201e-01 |
| `negctl_f32_permuted` | float32 | `upstream` | 3.639337e-01, permuted draws |

The float64 reference was not rebuilt. `bundle_ref/` is hardlinked from `bundle_min_043` and
re-hashed here; all four inputs match that bundle's MANIFEST.

## Three things worth not rediscovering

**Arms 2 and 3 are byte-identical and that is expected.** `AUTOCAST_SITE_CENSUS.json`: 13
non-test `torch.amp.autocast` context sites in the 0.4.3 tree, every one naming
`device_type="cuda"`, so torch enables none of them on CPU (1,644 entered, 0 enabled). At float32
`Tensor.float()` is identity, so `no_autocast` has nothing to remove. Arm 3 is an arithmetic
control. The arm-2-minus-arm-3 separation is observable only on CUDA. An earlier pass put this
site count at nine; the static census says 13.

**The bf16 arm is an upper bound, not the bf16 floor.** Of those 13 sites, 4 are `enabled=False`
and map faithfully to CPU, but 7 ask for `dtype=torch.float32` and CPU autocast has no float32
target, so they lose their upcast protection in this arm. Those 7 include
`core/loss/diffusion.py:78` and `core/utils/geometry/kabsch_alignment.py:64`. Whether upstream's
true bf16 floor clears 2.0e-02 is still open; settling it needs a CUDA box.

**The gradient digest is a function of `OMP_NUM_THREADS`; the floor is not.** Arms 2 and 3 failed
to reproduce their pre-relaunch digests only because their first launch ran at 7 threads and the
relaunch at 3. Re-running arm 2 at 7 reproduced `09f1217c…` exactly, 70 minutes and one code fix
later. The same arm at 3 vs 7 differs by mass-weighted 1.733589e-05 with 175 of 4,135 tensors
bit-identical (`THREAD_COUNT_FLOOR.json`), while the headline moves 0.6 %. Pin the thread count
before treating a byte-identity determinism control as valid.

## Matched scope matters here

The device arm does not cover a whole section, so a section-level comparison against it is a
scope mismatch. `compare_precision.py` defines the sets the device arm actually measured and
ASSERTS their mass against the published figure:

| set | n | % model | fp32 | bf16 | device |
|---|---|---|---|---|---|
| pairformer, the seven measured blocks (0, 8, 16, 23, 32, 40, 47) | 399 | 1.6440 | 3.417154e-05 | 2.942526e-01 | 0.14053 |
| pairformer block 47 alone | 57 | 1.0662 | 3.103001e-05 | 2.307760e-01 | 0.20129 |

On that set the device reads inside a bf16 step. Worth knowing before it is treated as a device
defect, subject to the upper-bound caveat above.

## Reproducing the comparison

    cd /home/ttuser/of3t_refprec
    OMP_NUM_THREADS=3 nice -n 15 /home/ttuser/tt-bio-dev/env/bin/python \
      /home/ttuser/.coworker/wt/of3t-refprec/perf/of3t_refprec/compare_precision.py \
      --ref bundle_ref/grads_f64_043.pt \
      --arm <label>=run/<label>/grads_f64.pt  ... one per arm ... \
      --manifest <label>=run/<label>/manifest.json  ... one per arm ... \
      --sections /home/ttuser/.coworker/wt/of3t-refprec/perf/of3t_orchestrator/SECTION_MASS_MEASURED.json \
      --out REFPREC.json --sidecar-dir sidecar

then `report.py REFPREC.json`. A few minutes, no card. The instrument self-test is
`INSTRUMENT_SELFTEST.json`: A/A against the reference reads exactly 0.0 at r 1.0 cos 1.0 and the
zero model reads exactly 1.0.
