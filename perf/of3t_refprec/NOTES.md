# of3t-refprec — where the row is, for whoever picks it up

## The question

Every gradient figure in this campaign is measured against a **float64** reference and nobody
trains in float64. This row measures the precision floor of the training recipe itself: upstream
OpenFold3 0.4.3, its own weights, its own frozen batch, its own draws, run in single precision
and in bf16 autocast, scored against `grads_f64_043.pt`.

## Inputs, all verified this pass

`/home/ttuser/of3t_refprec/bundle_ref/`, hardlinked from `of3t_rebase/bundle_min_043` and
re-hashed here rather than trusted:

    grads_f64_043.pt   1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4  2,947,844,653 B
    w0_043.pt          4aafc28670c12a7c6bf9a9cf8a51405c7b7b9a976677c7f3f8a940472f0ed758
    batch_step003.pt   3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f
    draws_recycles0.pt 36c8136c7c52cd55e7b15a2d1d627666094c64095e85ff7c9d38e05b66beacc5

All four match `bundle_min_043/MANIFEST.json`. The float64 arm is **not** rebuilt.

## The arms

`/home/ttuser/of3t_refprec/run_arm.sh <name> <dtype> <autocast> [draws]`, everything else pinned
to the reference's own inputs. Launched detached under `setsid`, logs `arm{2,3,4}.log`,
`negctl.log` in `/home/ttuser/of3t_refprec/`.

| arm | dtype | --autocast | draws |
|---|---|---|---|
| `arm2_f32_upstream` | float32 | `upstream` | the reference's |
| `arm3_f32_removed` | float32 | `removed` | the reference's |
| `arm4_bf16_autocast` | float32 params | `bf16` | the reference's |
| `negctl_f32_permuted` | float32 | `upstream` | **permuted**, negative control |

The float64 backward took 1,043 s at OMP=14 on this box. These run concurrently at OMP 7/7/3/3,
so wall clock is roughly an hour, not the 20 minutes a single arm would take.

## Already established, before any arm was read

  * **The instrument is calibrated at both ends.** A/A against the reference itself reads exactly
    `0.0` with `r` 1.0 and `cos` 1.0; the zero model reads exactly `1.0`.
    `INSTRUMENT_SELFTEST.json`.
  * **The four arms differentiate at the reference's point.** All four `w0.pt` share sha256
    `17667d457c5a66116b54a5fbb1086d82b481a09440a1d1ae53bd3454b0cedd16`, and all 4,936 state-dict
    tensors are bit-identical to `w0_043.pt` cast to float32. `W0_SAME_POINT.json`.
  * **The set definitions reproduce the published shares.** Device-arm scope 735 tensors /
    52.2644 %; the attention side by pattern is 438 tensors / 27.5862 % and the rest 297 /
    24.6782 %, which exceed the published 288 / 27.2441 % and 259 / 23.8917 % by 0.3421 % and
    0.7865 % — together the 1.1286 % the device arm never reached. A20: mine are the
    full-denominator sets and the difference is exactly the unreached mass.
  * **Arms 2 and 3 must agree bit for bit on this box, and the counters say why.** All nine
    `torch.amp.autocast` sites in the 0.4.3 tree name `device_type="cuda"`, so torch disables
    every one of them on a CPU run — `cast_policy` counts contexts entered against contexts torch
    actually enabled. At float32 `Tensor.float()` is identity, so `no_autocast` has nothing left
    to remove. Arm 3 is therefore an arithmetic control rather than a second measurement, and the
    separation between the two is only observable on CUDA. Reported as such, not as a finding.

## Resuming

If the arms are finished (`run/<arm>/manifest.json` exists):

    cd /home/ttuser/of3t_refprec
    OMP_NUM_THREADS=2 nice -n 15 /home/ttuser/tt-bio-dev/env/bin/python \
      /home/ttuser/.coworker/wt/of3t-refprec/perf/of3t_refprec/compare_precision.py \
      --ref bundle_ref/grads_f64_043.pt \
      --arm arm2_f32_upstream=run/arm2_f32_upstream/grads_f64.pt \
      --arm arm3_f32_removed=run/arm3_f32_removed/grads_f64.pt \
      --arm arm4_bf16_autocast=run/arm4_bf16_autocast/grads_f64.pt \
      --arm negctl_f32_permuted=run/negctl_f32_permuted/grads_f64.pt \
      --manifest arm2_f32_upstream=run/arm2_f32_upstream/manifest.json \
      --manifest arm3_f32_removed=run/arm3_f32_removed/manifest.json \
      --manifest arm4_bf16_autocast=run/arm4_bf16_autocast/manifest.json \
      --manifest negctl_f32_permuted=run/negctl_f32_permuted/manifest.json \
      --sections /home/ttuser/.coworker/wt/of3t-refprec/perf/of3t_orchestrator/SECTION_MASS_MEASURED.json \
      --out REFPREC.json --sidecar-dir sidecar

then `perf/of3t_refprec/report.py REFPREC.json` for the figures, and bank `REFPREC.json` plus the
sidecars under `perf/of3t_refprec/`. The comparison takes a few minutes and no card.

An arm that died leaves no `manifest.json`; re-launch just that one with `run_arm.sh`.

## Pass 2 (2026-09-20, from 08:37Z) — the arms had finished and the manifest step killed them

The first launch of arms 2, 3 and 4 completed the forward, the backward and the RNG-replay
forward, wrote `grads_f64.pt`, and then died on `NameError: name 'openfold3' is not defined`
while composing the manifest. D42 moved the provenance read into a module-level
`_openfold3_version()` that looks up `openfold3` as a global, and the only `import openfold3` in
`bundle_min.py` was local to `main`. The float64 reference predates D42, so the defect had never
been reached. Fixed as D43 (import inside the function, `"unknown"` if it fails), and the
function was CALLED before it was believed: it reads
`0.4.3 (from PKG-INFO beside the imported tree)`.

`bundle_ref/MANIFEST.json` records D42 with `"payload_rebuilt": false` — the field was corrected
by hand and the generator that was supposed to produce it had never run. After D43 the generator
produces that exact string.

**The crashed runs' gradients were hashed before anything was relaunched**
(`PRIOR_GRAD_SHA.txt`), so the relaunch has a determinism control it did not have before:

    arm2_f32_upstream    09f1217c8ea254d04f1bfdae73585f058cb2aec51557699bfae3a8090dadd548
    arm3_f32_removed     09f1217c8ea254d04f1bfdae73585f058cb2aec51557699bfae3a8090dadd548
    arm4_bf16_autocast   ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb
    negctl_f32_permuted  a5cbbf40819f0c24e6adca2f545b45a0c7aed23d0de95308636fec0a499f43a0

Arms 2 and 3 are **byte-identical**, which is the pass-1 prediction from the autocast-site
census turned into a measurement: all nine `torch.amp.autocast` sites in 0.4.3 name
`device_type="cuda"`, torch disables every one of them on a CPU box, and `Tensor.float()` is
identity at float32, so `no_autocast` has nothing to remove. Arm 3 is an arithmetic control, not
a second measurement, and the arm-2-minus-arm-3 separation the brief asks for is only observable
on CUDA.

## The instrument, after pass 2

Three additions, all recalibrated afterwards (A/A still exactly 0.0 at r 1.0 cos 1.0, zero model
still exactly 1.0):

  * the one leaf `blocks.N.attention_pair_bias.layer_norm_a.layer_norm_s.weight` as its own set
    (24 tensors, 25.5795 % of the model, device reads mass-weighted 10.6980), and the device-arm
    scope minus it (device reads 0.2929);
  * `blocks.8` of that leaf — 8.05416 % of the model alone, device rel 18.504 — emitted with this
    row's rel, r and cos beside the device figure, provenance per entry;
  * the denominator asserted against the campaign's published 10.279642678524985 instead of
    assumed. It reads 10.279642678524981 over 4,170 tensors.

A reference tensor with no counterpart in an arm used to score as unmeasurable, which drops its
mass from the numerator and leaves it in the denominator. It is now scored exactly the way the
zero model is scored, and still named.

## Reference facts the arms are compared against

`bundle_ref/MANIFEST.json`, `validated_gradient`: loss `1.267624369070698`, gradient global norm
`3.206188185139011`, 4,170 parameters, 0 absent, 4,161 non-zero, draws `draws_recycles0.pt` with
0 mismatches, forward 340.7 s and backward 1043.2 s at float64. `determinism_A13` already shows
two float64 runs bit-identical on 4,170 of 4,170.
