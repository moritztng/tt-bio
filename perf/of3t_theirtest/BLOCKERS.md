# Every CUDA-specific assumption on `test_training_full.py`'s path

Upstream openfold3 **0.5.0**, sdist `openfold3-0.5.0.tar.gz`
sha256 `a43357fddd4758dfb557e5ce801758f6e3069fc5422323d3a4dd91c33fbe2f6a`, installed tree
`/home/ttuser/of3t_gradients/of3pkg/openfold3`. Host qb2, torch **2.8.0+cpu**,
`torch.cuda.is_available() == False`, `torch.version.hip is None`, no Triton, no DeepSpeed,
no cuequivariance. FIRES / INERT is what was observed in the runs under `logs/`, not read off
the source.

## A. The test's own

| # | Site | What | Status |
|---|------|------|--------|
| A1 | `tests/test_training_full.py:174` | `@skip_unless_cuda_available()` | **FIRES** — the only gate reached unadapted |
| A2 | `tests/test_training_full.py:54` | imports it from `tests/utils/compare_utils.py` | — |
| A3 | `tests/utils/compare_utils.py:160-167` | `skip_unless_cuda_available()` -> `skip_unless_accelerator_available("cuda")` | **FIRES** |
| A4 | `tests/utils/compare_utils.py:136-157` | builds `pytest.mark.skipif(True, reason="Requires cuda; found cpu")` | **FIRES** |
| A5 | `tests/utils/compare_utils.py:123` | `ACCELERATORS = ("cuda", "rocm", "mps")` — CPU is not an accelerator by construction | **FIRES** |
| A6 | `tests/utils/compare_utils.py:127-133` | `current_accelerator()`: `torch.cuda.is_available()`, else `torch.backends.mps.is_available()`, else `None` | **FIRES** |
| A7 | `tests/conftest.py:24-31` | session device fixture parametrises `cpu` + `cuda` | INERT, this test takes no `device` fixture |

A1 is a hard gate and nothing behind it is reachable while it stands. It is also the *only*
one of the test's own assumptions that is CUDA-specific: the rest of the body
(`_require_local_subset`, `shutil.which("run_openfold")`, the yaml materialisation, the
`_run_streaming` watchdog, both asserts) is device-agnostic.

## B. Upstream's launcher and config, beneath the test

| # | Site | What | Status |
|---|------|------|--------|
| B1 | `entry_points/validator.py:126` | `accelerator: str = "gpu"` — the generated runner yaml carries no `accelerator` key, so pydantic fills this | **FIRES**, Lightning then refuses to start |
| B2 | `entry_points/validator.py:129` | `devices: int = 1  # number of GPUs per node` | INERT at 1 |
| B3 | `scripts/datasets/pdb_subset_helpers.py:548-555` | the generated `pl_trainer_args`: `devices: 1`, `precision: bf16-mixed`, no `accelerator` | source of B1 |
| B4 | `entry_points/experiment_runner.py:91-101` | `_accelerator_will_use_mps` | INERT, no MPS |
| B5 | `entry_points/experiment_runner.py:201,213` | `num_gpus` / `world_size` | INERT at 1x1 |
| B6 | `entry_points/experiment_runner.py:239-258` | `DeepSpeedStrategy` / `DDPStrategy`; NCCL would come from here | INERT — world size 1 takes the `"auto"` branch, so no NCCL and no process group |
| B7 | `entry_points/import_utils.py:42-43` | `torch.backends.cuda.preferred_blas_library("cublas")` on ROCm | INERT, guarded by `torch.cuda.is_available()` |

No `pin_memory` and no `nccl` appear anywhere in the package.

## C. Custom kernels in the model

| # | Site | What | Status |
|---|------|------|--------|
| C1 | `projects/of3_all_atom/config/model_config.py:119` | `settings.memory.eval.use_triton_triangle_kernels: True` in the base config | **FIRES** |
| C2 | `core/model/layers/triangular_multiplicative_update.py:1125` | `raise RuntimeError(_TRITON_UNAVAILABLE_ERROR)` | **FIRES**, in the first sanity-check validation step |
| C3 | `core/model/primitives/attention.py:449-453` | the same refusal on the attention side | not reached, C2 raises first |
| C4 | `core/model/primitives/attention.py:54` | Triton evoformer kernel import | INERT — warns and takes the non-kernel path |
| C5 | `projects/of3_all_atom/config/model_config.py:105` | `memory.train.use_triton_triangle_kernels: False` | INERT by default |
| C6 | `pdb_subset_helpers.py:~568` | the generator sets `memory.train.use_deepspeed_evo_attention: False` with the comment that a kernel build issue must not be a variable | INERT, deliberately |
| C7 | `core/kernels/cueq_utils.py:16` | `is_cuequivariance_available()` requires `torch.cuda.is_available()` | INERT, yaml sets `use_cueq_triangle_kernels: False` |
| C8 | `core/kernels/triton/swiglu.py:30` | `torch.version.hip is not None` | INERT |

C1 is the asymmetry that matters. The generator disables all three custom-kernel flags on the
**train** path and leaves the **eval** path at the base config's `True`. On CUDA and ROCm that
is invisible because torch's GPU wheels ship Triton. Off both, Lightning's sanity-check
validation runs before the first training step, so C2 raises before any training happens at
all. One config key stands between the test and its first optimizer step.

## D. Hardcoded `autocast("cuda")` — does not raise, and changes the arithmetic

| # | Site | What |
|---|------|------|
| D1 | `core/model/primitives/attention.py:158` | `torch.amp.autocast("cuda", dtype=attn_dtype)` |
| D2 | `core/model/heads/prediction_heads.py:224` | `torch.amp.autocast(device_type="cuda", dtype=pairformer_dtype)` |
| D3 | `core/loss/loss_module.py:150` | `torch.amp.autocast(device_type="cuda", dtype=torch.float32)` |
| D4 | `core/loss/diffusion.py:80`, `core/utils/geometry/kabsch_alignment.py:67` | `autocast(autocast_device_type(H), dtype=torch.float32)` |

D1-D3 emit `torch/amp/autocast_mode.py:266 UserWarning: User provided device_type of 'cuda',
but CUDA is not available. Disabling`. D4 emits `autocast_mode.py:283 ... In CPU autocast, but
the target dtype is not supported. Disabling autocast. CPU Autocast only supports dtype of
torch.bfloat16, torch.float16`. Both warnings are in `logs/B_smoke_cpu_notriton.log`.

So on CPU the fp32 islands upstream deliberately keeps around softmax, the pairformer head,
the loss module, the diffusion loss and the Kabsch alignment are silently **not** islands.
Whatever a CPU run of this test measures, it is not the same arithmetic as the H200 run — the
test asserts an exit status and a checkpoint file, so it would not notice.

## E. `torch.cuda` calls that are no-ops on a CPU-only torch

`core/utils/callbacks.py:269` `torch.cuda.manual_seed_all`; `projects/of3_all_atom/runner.py:849`
and `core/utils/device_utils.py:42` `torch.cuda.empty_cache()` (guarded internally by
`is_initialized()`); `core/utils/callbacks.py:174-182` guarded by `torch.cuda.is_available()`;
`core/utils/callbacks.py:93-130` `torch.cuda.memory._dump_snapshot` / `_record_memory_history`
in the OOM callback, not enabled by this runner yaml; `core/utils/rigid_utils.py:747,750,1379`
and `core/utils/geometry/rigid_matrix_vector.py:185` `.cuda()` inside `.cuda()` helper methods,
never called on this path. None of these fired.

## F. Not CUDA, found while clearing the way

| # | What |
|---|------|
| F1 | `pdb_training_set/templates/val_template_cache/7kud_A.npz` is in the manifest `download_subset.py` builds from upstream's own subset cache and **404s** on `s3://openfold3-data`. `--verify` reports `MISSING 1 files`. Validation tolerates it: run B reached validation structure 3 of 4 with no step failure. |
| F2 | The checked-in `scripts/datasets/train_pdb_subset.yaml` in the sdist is stale against its own generator — `epoch_len: 32` vs 4, `use_deepspeed_evo_attention: true` vs False, and absolute `/home/jnwei/workspace/...` dataset paths. `generate_subset_cache.py` overwrites it, so it is not what runs, but a reader who opens that file gets the wrong config. |

## What closing the real blocker costs

A1, B1 and C1 are each one line, and with them cleared the test executes: the shims under
`harness/` clear A1 and B1, and the C1 diagnostic shows the training step itself runs. **None
of that puts it on our backend.** The test drives `run_openfold train`, which builds upstream's
`torch.nn.Module` tree under PyTorch Lightning, and Lightning dispatches by torch device.
tt-bio does not give ttnn a torch device: there is no `rename_privateuse1_backend`, no
`torch.library` registration and no Lightning `Accelerator` anywhere in `tt_bio/`. Its reverse
mode is a separate tape over raw ttnn handles, and `tt_bio/autograd.py:12-19` says why — a ttnn
handle returned from `torch.autograd.Function.forward` gets no `grad_fn`, so torch's engine
cannot route it. On top of that, tt-bio's OF3 is an independent reimplementation whose weight
remap fuses pairs of upstream tensors (campaign R3), so it is not a drop-in for upstream's
module tree either.

Two ways to close it, and they are not equivalent:

1. **A torch PrivateUse1 backend over ttnn**, covering every op upstream's training step uses
   plus autograd formulas, and a Lightning `Accelerator` on top. This keeps the artifact honest
   — upstream's test, upstream's model, our silicon. It is a new subsystem, not a shim, and
   `tt_bio/autograd.py`'s docstring records that the carrier-tensor route was tried and
   measured as a dead end.
2. **Swap `OF3ProjectEntry`'s model for tt-bio's ttnn OF3** inside upstream's Lightning module.
   Much less work, and it destroys the artifact: the test would then assert that *our* model
   trains, which is what the rest of the campaign already measures directly and with far better
   instruments than an exit code and a `*.ckpt` glob.
