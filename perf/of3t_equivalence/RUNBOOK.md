# Reproducing the upstream OpenFold3 training test on this host

Everything here is upstream`s, except the parser swap in `subset_nan_shim.py` and the
switch overrides in `logs/cpu_probe.py`. Both are recorded because both change what ran.

## The version question

`openfold3/tests/test_training_full.py` is **not in 0.4.3**, the version tt-bio pins at
`NOTICE:45`. It is in **0.5.0**. Confirmed by enumerating every `test_*.py` in both sdists:
0.4.3 ships 67 test files and none of them trains; `test_model_runner.py` is 133 lines of
mocked OOM handling over `predict_step`. 0.5.0 ships the training test.

    pip download openfold3==0.4.3 --no-deps   # no test_training_full.py
    pip download openfold3==0.5.0 --no-deps   # has it

## Setup

    python3 -m venv --system-site-packages venv      # torch 2.11.0+cpu, pytorch_lightning 2.6.5
    ./venv/bin/pip install openfold3==0.5.0 pytest   # the wheel ships openfold3/tests/ and run_openfold

    # upstream`s subset generator cannot parse upstream`s published cache -- see below
    ./venv/bin/python perf/of3t_equivalence/subset_nan_shim.py \
        <sdist>/scripts/datasets --target-dir ./datasets
    ./venv/bin/python <sdist>/scripts/datasets/download_subset.py \
        --target-dir ./datasets --workers 16

The cache is `s3://openfold3-data/pdb_training_set/`, public and unsigned, same bucket as the
weights. 1.68 GB for the full training cache, 73 MB for the 435 subset files.

## The three gates that stop it

1. **`NaN` in their published cache.** `pdb_subset_helpers.enumerate_structure_ids` streams the
   cache with ijson; the cache contains bare `"resolution": NaN`, which is not JSON. No ijson
   backend accepts it: yajl2_c, yajl2_cffi, yajl2 and the pure-python backend all raise.
   Python`s own `json` does accept it, so the shim swaps the parser and leaves the ID ordering,
   the `random.Random(42).sample` and `write_subset` untouched -- the subset is still theirs.
2. **`@skip_unless_cuda_available()`** on `test_train`. Run as shipped on this host the test
   reports `2 skipped -- Requires cuda; found cpu`. It is not a passing test here and saying so
   is the point; see `logs/their_test_as_shipped.log`.
3. **Two kernel switches that assume NVIDIA.** `use_deepspeed_evo_attention: true` is baked into
   the generated runner yaml, and `use_triton_triangle_kernels` defaults on. With the first
   turned off the run reaches `triangular_multiplicative_update.py:1125` and raises
   `Triton kernels requested ... but openfold3.core.kernels.triton is not available`.

## Where our backend plugs in

`ModelRunner.__init__` takes `model_class` and does `self.model = model_class(self.config)`
(`core/runners/model_runner.py:51`), and `training_step` is `outputs = self.model(batch)` then
`self.loss(batch, outputs)`. So the substitution point is a single injected class, and their
loss, LR schedule, `grad_manager`, optimizer and EMA stay theirs. That is the cheapest honest
route to "their training test, our backend", and it is what PROTOCOL §2 wants anyway.
