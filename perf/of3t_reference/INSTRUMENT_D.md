# Upstream's own training test, passed

OpenFold3 ships exactly one end-to-end training test, `openfold3/tests/test_training_full.py`. On
every host this fleet owns it reported `2 skipped, "Requires cuda; found cpu"`, so the campaign
recorded it as not passed. On a rented A100 it passes:

```
openfold3/tests/test_training_full.py::test_train[smoke]        PASSED
openfold3/tests/test_training_full.py::test_train[full_subset]  PASSED
=================== 2 passed, 1 warning in 648.43s (0:10:48) ===================
```

Exit 0, wall 655 s. Both cases wrote a checkpoint, 6,713,476,430 B each. The test drives the real
`run_openfold train` console script through their `pl.LightningModule`, so it exercises their
training entry point rather than a harness of ours.

Box: A100-SXM4-80GB, driver 595.71.05, torch 2.5.1+cu124, CUDA 12.4; openfold3 0.5.0 at
`c4771653c5d0a3ebb0b3af71b05efd64bc44ee86`.

## Three gates, and CUDA was only the first

`@skip_unless_cuda_available()` is the one the campaign kept hitting. Two more sit behind it:

1. **The local PDB subset.** `_require_local_subset()` at line 117. Build it with their own
   scripts:

   ```
   cd scripts/datasets
   python3 generate_subset_cache.py      # 8 training entries, 4 pinned validation entries, runner yaml
   python3 download_subset.py --workers 16
   ```

   388 objects, 1.8 GB, and **one 404 on their S3 bucket**:
   `pdb_training_set/templates/val_template_cache/7kud_A.npz`. Both cases pass without it, so it
   is not on the path either takes. Upstream's gap, not ours.

   Both scripts default their target to `./datasets` relative to where they are invoked, so run
   from `scripts/datasets/` the data lands in `scripts/datasets/datasets/`. Point the test at it
   with `OPENFOLD_PDB_SUBSET_DIR` instead of moving anything.

2. **`run_openfold` on PATH.** A third skip at line 181 that nothing had reached before. A source
   checkout is importable but installs no console script. `pip install -e . --no-deps
   --no-build-isolation`.

## What it does and does not say

It says upstream's training stack runs end to end on their own data through their own entry point
on a stock CUDA box, and therefore that the reference in `bundle_min/` was built against a stack
that works. It says nothing about our side: the test never touches tt-bio.

The 648 s is not a per-step figure. The two cases differ, and `d19_forward_discriminator.py` shared
the card for part of the run.
