# Reproducing the `test_training_full.py` runs on qb2

Nothing here is installed into a shared tree except one file, noted at the bottom.

## Environment

| Piece | Where |
|---|---|
| openfold3 0.5.0 (installed tree) | `/home/ttuser/of3t_gradients/of3pkg/openfold3` |
| openfold3 0.5.0 sdist, for `scripts/datasets/` | `/home/ttuser/of3t_theirtest_env/src/openfold3-0.5.0` |
| pytorch_lightning 2.6.6, torchmetrics, ml_collections | `/home/ttuser/of3t_gradients/deps` |
| lmdb, ijson, memory_profiler, wandb, boto3, awscrt, kalign-python | `/home/ttuser/of3t_theirtest_env/deps` |
| python (torch 2.8.0+cpu) | `/home/ttuser/tt-bio-dev/env/bin/python`, read-only |
| PDB subset | `/home/ttuser/of3t_theirtest_env/datasets`, symlinks onto `of3t-data`'s `/home/ttuser/of3t-data-xhost/datasets` |

The sdist is `openfold3-0.5.0.tar.gz`,
sha256 `a43357fddd4758dfb557e5ce801758f6e3069fc5422323d3a4dd91c33fbe2f6a`. `pip download` refuses
it (its `PKG-INFO` says version 0.0.0 while the filename says 0.5.0), so fetch the file URL
from `https://pypi.org/pypi/openfold3/0.5.0/json` with curl.

`numpy` was pulled into `of3t_theirtest_env/deps` as a transitive dependency and deleted again,
so the base env's numpy 2.5.2 is what imports. Check it if that directory is ever rebuilt.

## The subset

`of3t-data` already had 8 train / 4 val structures in the layout `download_subset.py` expects.
Verified with upstream's own verifier, not by eye:

    cd /home/ttuser/of3t_theirtest_env/src/openfold3-0.5.0/scripts/datasets
    python download_subset.py --verify --target-dir /home/ttuser/of3t_theirtest_env/datasets
    # 388 structure/alignment/template files + 48 reference mols, MISSING 1:
    #   pdb_training_set/templates/val_template_cache/7kud_A.npz  (404 on upstream's S3)

The runner yaml is written by upstream's own `write_runner_yaml`, not by hand:

    from pdb_subset_helpers import write_runner_yaml
    write_runner_yaml(D / "train_pdb_subset.yaml",
                      {"train": D / "training_cache_with_templates_subset_8.json",
                       "val":   D / "validation_cache_with_templates_subset_4.json"},
                      D / "pdb_training_set")

## The runs

    /home/ttuser/of3t_theirtest_env/run_theirtest.sh <-k expr> <accelerator> <eval_triton_off>

| Log | Shims | Command |
|---|---|---|
| `logs/D_unmodified.log` | none | plain pytest, no `-p of3t_theirtest_shim` |
| `logs/C_bothcases_cpu.log` | 1+2 | `run_theirtest.sh "" cpu` |
| `logs/A_smoke_cpu.log` | 1+2 | `run_theirtest.sh smoke cpu` |
| `logs/B_smoke_cpu_notriton.log` | 1+2+3 | `run_theirtest.sh smoke cpu 1` |

## The one shared-tree side effect

`openfold3/tests/conftest.py:236-241` runs `setup_biotite_ccd` as a session-autouse fixture and
writes to `biotite.setup_ccd.OUTPUT_CCD`, which resolves inside whichever venv is running. That
put a 63.4 MB `components.bcif` (upstream's pinned CCD, from `s3://openfold3-data`) into
`/home/ttuser/tt-bio-dev/env/lib/python3.12/site-packages/biotite/structure/info/`. The file did
not exist there before — everything else in that directory is dated 1 Sep. It is additive, not
an overwrite, but it does change what biotite hands any co-tenant that asks for the CCD, so it
is recorded here rather than left to be found. A copy is kept at
`/home/ttuser/of3t_theirtest_env/components.bcif` and the shared one is removed at the end of
the row; re-running any of the above downloads it again.
