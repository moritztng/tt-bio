#!/bin/bash
# Run upstream openfold3 0.5.0 tests/test_training_full.py on qb2.
# $1 = pytest -k expression (empty = both cases); $2 = OF3T_ACCELERATOR ("" = unmodified)
set -u
OF3=/home/ttuser/of3t_gradients/of3pkg
export PYTHONPATH=$OF3:/home/ttuser/of3t_gradients/pylibs:/home/ttuser/of3t_gradients/deps:/home/ttuser/of3t_theirtest_env/deps:/home/ttuser/.coworker/wt/of3t-theirtest/perf/of3t_theirtest/harness
export PATH=$OF3/bin:$PATH
export OPENFOLD_PDB_SUBSET_DIR=/home/ttuser/of3t_theirtest_env/datasets
export WANDB_MODE=offline
export OF3T_ACCELERATOR="${2:-}"
export OF3T_EVAL_TRITON_OFF="${3:-}"
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$OF3"
K=()
[ -n "${1:-}" ] && K=(-k "$1")
"$PY" -m pytest openfold3/tests/test_training_full.py -v -rs --no-header -p no:cacheprovider \
      -p of3t_theirtest_shim "${K[@]}"
echo "PYTEST_EXIT=$?"
