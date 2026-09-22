#!/usr/bin/env bash
# D117: run the ref_space_uid_to_perm contract check.
#
# Exits 0 only if both shipped defects still reproduce, all three fixed arms and the positive complete,
# check_perm_contract flags both and passes a valid batch, map_batch_tensors is
# bit-exact with tensor_tree_map where no list-axis feature exists, and no copy of the
# pop/restore workaround has reappeared outside tt_bio/openfold3_batch.py.
#
# OF3_PKG points at an installed openfold3 source tree; the check reads
# openfold_batch_collator out of it rather than importing data_module, which drags in
# pytorch_lightning and lmdb for no reason connected to collation.
set -euo pipefail

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OF3_PKG="${OF3_PKG:-/home/ttuser/of3t_gradients/of3pkg}"
PY="${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}"

PYTHONPATH="$OF3_PKG:$WT" "$PY" "$WT/perf/of3t_d117/perm_contract_check.py" \
  | tee "$WT/perf/of3t_d117/perm_contract_check.json"
