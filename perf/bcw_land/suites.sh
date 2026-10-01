#!/bin/bash
# The BC2 suites the three BCW branches name, on the merged tree. $1 = card ("" = card-free run).
cd "$(dirname "$0")/../.."
card=$1; tag=${card:+card$card}; tag=${tag:-cardfree}
SUITES="tests/test_bindcraft2.py tests/test_bindcraft2_hw.py tests/test_bindcraft2_stall.py
 tests/test_autograd_reference_gate.py tests/test_verbs_scope_autograd.py tests/test_af2_triatt_fused.py
 tests/test_af2_device_floor.py tests/test_tape_generic_op_entries.py tests/test_tuning_flag_docs.py
 tests/test_envflags.py tests/test_triatt_qkv_packed.py tests/test_triatt_bw_query_chunk.py
 tests/test_triatt_narrow_q_fallback.py tests/test_triatt_hifi_pad_up.py tests/test_triatt_hifi_tape_latch.py
 tests/test_triatt_sdpa_hifi_defaults.py"
export PYTHONPATH=$PWD:$HOME/bcx_e2e/bc2
if [ -n "$card" ]; then export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:bcw-land
else export TT_VISIBLE_DEVICES=; fi
echo "tree $(git rev-parse --short HEAD) start $(date -u +%FT%TZ)"
timeout 2700 ~/bcx_e2e_venv/bin/python3 -m pytest -q -rs -p no:cacheprovider $SUITES 2>&1 | tail -80
echo "rc=${PIPESTATUS[0]} end $(date -u +%FT%TZ)"
