#!/usr/bin/env bash
# The changed code is reachable only through fp32_softmax=True: RF3's pairformer and OpenFold3's
# four sites. These are those models' own gate cells for it.
set -u
WT=/home/ttuser/.coworker/wt/rf3-1024aa-exponent-gate
cd "$WT" || exit 1
PYTHONPATH=. TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:rf3-1024aa-exponent-gate \
  /home/ttuser/tt-bio-dev/env/bin/python3 -m pytest -q \
    tests/test_openfold3_pairformer.py tests/test_openfold3_msa.py \
    tests/test_openfold3_msa_embedder.py tests/test_rf3_featurizer.py \
    tests/test_fp32_softmax_l1_backoff.py 2>&1 | tail -15
echo ALLDONE_TESTS
