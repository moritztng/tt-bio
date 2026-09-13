#!/bin/bash
# The cross-model spot check for TT_BIO_DEVICE_CONDITIONING, on the tree that ships it on.
#
# The call-site census says no model but Boltz-2 executes a changed branch. Digest reproduction is
# a stronger statement than an RMSD comparison, so this folds the two sibling models on the default
# tree and checks they still write the digests docs/tuning-flags.md quotes:
# protenix-v2 15772214c5b9e990, openfold3 6ee6ac7a3e730688 (perf/b2z2_msa_census/xmodel/*.json).
# One arm, because this is a reproduction and not an A/B.
set -u
cd /home/ttuser/.coworker/wt/b2z2-cond-ship || exit 1
for m in protenix-v2 openfold3; do
  echo "=== $m ==="
  env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:b2z2-cond-ship \
    /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2z2_msa_census/xmodel_pwa_ab.py \
      --model "$m" --size 512 --arms on \
      --out "perf/b2z2_cond/out/xmodel_${m}_512.json" || echo "FAILED $m"
done
