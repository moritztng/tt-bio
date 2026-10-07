#!/bin/bash
# Host-JAX one_hot_weight sweep across the harden boundary, card-free.
# Tells a step at the one-hot crossing from a smooth trend; the card arm still runs 0 and 1 only.
set -u
cd /home/ttuser/.coworker/wt/bci-accept
export PYTHONPATH=/home/ttuser/bcx_e2e/bc2:/home/ttuser/.coworker/wt/bci-accept
export JAX_PLATFORMS=cpu
exec nice -n 19 timeout 5400 /home/ttuser/fdv_fresh/venv/bin/python perf/bci_accept/harden_forward_ab.py \
  --states .bci/harden_states_60.pkl \
  --af2-weights /home/ttuser/bcx_e2e/af2_params \
  --target-pdb /home/ttuser/bcx_e2e/bc2/settings/target/structures/hPDL1.pdb \
  --binder-length 60 \
  --one-hot-weights 0.0 0.25 0.5 0.75 1.0
