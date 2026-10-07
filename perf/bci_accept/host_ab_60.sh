#!/bin/bash
# Host-JAX half of the harden A/B, card-free, on the real 60-aa states the states60 capture wrote.
# No device is opened: harden_forward_ab.py runs the host arm alone when --card is omitted.
set -u
cd /home/ttuser/.coworker/wt/bci-accept
exec nice -n 19 timeout 5400 /home/ttuser/fdv_fresh/venv/bin/python perf/bci_accept/harden_forward_ab.py \
  --states .bci/harden_states_60.pkl \
  --af2-weights /home/ttuser/bcx_e2e/af2_params \
  --target-pdb /home/ttuser/bcx_e2e/bc2/settings/target/structures/hPDL1.pdb \
  --binder-length 60
