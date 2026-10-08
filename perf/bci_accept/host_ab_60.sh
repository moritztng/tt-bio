#!/bin/bash
# Host-JAX half of the harden A/B, card-free, on the real 60-aa states the states60 capture wrote.
# No device is opened: harden_forward_ab.py runs the host arm alone when --card is omitted.
#
# PYTHONPATH is not optional. BindCraft 2's settings.py resolves CORE_DEFAULTS relative to its own
# __file__'s grandparent, so the site-packages copy in fdv_fresh/venv looks for
# site-packages/settings/core/default.json and raises FileNotFoundError. The bcx_e2e source tree has
# the settings/ directory beside bindcraft/, and putting it first on PYTHONPATH is what every
# working BC2 run on this box does.
set -u
cd /home/ttuser/.coworker/wt/bci-accept
export PYTHONPATH=/home/ttuser/bcx_e2e/bc2:/home/ttuser/.coworker/wt/bci-accept
export JAX_PLATFORMS=cpu
exec nice -n 19 timeout 5400 /home/ttuser/fdv_fresh/venv/bin/python perf/bci_accept/harden_forward_ab.py \
  --states .bci/harden_states_60.pkl \
  --af2-weights /home/ttuser/bcx_e2e/af2_params \
  --target-pdb /home/ttuser/bcx_e2e/bc2/settings/target/structures/hPDL1.pdb \
  --binder-length 60
