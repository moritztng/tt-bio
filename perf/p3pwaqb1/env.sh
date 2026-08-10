# one place for the card, the wheel and the worktree, so no run can silently take the wrong one
WT=/home/ttuser/.coworker/wt/protenix-trunk--p3-pwa-qb1
PY=/home/ttuser/tt-bio-dev/env/bin/python3          # ttnn 0.67.4 -- the campaign-absolute wheel
PY68=/home/ttuser/tt-boltz2/env/bin/python3         # ttnn 0.68.0 -- the secondary spot-check
export TT_VISIBLE_DEVICES=2
export TT_BIO_LEASE_HOLDER=worker:protenix-trunk--p3-pwa-qb1
export PYTHONPATH=$WT
