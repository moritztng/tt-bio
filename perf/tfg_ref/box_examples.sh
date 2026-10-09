#!/usr/bin/env bash
# Upstream's four examples/tfg cases (unconstrained, contact, pocket; seed 101, 5 samples) plus parity fixtures of
# 1a14 contact and pocket recorded through capture.py. Waits for box_setup.sh to finish.
set -euo pipefail
until grep -q "^torch .* opendde 1.2.0" /root/setup.log 2>/dev/null; do sleep 30; done
cd /root
printf '1a14\n9lh2\n9sat\n9xqn\n' > examples.txt
RUN_CWD=/root/OpenDDE bash box_run.sh /root/OpenDDE/examples/tfg /root/runs/examples examples.txt 101 "unconstrained contact pocket"
echo 1a14 > fx.txt
for cond in contact pocket; do
  FIXTURE_DIR=/root/fixtures FIXTURE_NAME_PREFIX=1a14_${cond}_ OPENDDE_CMD="python3 /root/capture.py" RUN_CWD=/root/OpenDDE \
    bash box_run.sh /root/OpenDDE/examples/tfg /root/runs/fixtures fx.txt 101 "$cond"
done
echo EXAMPLES_DONE
