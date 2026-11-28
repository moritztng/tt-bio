#!/bin/bash
# Keep dfd-eval's phase-2 device arm alive on qb2 and bring its poses home. Run from pc every
# ~10 min (artifacts/dfd-eval/tick.d). The first arm waited for its window as a bare nohup and
# died unseen in qb2's 2026-11-27 14:40Z poweroff; nothing relaunched it for ten hours.
# eval_device_arm.sh exits at once when it is already armed or done, so calling it is idempotent.
#
#   eval_device_rearm.sh <name> <forms> <chip> <seconds>
set -uo pipefail
name=${1:-dev3} forms=${2:-gemm,tile} chip=${3:-2} dur=${4:-10800}
R=/home/ttuser/scratch/dfd/eval L=/home/moritz/scratch/dfd/eval/device
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15"
mkdir -p "$L"
if timeout 60 $SSH qb2 "[ -e $R/$name.done ]"; then
  timeout 600 rsync -a -e "$SSH" "qb2:$R/$name.log" $(for f in ${forms//,/ }; do echo "qb2:$R/poses-$f"; done) "$L/" \
    && touch "$L/$name.pulled"
  exit 0
fi
# bounded: an sshd that holds the channel open past setsid+nohup must not hang the tick
timeout 30 $SSH qb2 "setsid nohup bash /home/ttuser/.coworker/wt/dfd-eval/scripts/diffdock_port/eval_device_arm.sh \
  $name $forms $chip $dur $R/subset85.json > /dev/null 2>&1 < /dev/null &"
exit 0
