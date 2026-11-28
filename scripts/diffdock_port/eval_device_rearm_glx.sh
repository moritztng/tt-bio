#!/bin/bash
# Keep dfd-eval's phase-2 device arm alive on one anx-glx chip and bring its poses home. Run from
# pc every ~10 min (artifacts/dfd-eval/tick.d); the qb2 twin is eval_device_rearm.sh.
# The first call at or after <start_epoch> ships this worktree's tree, so both forms run one tree,
# and sets aside poses an earlier arm docked on another one. eval_device_arm_glx.sh exits at once
# when it is already armed or done, so every later call is a no-op plus a pull.
#
#   eval_device_rearm_glx.sh <name> <forms> <chip> <start_epoch> <stop_epoch>
set -uo pipefail
name=${1:?name} forms=${2:?forms} chip=${3:?chip} start=${4:?start epoch} stop=${5:?stop epoch}
WT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
R=/home/ttuser/dfd-eval L=/home/moritz/scratch/dfd/eval/device-$name
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15"
[ "$(date +%s)" -ge "$start" ] || exit 0
[ -e "$L/$name.pulled" ] && exit 0
mkdir -p "$L"
if ! timeout 60 $SSH anx-glx "[ -e $R/device/$name.tree ]" </dev/null; then
  timeout 600 rsync -a --delete --exclude __pycache__ -e "$SSH" "$WT/tt_bio" "$WT/pyproject.toml" anx-glx:$R/wt/ || exit 1
  timeout 300 rsync -a --delete --exclude __pycache__ -e "$SSH" "$WT/scripts/diffdock_port" anx-glx:$R/wt/scripts/ || exit 1
  timeout 60 $SSH anx-glx "cd $R/device && for f in ${forms//,/ }; do [ -d poses-\$f ] && mv poses-\$f poses-\$f.before-$name; done; \
    echo $(git -C "$WT" rev-parse --short=10 HEAD) > $name.tree" </dev/null || exit 1
fi
# bounded: an sshd that holds the channel open past setsid+nohup must not hang the tick
timeout 30 $SSH anx-glx "setsid nohup bash $R/wt/scripts/diffdock_port/eval_device_arm_glx.sh \
  $name $forms $chip $stop $R/device/subset85.json > /dev/null 2>&1 < /dev/null &" </dev/null
timeout 600 rsync -a -e "$SSH" "anx-glx:$R/device/$name.log" "anx-glx:$R/device/$name.tree" \
  $(for f in ${forms//,/ }; do echo "anx-glx:$R/device/poses-$f"; done) "$L/" 2>/dev/null
timeout 60 $SSH anx-glx "[ -e $R/device/$name.done ]" </dev/null && touch "$L/$name.pulled"
exit 0
