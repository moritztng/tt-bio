#!/bin/bash
# The issue #21 card leg, one trajectory per PROCESS.
#
#   card_chain21.sh <arm> <card> [target_trajectories]
#
# Why not card_run.sh: one process running the whole campaign OOMs. On 2026-10-07 the 8-trajectory
# run (pid 1416740) completed trajectories 1-3 at 224 tokens in `fast` and then refused in the
# Evoformer backward on trajectory 4, at the same token axis, the same mode and the same card, with
# "33.454 GB of 34.226 GB held by this fold". The memory law puts a single 224-token `fast` round at
# 2.778 GB and `fast` holds 672 tokens on this card, so that is device memory accumulating across
# trajectories inside the process, not a fold that is too big. `memory=lean` (1.866 GB a round)
# would only delay it.
#
# BindCraft 2 keeps campaign progress in <project>/.campaign_state.json and `max_trajectories` caps
# the campaign TOTAL (bc2 bindcraft/campaign.py:230, `trajectory_count >= max_trajectories`). So a
# fresh process with the cap raised by one does exactly one more trajectory against the same project
# and then exits, which hands the device back to the driver every trajectory.
#
# Taking the flock per trajectory rather than per campaign is deliberate: the other BCI rows get the
# chip between trajectories instead of waiting out a whole campaign.
set -euo pipefail
cd "$(dirname "$0")/../.."

arm=${1:?arm: fixed | prefix | control}
card=${2:?the card the grant in state/bci/CHIPS.md names}
target=${3:-8}

case "$arm" in
  fixed)   tree=$PWD;                    extra=() ;;
  prefix)  tree=/home/ttuser/bci_prefix; extra=() ;;
  control) tree=/home/ttuser/bci_prefix; extra=(--no-extra-msa) ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
esac

out=/home/ttuser/bci_card/$arm
mkdir -p "$out"                      # resumed on purpose, never cleared

export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export PYTHONPATH=$tree
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card
export TT_BIO_LEASE_HOLDER=worker:bci-validation
export JAX_COMPILATION_CACHE_DIR=/home/ttuser/bci_card/xlacache

state=$out/.campaign_state.json
done_n() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['trajectories'])" "$state" 2>/dev/null || echo 0; }
acc_n()  { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['accepted'])"     "$state" 2>/dev/null || echo 0; }

while :; do
  n=$(done_n); a=$(acc_n)
  echo "=== chain21 $(date -u +%FT%TZ): $n trajectories done, $a accepted, target $target" >&2
  [ "$n" -ge "$target" ] && { echo "=== target reached"; break; }

  cap=$((n + 1))
  # SIGINT then SIGTERM on a hang, per CHIPS.md RULES. 2 h is ~6x the 1152 s slowest trajectory.
  flock /home/ttuser/bci_chip"$card".lock \
    nice -n 10 timeout -s INT -k 120 2h \
    /home/ttuser/fdv_fresh/venv/bin/python3 -u perf/bci_validation/card_campaign.py \
      --arm "$arm" --ttbio "$tree" --out "$out" --max-trajectories "$cap" \
      "${extra[@]}" "${@:4}" && rc=0 || rc=$?

  [ -f "$out/summary.json" ] && cp "$out/summary.json" "$out/summary_t$cap.json"
  echo "=== chain21 trajectory cap=$cap exited rc=$rc" >&2

  after=$(done_n)
  if [ "$after" -le "$n" ]; then
    echo "=== chain21 STOP: trajectory count did not advance ($n -> $after), rc=$rc" >&2
    exit "${rc:-1}"
  fi
done
echo "=== chain21 finished: $(done_n) trajectories, $(acc_n) accepted"
