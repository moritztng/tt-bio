#!/bin/bash
# Take the first card that frees, on either box, and run the whole chain on it.
#
# Ask 11511 granted this row a STANDING permission -- "take the first that frees, no further ask"
# -- and a promise is not a mechanism. The fleet dispatcher ticks every two minutes and this
# row's turns are bounded, so "first claim" only means anything if something is watching when the
# card actually frees. This is that something.
#
# It runs on the ORCHESTRATOR host, which is the only one that can answer card_free.sh's fourth
# condition: a live worker.sh ASSIGNED to a host+card, held by a row that is between opens and
# about to open again. The other three are readable remotely; that one is only in pc's process
# table, and on 2026-09-26 qb2 card 3 read free on all three of the others while land-standing
# held the assignment.
#
# It launches and then STOPS. It does not interpret, re-run or conclude anything: the artifacts
# land on disk under the worktree and the next pass of this row reads them. A watcher that
# reached a verdict would be handing one tick's judgement to the next without anyone taking it.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
WT=/home/ttuser/.coworker/wt/bcx-bwbytes
LOCK=$HERE/grab.lock
TRIES=${TRIES:-480}          # 45 s apart -- six hours, past every holder's projected finish
SLEEP=${SLEEP:-45}

if [ -e "$LOCK" ]; then echo "another grabber holds $LOCK -- not starting a second"; exit 1; fi
echo "$$ $(date -u +%FT%TZ)" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

for i in $(seq 1 "$TRIES"); do
  out=$(bash "$HERE/card_free.sh" 2>&1); rc=$?
  ts=$(date -u +%H:%M:%SZ)
  if [ $rc -ne 0 ]; then
    [ $((i % 10)) -eq 1 ] && echo "$ts none free"
    sleep "$SLEEP"; continue
  fi
  line=$(echo "$out" | grep -m1 ": FREE$")
  short=$(echo "$line" | awk '{print $1}')
  card=$(echo "$line" | awk '{print $3}')
  case $short in qb1) host=tt-quietbox;; qb2) host=tt-quietbox2;; *) echo "$ts unknown box in: $line"; exit 1;; esac
  echo "$ts CLAIMING $short card $card -- $line"

  # The chain itself re-checks the lease and the device node before it opens anything, so this is
  # the first of two independent refusals rather than the only one.
  # The success test is a marker the chain WRITES after its own preflight passes, not `pgrep`.
  # A pgrep for the chain matches the launching command's own text -- that is how the dry run at
  # 03:33Z reported LAUNCHED over a chain that had already refused all seven steps and exited,
  # and it is the third time in one night that a pgrep on this fleet matched its own caller.
  if ssh -o BatchMode=yes -o ConnectTimeout=10 "ttuser@$host" \
       "cd $WT && rm -f $WT/perf/bcx_bwbytes/runs/chain.claimed; \
        setsid nohup env CARD=$card $WT/perf/bcx_bwbytes/chain.sh \
        > $WT/perf/bcx_bwbytes/runs/chain.log 2>&1 < /dev/null & sleep 10; \
        test -s $WT/perf/bcx_bwbytes/runs/chain.claimed"; then
    echo "$ts LAUNCHED on $short card $card; log $WT/perf/bcx_bwbytes/runs/chain.log"
    exit 0
  fi
  echo "$ts launch on $short card $card did not take -- the chain refused or died; retrying"
  sleep "$SLEEP"
done
echo "$(date -u +%H:%M:%SZ) grab window expired without a free card"
exit 1
