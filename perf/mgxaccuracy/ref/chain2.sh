#!/bin/bash
# Second upstream draw per size, each starting only when its predecessor frees its cores, so
# qb2's load stays where it is now (15.7 of 16, shared with two other rows' jobs).
#
# Why a second draw: the device cells are n=8. Against n=8 the smallest one-sided p a rank
# test can reach with ONE upstream draw is 1/C(9,1) = 0.111, so m=1 cannot reject at 0.05
# however good upstream turns out to be; m=2 floors at 1/C(10,2) = 0.022. Two is the minimum
# that can decide whether the 1536 level is the port's or BoltzGen's. Verified against
# scipy's exact Mann-Whitney on a perfectly separated toy sample.
#
# The draws are independent because nothing here sets a seed and this pipeline is not
# reproducible run to run. These are qb2 jobs; the whglx quiet window does not reach them.
set -u
P512=$1     # pid of the running 512 draw; its cores go to the second 1536 draw
P1536=$2    # pid of the running 1536 draw; its cores go to the second 512 draw
cd "$HOME/bgref-work"

# Waits on the pid AND on its cmdline still naming the output dir it was launched for. A bare
# `kill -0` would wait forever if the pid were recycled to somebody else's process on a box
# this busy; the cmdline check makes the wait end when THIS run ends, whatever the pid does.
slot () {
    pid=$1; watch=$2; size=$3; label=$4
    while [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "$watch"; do
        sleep 120
    done
    echo "[chain] $label: $watch run ended at $(date -u +%FT%TZ), starting ${size} draw b"
    sleep 30
    "$HOME/bgref-work/run_valid_b.sh" "$size" 6
    echo "[chain] $label: finished at $(date -u +%FT%TZ)"
}

nohup setsid bash -c "$(declare -f slot); slot $P512 out512_valid 1536 slot-A" \
      </dev/null >> "$HOME/bgref-work/chain_1536b.log" 2>&1 &
nohup setsid bash -c "$(declare -f slot); slot $P1536 out1536_valid 512 slot-B" \
      </dev/null >> "$HOME/bgref-work/chain_512b.log" 2>&1 &
sleep 1
echo "[chain] armed $(date -u +%FT%TZ): slot-A waits on $P512 -> out1536_valid_b; slot-B waits on $P1536 -> out512_valid_b"
