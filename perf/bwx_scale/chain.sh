#!/bin/bash
# The whole bwx-scale sitting, on a dev Galaxy box, detached: the trajectory price on ONE chip,
# then the same workload on TWO chips of the same Galaxy at once.
#
#   chain.sh <out root>
#
# Two phases, each costing the box one agent restart and holding its chips only for its own
# length. The agent is UP for both phases: it serves the other 29-30 chips throughout and
# re-canaries each chip on the way back in, which is where the before-and-after health of the
# chips this sitting used comes from.
#
# Phase two takes a second chip, so it needs its own handover -- the agent's chip worker owns
# that chip until it is stopped. Between the phases the first chip goes back to the agent and is
# canaried, which is the health check the campaign price is graded on.
set -uo pipefail
out=${1:?usage: chain.sh <out root>}
root=${BWX_ROOT:-$HOME/bwx-scale/tt-bio}
py=${BWX_PY:-$HOME/bwx/venv/bin/python}
params=${BWX_PARAMS:-$HOME/bwx/af2_params}
one=${BWX_CHIP_ONE:-30}
two=${BWX_CHIP_TWO:-29}
mkdir -p "$out"
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$out/chain.log"; }

. ~/japanfold/env.sh
export PYTHONPATH=$root
export BCX_BC2=${BCX_BC2:-$HOME/bwx/bc2}
export TT_METAL_CACHE=${TT_METAL_CACHE_BWX:-$HOME/bwx/cache/tt-metal}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$HOME/bwx/cache/xla}
export TT_BIO_LEASE_HOLDER=worker:bwx-scale

# Never leave a dev Galaxy down because this script died.
trap 'sudo -n systemctl start japanfold-agent@ubuntu; say "trap: agent start rc=$?"' EXIT

idle(){
    # The box's own answer to "is anything running here", which is the question /v1/cluster
    # answers one layer up. The agent logs `leased <id>: <kind>` when it takes a job and
    # `settled <id>` when it lets go, so an unsettled lease is a job in flight; a job leased
    # before the window would otherwise be invisible.
    local j leased settled recent hot
    j=$(sudo -n journalctl -u japanfold-agent@ubuntu --since "-6 hours" --no-pager 2>/dev/null)
    leased=$(grep -cE "\[agent\].* leased [0-9a-f]{8}" <<< "$j" || true)
    settled=$(grep -cE "\[agent\].* settled [0-9a-f]{8}" <<< "$j" || true)
    recent=$(sudo -n journalctl -u japanfold-agent@ubuntu --since "-10 min" --no-pager 2>/dev/null \
             | grep -cE "\[agent\].* leased [0-9a-f]{8}" || true)
    hot=$(ps -eo pcpu,args --no-headers | awk '/japanfold.chipworker/ && $1 > 50' | wc -l)
    say "idle check: leased $leased settled $settled (in flight $((leased-settled))), $recent in 10 min, $hot busy workers, load $(cut -d' ' -f1 /proc/loadavg)"
    [ "$((leased-settled))" -le 0 ] && [ "$recent" -eq 0 ] && [ "$hot" -eq 0 ]
}

wait_idle(){
    for _ in $(seq 60); do idle && return 0; say "box busy, waiting 60 s"; sleep 60; done
    say "box still busy after an hour, giving up rather than interrupting it"; return 1
}

phase(){
    local tag=$1 chips=$2
    wait_idle || return 1
    say "phase $tag chips=$chips: agent stop"
    sudo -n systemctl stop japanfold-agent@ubuntu; say "agent stop rc=$?"
    for _ in $(seq 90); do pgrep -f japanfold.chipworker >/dev/null || break; sleep 2; done
    mkdir -p "$out/$tag"
    setsid nohup "$py" -u "$root/perf/bwx_scale/scale.py" --chips "$chips" \
        --params "$params" --out "$out/$tag" > "$out/$tag/scale.log" 2>&1 < /dev/null &
    local pid=$!
    echo "$pid" > "$out/$tag/pid"; say "phase $tag pid $pid"
    # Back up as soon as the sitting owns every chip it asked for, so the box is down for the
    # handover and not for the phase.
    local want; want=$(awk -F, '{print NF}' <<< "$chips")
    for _ in $(seq 180); do
        local got=0 c
        for c in ${chips//,/ }; do
            grep -q "\"pid\": $pid" "$HOME/japanfold/state/leases/"*card"$c".json 2>/dev/null \
                && got=$((got+1))
        done
        [ "$got" -eq "$want" ] && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    sudo -n systemctl start japanfold-agent@ubuntu; say "agent start rc=$? after ${want} lease(s)"
    while kill -0 "$pid" 2>/dev/null; do sleep 10; done
    say "phase $tag done, scale.log tail: $(tail -1 "$out/$tag/scale.log")"
}

say "chain start root=$root commit=$(git -C "$root" rev-parse --short HEAD) chips $one then $one,$two"
phase one "$one" && phase two "$one,$two"
say "chain finished"
