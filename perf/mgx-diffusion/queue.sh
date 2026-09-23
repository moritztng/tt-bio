#!/usr/bin/env bash
# queue.sh <tree>:<plan> ...  run each plan in order through <tree>/perf/mgx-diffusion/launch.sh
# on a free whglx chip, again on the next free chip until its ladder exits 0 (a chip another row
# takes between folds ends the ladder with 1, and it resumes from runs.jsonl).
# A chip is claimed with an atomic mkdir before launch, so two queues never pick the same one;
# the lease is still what refuses a collision with any other row.
set -u
here=$(cd "$(dirname "$0")" && pwd)
py=$HOME/env/bin/python
pick() {
    while :; do
        for c in $(cd "$here" && "$py" -c "import lanes; print(*[c for c in lanes.CARDS if lanes.free(c)])"); do
            mkdir "/tmp/mgxd-claim-$c" 2>/dev/null && { echo "$c"; return; }
        done
        sleep 20
    done
}
for tp in "$@"; do
    tree=${tp%%:*} plan=${tp#*:} name=$(basename "$plan" .txt)
    while :; do
        card=$(pick)
        echo "[$(date -u +%FT%TZ)] $tree $plan on $card"
        bash "$tree/perf/mgx-diffusion/launch.sh" "$card" "$tree/perf/mgx-diffusion/$plan"
        rmdir "/tmp/mgxd-claim-$card"
        rc=$(grep -o 'EXIT [0-9]*' "$tree/perf/mgx-diffusion/logs/$name-$card.log" | tail -1)
        echo "[$(date -u +%FT%TZ)] $rc"
        [ "$rc" = "EXIT 0" ] && break
        sleep 30
    done
done
