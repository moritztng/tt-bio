#!/usr/bin/env bash
# Two-host leg of perf/r4t_scale/xhost_hier.py on two LAN Galaxy hosts lent to r4t (glx_elastic).
# Run from pc (ssh reaches ubuntu@172.16.102.*). Host only: no chips opened.
#   perf/r4t_scale/run_xhost_pair.sh 110 107 OUTDIR [LOCALS="16 32"]
# Ships the branch's tt_bio + harness (git archive of HEAD from qb2's worktree) to ~/r4t-xh/<sha>
# on both, refuses unless both hosts are quiet (loadavg < 6, no perf bench), then per width:
# a PASS arm (8 steps) and a broken arm (host 1 --broken, must FAIL). Results copied to OUTDIR.
set -euo pipefail
A=$1 B=$2 OUT=$3 LOCALS=${4:-"16 32"}
SSHO=(-o BatchMode=yes -o ConnectTimeout=15)
WT=/home/ttuser/.coworker/wt/r4t-scale
mkdir -p "$OUT"
SHA=$(ssh "${SSHO[@]}" ttuser@tt-quietbox2 "git -C $WT rev-parse --short=9 HEAD")
ssh "${SSHO[@]}" ttuser@tt-quietbox2 "git -C $WT archive --format=tar HEAD tt_bio perf/r4t_scale/xhost_hier.py" > "$OUT/src.tar"
D=r4t-xh/$SHA
for h in $A $B; do
  ssh "${SSHO[@]}" ubuntu@172.16.102.$h "mkdir -p ~/$D && tar -C ~/$D -xf -" < "$OUT/src.tar"
done
quiet() {  # loadavg 1-min < 6 and no perf bench on the host
  ssh "${SSHO[@]}" ubuntu@172.16.102.$1 'l=$(cut -d" " -f1 /proc/loadavg); b=$(pgrep -fc "perf/.*(bench|ab)\.py|pytest" || true); echo "$l $b"; awk -v l="$l" -v b="$b" "BEGIN{exit !(l<6 && b==0)}"'
}
for h in $A $B; do
  if ! q=$(quiet $h); then echo "host .$h not quiet ($q): refusing, nothing run" | tee "$OUT/REFUSED"; exit 3; fi
  echo ".$h quiet: $q" | tee -a "$OUT/progress"
done
PY='$(ls -td ~/spd/spd-shipflow/venv-*/bin/python 2>/dev/null | head -1 || echo python3)'
TOK=$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')
run_arm() {  # $1 local width, $2 tag, $3 extra args for host 1
  local L=$1 tag=$2 extra=$3
  ssh "${SSHO[@]}" ubuntu@172.16.102.$A "cd ~/$D && P=$PY; PYTHONPATH=. TT_VISIBLE_DEVICES= timeout 900 \$P perf/r4t_scale/xhost_hier.py --host 0 --local $L --peer 172.16.102.$B --bind 172.16.102.$A --token $TOK --out h0_$tag.json > h0_$tag.log 2>&1; echo rc=\$?" > "$OUT/h0_$tag.rc" &
  local p0=$!
  sleep 3
  ssh "${SSHO[@]}" ubuntu@172.16.102.$B "cd ~/$D && P=$PY; PYTHONPATH=. TT_VISIBLE_DEVICES= timeout 900 \$P perf/r4t_scale/xhost_hier.py --host 1 --local $L --peer 172.16.102.$A --token $TOK $extra --out h1_$tag.json > h1_$tag.log 2>&1; echo rc=\$?" > "$OUT/h1_$tag.rc" || true
  wait $p0 || true
  scp -q "${SSHO[@]}" ubuntu@172.16.102.$A:"$D/h0_$tag.*" "$OUT/" || true
  scp -q "${SSHO[@]}" ubuntu@172.16.102.$B:"$D/h1_$tag.*" "$OUT/" || true
  echo "$tag h0 $(cat "$OUT/h0_$tag.rc") h1 $(cat "$OUT/h1_$tag.rc") $(date -u +%FT%TZ)" | tee -a "$OUT/progress"
}
for L in $LOCALS; do
  run_arm "$L" "L$L" ""
  run_arm "$L" "L${L}_broken" "--broken"
done
for h in $A $B; do ssh "${SSHO[@]}" ubuntu@172.16.102.$h "uptime" | sed "s/^/.$h after: /" | tee -a "$OUT/progress"; done
echo "sha $SHA DONE $(date -u +%FT%TZ)" | tee -a "$OUT/progress"
