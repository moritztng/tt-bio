#!/bin/bash
# phase18.sh: which Tensix clock-gate block hangs qb2 chip 0 (ASIC B9017B91B319B0DE) at 1350 MHz. Phase 17 showed
# stock fixm pcG x1000 hangs at iteration 1 with fw 19.15's CG_CTRL_EN=0xFFFFFFFF and runs 3/3 clean with it 0.
# Bisect: enable only a subset of the 17 CG bits after each reset; a hang names a sufficient subset, recurse into it.
# Then confirm (culprit alone x2 hang; all-but-culprit x3 clean) and try the fw hysteresis (2) raised to 4/8/0x10
# with all blocks gated. Index phase18.log: "== <tag> cg=<hex> hyst=<hex>" + "<tag> verdict=HANG|CLEAN".
cd ~/spd/spd-nocdeadlock
until grep -q P17-DONE phase17.log 2>/dev/null; do sleep 30; done
for c in 0 1; do eval "exec $((8+c))>~/spd_qb2_chip$c.lock"; flock $((8+c)); done
echo "locks 0,1 held $(date -u +%FT%TZ)"
py=~/.tenstorrent-venv/bin/python; tree=~/spd/spd-nocdeadlock/fixm
$py tel.py p18_tel.csv 0.2 & tp=$!
cg() { TT_VISIBLE_DEVICES=0 timeout -s TERM 300 ~/triage_venv/bin/python cgctl.py "$@" 2>&1 | tail -1; }
n=0
leg() { # leg <cg hex> [hyst hex] -> sets V=HANG|CLEAN
  n=$((n+1)); tag=p18_$(printf %02d $n)
  timeout 300 ~/.local/bin/tt-smi -r 0,1 > /dev/null 2>&1; echo "reset 0,1 rc=$? $(date -u +%FT%TZ)"; sleep 15
  echo "cg set $1: $(cg set $1)"; [ -n "$2" ] && echo "cg hyst $2: $(cg hyst $2)"
  echo "== $tag cg=$1 hyst=${2:-02020202} start $(date -u +%FT%TZ)"
  ./repro.sh 0 $tag $tree --iters 1000 --every 50
  d=$(grep -c REPRO-DONE rp_${tag}_c0.log); V=CLEAN; [ "$d" = 1 ] || V=HANG
  echo "$tag verdict=$V cg=$1 hyst=${2:-02020202} last: $(grep "^it " rp_${tag}_c0.log | tail -1) end $(date -u +%FT%TZ)"
  echo "cg after run: $(cg read)"
}
hex() { printf %08x $1; }
# bisect over bit lists
bits=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
mask() { local m=0; for b in "$@"; do m=$((m | (1<<b))); done; echo $m; }
while [ ${#bits[@]} -gt 1 ]; do
  h=$(( ${#bits[@]} / 2 )); lo=("${bits[@]:0:$h}"); hi=("${bits[@]:$h}")
  leg $(hex $(mask "${lo[@]}")); if [ $V = HANG ]; then bits=("${lo[@]}"); continue; fi
  leg $(hex $(mask "${hi[@]}")); if [ $V = HANG ]; then bits=("${hi[@]}"); continue; fi
  echo "BISECT-SPLIT neither half of {${bits[*]}} hangs alone: interaction"; break
done
echo "BISECT-RESULT bits={${bits[*]}} $(date -u +%FT%TZ)"
cm=$(mask "${bits[@]}")
leg $(hex $cm); leg $(hex $cm)
rest=$(( 0x1FFFF & ~cm )); leg $(hex $rest); leg $(hex $rest); leg $(hex $rest)
for hy in 04040404 08080808 10101010; do leg ffffffff $hy; done
kill -TERM $tp; timeout 300 ~/.local/bin/tt-smi -r 0,1 > /dev/null 2>&1; echo "P18-DONE reset rc=$? $(date -u +%FT%TZ)"
