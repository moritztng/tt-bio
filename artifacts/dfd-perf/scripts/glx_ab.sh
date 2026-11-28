#!/bin/bash
# dfd-perf on anx-glx chip 31 (node 23), D-dfd-orch-114: ab1 (shared scatter divisor + gather slice
# count on the gated ligand) then ab2 (per-pose batched scatter probe). Replaces glx_ab1.sh/glx_ab2.sh:
# starts only outside ~/fleet/HOLDER.windows with room before the next window, node closed 3 min,
# and waits until 12:00Z for 1-min load under 32 (after that it runs and the timings are shares only).
# Hard stop 12:50Z. Load and AICLK recorded every 5 s.
chip=31; node=23; OUT=$HOME/dfd/out; W=$HOME/fleet/HOLDER.windows; cd $HOME/dfd/tt-bio || exit 1
[ -e $HOME/fleet/HANDBACK ] && exit 0
exec 9>$OUT/ab.runlock; flock -n 9 || exit 0
log() { echo "ab: $* $(date -u +%FT%TZ)" >> $OUT/ab.log; }
stop=$(date -ud 2026-11-28T12:50:00Z +%s); loadwait=$(date -ud 2026-11-28T10:20:00Z +%s)
room() {  # seconds until the next open window starts (0 inside one), capped by the hard stop
  local now=$(date +%s) r=$(( stop - now )) who a b rest s e
  while read -r who a b rest; do
    case "$who" in dfd*|"#"*|"") continue;; esac
    case "$rest" in *"CLOSED EARLY"*) continue;; esac
    s=$(date -ud "$a" +%s 2>/dev/null) && e=$(date -ud "$b" +%s 2>/dev/null) || continue
    [ $now -ge $e ] && continue
    [ $now -ge $(( s - 120 )) ] && { echo 0; return; }
    [ $(( s - 120 - now )) -lt $r ] && r=$(( s - 120 - now ))
  done < $W
  echo $r
}
log "armed"
closed=0
while :; do
  [ $(date +%s) -ge $(( stop - 900 )) ] && { log "no slot before the stop, giving up"; exit 0; }
  if fuser -s /dev/tenstorrent/$node 2>/dev/null; then closed=0; else closed=$(( closed + 1 )); fi
  ld=$(cut -d" " -f1 /proc/loadavg); ld=${ld%.*}
  if [ $closed -ge 9 ] && [ $(room) -ge 900 ] && { [ $ld -lt 32 ] || [ $(date +%s) -ge $loadwait ]; }; then break; fi
  sleep 20
done
export PYTHONPATH=$PWD DFD_CKPT=$HOME/dfd/data/ckpt/extracted OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TT_VISIBLE_DEVICES=$chip TT_BIO_LEASE_CARDS=$chip TT_BIO_LEASE_HOLDER=worker:dfd-perf
CLK=/sys/class/tenstorrent/tenstorrent!$chip/tt_aiclk
( while :; do echo "$(date -u +%FT%TZ) $(cat $CLK 2>/dev/null) $(cut -d" " -f1 /proc/loadavg)"; sleep 5; done ) > $OUT/ab.aiclk 2>/dev/null &
clk=$!
# the batched-scatter probe first: 20 min, and it decides whether a lever gets written at all
left=$(room); [ $left -gt 1200 ] && left=1200
log "ab2 start load $(cut -d" " -f1 /proc/loadavg) cap ${left}s"
timeout $left $HOME/fleet/env/bin/python $HOME/dfd/scatprobe.py --out $OUT/ab2_scatter.json >> $OUT/ab.log 2>&1
log "ab2 rc=$?"
left=$(room); [ $left -gt 2700 ] && left=2700
if [ $left -ge 600 ]; then
  log "ab1 start head $(git rev-parse --short HEAD) load $(cut -d" " -f1 /proc/loadavg) cap ${left}s"
  timeout $left $HOME/fleet/env/bin/python $HOME/dfd/abprobe.py --edge-tp tile \
      --workload /home/ttuser/scratch/dfd/wl/workload --out $OUT/ab1_tile.json >> $OUT/ab.log 2>&1
  log "ab1 rc=$?"
else log "ab1 skipped, ${left}s left"; fi
kill $clk 2>/dev/null
log "aiclk during: $(cut -d" " -f2 $OUT/ab.aiclk | sort -n | uniq -c | tr "\n" " ")"
log done
