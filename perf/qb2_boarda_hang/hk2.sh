#!/bin/bash
# hk2.sh A|B TAG [ITS]: rerun cfd3-qb2's D3A leg (Small k2 1x2, D2A state in, 1100 its, same image and runtime flags)
# on one qb2 board, to tell a chip hang from a state-dependent software stall. Card take, reset and smoke are
# run_k2.sh's (bounded drain, nothing killed). Adds a stall watch: prog.py reads each chip's global-reduction counter
# over the NoC every 30 s without halting anything; if neither chip's counter moves for 300 s the run is stalled, prog.py
# samples every core's RISC PCs off the debug bus (no halt) and the run is TERMed. Otherwise it ends on its own or at
# the TERM cap (CAP s, D2A took 3,836 s). Never tt-triage dump_callstacks a live run: it halts BRISC to unwind and on
# 2026-10-16 left r1B's BRISCs halted, which froze a healthy run and raised ABORT on both chips.
# Writes ~/qb2-boarda-hang/hk2_<TAG>.status (+ .log, aiclk, .prog). STOP: touch ~/qb2-boarda-hang/STOP.
W=$HOME/qb2-boarda-hang; D=$HOME/cfd3/qb2; M=$D/code_bhf/resident/mesh; PY=$HOME/scratch/shipflow/venv-4e80089afdf7/bin/python
B=$1; T=$2; ITS=${3:-1100}; CAP=${CAP:-9000}
case $B in A) C="0 1"; port=50090;; B) C="2 3"; port=50092;; *) echo "usage: $0 A|B TAG [ITS]"; exit 2;; esac
CC=${C/ /,}; c1=${C% *}; c2=${C#* }; L=$W; O=$W/hk2_$T.status; IV=$D/img_small_k2_d0c8f_gw1
SIN=$D/logs/qb2k2_D2A_case_1x2_state.npz; FJ=$HOME/spd/feed/jobs; until=$(date -d ${UNTIL:-2026-10-17T06:00:00Z} +%s)
st() { echo "$(date -u +%FT%TZ) $*" >> $O; }
quit() { { [ -e $W/STOP ] || [ $(date +%s) -ge $until ]; } && { st "gave up ($1)"; exit 0; }; }
FL="--arch bh --mesh 1,2 --layout rcb --l1 0 --wplace 2 --gw 1 --nsh 1 --page 16 --nwin 8192 --tabstream --amg --amg-stack --amg-vec --amg-smoothed 0 --amg-min-core 10 --lsched --phase --sliced --fuse --cse 1 --hlate 1 --vpre 1 --vpost 1 --strips --forces $HOME/cfd2/acc/small_fc_controlDict:forceCoeffsAll --case $D/small_tt"
XV="--amg-depth 0 --vcoarse 8"; R="--gcache 6 --bgts 5 --tgts 2"
st "armed pid $$ board $B chips $CC tag $T its $ITS code $(cat $D/code_bhf/COMMIT)"
until [ -e $IV.ready ]; do quit "image"; sleep 60; done
st "image $(md5sum < $IV/digests.json | cut -c1-8) ready $(stat -c %y $IV.ready | cut -c1-19)"
exec 6>/home/ttuser/spd_qb2_chip$c1.lock 7>/home/ttuser/spd_qb2_chip$c2.lock
S=""; SAMP=""; trap 'kill $S $SAMP 2>/dev/null' EXIT
REL="scripts/release_gate.py|gate_fanout -c"
nap() { sleep $1 6>&- 7>&-; }
drop() { flock -u 6; flock -u 7; kill $S 2>/dev/null; S=""; }
busy() { local j; for j in $FJ/*.card; do [ -e ${j%.card}.end ] && continue; grep -qx "$1" $j || continue
  kill -0 $(cat ${j%.card}.pid 2>/dev/null) 2>/dev/null && { echo $(( $(date +%s) - $(stat -c %Y $j) )); return; }; done; }
sent() { ( exec -a "qb2-boarda-hang spd_qb2_chip$1.lock release_gate hk2$T" sleep 86400 ) > /dev/null 6>&- 7>&- & echo $!; }
fd() { [ $1 = $c1 ] && echo 6 || echo 7; }
take2() { local x y ex ey t
  while pgrep -f "$REL" > /dev/null; do quit "release leg alive"; nap 120; done
  ex=$(busy $c1); ey=$(busy $c2); x=$c1; y=$c2; [ ${ey:-0} -gt ${ex:-0} ] && { x=$c2; y=$c1; }
  S=$(sent $x); st "stage 1: sentinel chip $x ($S), feed job elapsed $c1:${ex:-free} $c2:${ey:-free}"
  until flock -n $(fd $x); do quit "card take"; pgrep -f "$REL" > /dev/null && { drop; st "release leg appeared"; return 1; }; nap 30; done
  S="$S $(sent $y)"; t=$(date +%s); st "stage 2: holds chip $x, sentinel chip $y (feed job elapsed $(busy $y))"
  until flock -n $(fd $y); do quit "card take"; pgrep -f "$REL" > /dev/null && { drop; st "release leg appeared"; return 1; }
    [ $(( $(date +%s) - t )) -ge 600 ] && { drop; st "chip $y not free 600 s after chip $x, gave chip $x back"; return 1; }; nap 30; done
  kill $S 2>/dev/null; S=""; st "holds chips $CC, sentinels down"; }
until take2; do quit "retake"; done
tel() { for n in $*; do printf 'node%s aiclk=%s hb=%s ' $n "$(cat /sys/class/tenstorrent/tenstorrent!$n/tt_aiclk)" "$(cat /sys/class/tenstorrent/tenstorrent!$n/tt_heartbeat)"; done; }
while fuser -s /dev/tenstorrent/$c1 /dev/tenstorrent/$c2 2>/dev/null; do quit "device still open"; nap 30; done
st "held $CC: $(tel $C) fw $(for n in $C; do cat /sys/class/tenstorrent/tenstorrent!$n/tt_fw_bundle_ver; done | tr '\n' ' ')"
export TT_MESH_GRAPH_DESC_PATH=$(ls $(dirname $PY)/../lib/python3*/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto)
LP=$W/lp_$T; mkdir -p $LP
BENV="TT_VISIBLE_DEVICES=$CC CFD_TT_NODES=$CC TT_METAL_CACHE=$D/cache_$B"
IENV="TT_METAL_LOGS_PATH=$LP TT_METAL_INSPECTOR=1 TT_METAL_INSPECTOR_RPC=1 TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS=localhost:$port"
timeout -s INT 300 $HOME/.local/bin/tt-smi -r /dev/tenstorrent/$c1 /dev/tenstorrent/$c2 > $W/reset_$T.log 2>&1
st "reset rc $? $(tail -n1 $W/reset_$T.log | cut -c1-120)"; nap 10
( cd $D && env $BENV nice -n 10 timeout -s TERM 300 $PY -u fabsmoke2.py > $W/smoke_$T.log 2>&1 ); r=$?
st "smoke rc $r"; [ $r = 0 ] || exit 1
AL=$W/hk2_aiclk_$T.log
( while :; do echo "$(date -u +%T) aiclk $(for n in $C; do cat /sys/class/tenstorrent/tenstorrent!$n/tt_aiclk; done | tr '\n' ' ') power_uW $(for n in $C; do cat /sys/class/tenstorrent/tenstorrent!$n/device/hwmon/*/power1_input; done | tr '\n' ' ')"; sleep 10; done ) 6>&- 7>&- >> $AL 2>&1 & SAMP=$!
PG=$W/hk2_$T.prog
st "launch: case $ITS $XV $R, state-in $(basename $SIN), cap $CAP s, stall = reduction counter frozen 300 s"
( cd $M && env $BENV $IENV /usr/bin/time -f 'process wall %e s maxrss %M KB' nice -n 10 timeout -s TERM $CAP $PY -u run_mesh.py case $ITS \
    $FL --runs 1 --noemu --nwin 1024 --links 0 --rlink 1 --os 1 $XV --from-images $IV $R --state-in $SIN --out $W/hk2_$T.jsonl > $W/hk2_$T.log 2>&1 6>&- 7>&- ) & RP=$!
t0=$(date +%s); stalled=""
sleep 60 6>&- 7>&-
( env TT_VISIBLE_DEVICES=$CC ~/triage_venv/bin/python -I $W/prog.py $PG 30 300 ) > $PG.err 2>&1 6>&- 7>&- & PM=$!
trap 'kill $S $SAMP $PM 2>/dev/null' EXIT
while kill -0 $RP 2>/dev/null; do nap 30
  [ -z "$stalled" ] && grep -q STALL $PG 2>/dev/null || continue
  stalled=1; st "STALL: reduction counters frozen 300 s at +$(( $(date +%s) - t0 )) s: $(grep -m1 STALL $PG); PCs in $PG.pcs"
  for i in $(seq 60); do [ $(cat $PG.pcs 2>/dev/null | wc -l) -ge 1100 ] && break; nap 5; done
  st "TERM"; for p in $(pgrep -f "hk2_$T.jsonl"); do kill -TERM $p 2>/dev/null; done
done
kill $PM 2>/dev/null
wait $RP; r=$?; st "rc $r $(grep 'process wall' $W/hk2_$T.log) $(grep -h '^run .*wall' $W/hk2_$T.log | cut -c1-200)"
for n in $C; do st "after: $(tel $n)"; done
[ -n "$stalled" ] && { timeout -s INT 300 $HOME/.local/bin/tt-smi -r /dev/tenstorrent/$c1 /dev/tenstorrent/$c2 > $W/reset2_$T.log 2>&1; st "post-stall reset rc $? $(tel $C)"; }
[ -s $W/hk2_${T}_case_1x2_state.npz ] && $PY -c "import numpy as np,sys;open(sys.argv[2],'w').write(str(np.load(sys.argv[1])['hist']))" $W/hk2_${T}_case_1x2_state.npz $W/hk2_$T.hist.json
st "done, releasing chips $CC"
