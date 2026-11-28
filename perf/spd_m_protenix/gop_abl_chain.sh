#!/bin/bash
# gated_out_proj stage ablation on one anx-glx Blackhole chip: one gop_abl.py process per arm, each under the chip's
# flock, never inside a whole-box window, only on an idle node.   usage: gop_abl_chain.sh CHIP TREE VENV OUT
set -u
CHIP=$1; T=$2; V=$3; O=$4; mkdir -p $O
NODE=$(for n in /sys/class/tenstorrent/tenstorrent\!*; do echo "$(basename "$(readlink -f "$n/device")") ${n##*!}"; done | sort | sed -n "$((CHIP + 1))p" | cut -d" " -f2)
say() { echo "$(date -u +%FT%TZ) $*" >> $O/chain.log; }
room() { local now s e r=99999 n a b _; now=$(date +%s); while read -r n a b _; do [ -z "${n:-}" ] || [ "${n:0:1}" = "#" ] && continue
  s=$(( $(date -u -d "${a%Z}:00Z" +%s) - 240 )); e=$(date -u -d "${b%Z}:00Z" +%s)
  [ $now -ge $e ] && continue; [ $now -ge $s ] && { echo 0; return; }; [ $((s - now)) -lt $r ] && r=$((s - now)); done < $HOME/fleet/HOLDER.windows; echo $r; }
for a in base READ WRITE READ,WRITE SIG MM SIG,MM READ,WRITE,SIG,MM base2; do
  n=$(echo $a | tr , - | tr A-Z a-z); [ -e $O/$n.done ] && continue
  for try in 1 2 3 4 5 6 7 8; do
    [ -e $HOME/fleet/HANDBACK ] || [ -e $O/STOP ] && { say "stop file"; exit 0; }
    while [ $(room) -lt 420 ]; do sleep 120; done
    (
      flock -w 3600 9 || exit 98
      [ $(room) -lt 360 ] && exit 97
      [ -n "$(cat /proc/driver/tenstorrent/$NODE/pids 2>/dev/null)" ] && exit 96
      say "start $n chip $CHIP node $NODE load $(cut -d" " -f1-3 /proc/loadavg)"
      rm -f $O/$n.jsonl
      cd $T && TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=spd-m-protenix-8 TT_METAL_LOGGER_LEVEL=FATAL LOGURU_LEVEL=WARNING OMP_NUM_THREADS=2 \
        PYTHONPATH=$T PATH=$V/bin:$PATH TT_BIO_TRIATT_TAIL_ABLATE=$([ ${a%2} = base ] || echo $a) \
        timeout -s KILL 330 timeout -s INT 300 python perf/spd_m_protenix/gop_abl.py $O/$n.jsonl --node $NODE > $O/$n.log 2>&1 9>&-
    ) 9> $HOME/spd/locks/chip$CHIP
    rc=$?; say "$n try $try rc=$rc"
    [ $rc = 0 ] && { touch $O/$n.done; break; }
    [ $rc -lt 96 ] && [ $try -ge 2 ] && break; sleep 45
  done
done
say CHAIN-END; touch $O/chain.done
