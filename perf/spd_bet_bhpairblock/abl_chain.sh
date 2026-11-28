#!/bin/bash
# Stage ablation of every Pairformer pair kernel at Protenix-v2's shape (512 tokens, c_z 256) on one Blackhole chip.
# Each arm skips a kernel's DRAM stages or its compute, so the op's time splits into data movement and math
# (SFPU included, which no FLOP roof prices). Outputs under an ablation are wrong on purpose; only time is read.
# usage: abl_chain.sh CHIP TREE OUT DEADLINE_UTC     (run.sh flocks the chip and samples its AICLK every second)
CHIP=$1 T=$2 O=$3 END=$4
export PATH=$HOME/fleet/env/bin:$PATH; cd "$T" || exit 1; mkdir -p "$O"
arm() {  # arm NAME [ENV=VAL] -- cmd...
  local n=$1 e=; shift; [ "$1" != -- ] && { e=$1; shift; }; shift
  local left=$(( $(date -u -d "$END" +%s) - $(date +%s) ))
  [ -e "$HOME/fleet/HANDBACK" ] && { echo "HANDBACK, stop" >> "$O/chain.log"; exit 0; }
  [ $left -lt 240 ] && { echo "skip $n: $left s left" >> "$O/chain.log"; return; }
  [ -e "$O/$n/done" ] && return
  uptime > "$O/$n.load"
  env $e bash perf/spd_gap_whlm/run.sh "$CHIP" "$O/$n" -- timeout -s INT $(( left < 600 ? left - 60 : 600 )) "$@" && touch "$O/$n/done"
  echo "$n rc=$(cat "$O/$n/rc") $(date -u +%T)" >> "$O/chain.log"
}
T1="python perf/spd_trimul/in_t1.py --ns 512 --c 256 --hidden 256 --abl 16,64,112,527,1,512"
arm t1    -- $T1 --out "$O/t1.jsonl"
arm t1g8  -- $T1 --set T1_G8=1 --out "$O/t1g8.jsonl"
arm t2    -- python perf/spd_trimul/tail_t2.py --ns 512 --c 256 --gate --abl 1,2,4,8,15,16,32,48 --out "$O/t2.jsonl"
arm tl    -- python perf/spd_m_protenix/tl_attr.py --n 512 --c 256
for a in "" EXP MASKADD KVREAD EXP,MASKADD,KVREAD; do
  n=ta-$(echo "${a:-base}" | tr , - | tr A-Z a-z)
  arm $n TT_BIO_TRIATT_ABLATE=$a -- python perf/spd_m_opendde/ta_bench.py "$O/$n.json" --seq 512 --heads 8 --tag "$n"
done
echo "CHAIN-END $(date -u +%FT%TZ)" >> "$O/chain.log"
