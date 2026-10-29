#!/bin/bash
# fg256: lever trimul_t1_fgate (T1's gate fused in DST) at Protenix-v2's T1 key (8, 16, 8), c_z 256 hidden 256,
# bfp8 weights, packer sigmoid, unmasked (the trunk passes no mask at unpadded sizes). base / fg alternated 3x per
# fidelity, 20 reps each, f64 error and median ms per leg. qb1: logical card -> node 0:1 1:2 2:3 3:0.
#   perf/spd_m_protenix/fg256/run.sh <logical card> <node>     (run from the tree)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/perf/spd_m_protenix/fg256/out; VENV=$(ls -dt $HOME/scratch/shipflow/venv-*/done | head -1); PY=${VENV%done}bin/python
C=${1:?card}; NODE=${2:?node}; H=$(hostname -s | sed 's/tt-quietbox2/qb2/;s/tt-quietbox/qb1/')
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:spd-m-protenix-3
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait booking card$C"
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = spd-m-protenix-3 ]; do
  python3 $HOME/spd/feed/book.py add spd-m-protenix-3 $C 1800 --note fg256 >/dev/null 2>&1; sleep 30; done
exec 9>"$HOME/spd_${H}_card$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$NODE" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE 2>/dev/null)"
( while :; do echo "$(date -u +%T) $(cat "/sys/class/tenstorrent/tenstorrent!$NODE/tt_aiclk" 2>/dev/null) $(cut -d" " -f1 /proc/loadavg)" >> $O/aiclk.log; sleep 1; done ) 9>&- &
SP=$!
R="--c 256 --hidden 256 --w8 --no-mask --ns 512 --reps 20"
for fid in HiFi2 HiFi3; do for i in 1 2 3; do
  for leg in base fg; do
    S="T1_PSIG=1"; [ $leg = fg ] && S="$S,T1_FG=1"
    PYTHONPATH=$W timeout 900 $PY perf/spd_trimul/in_t1.py $R --set $S --fid $fid --out $O/${leg}_$fid.jsonl >> $O/${leg}_$fid.log 2>&1
    stamp "$leg $fid $i rc=$?"
  done; done; done
kill $SP; stamp done
python3 $HOME/spd/feed/book.py rm spd-m-protenix-3 >/dev/null 2>&1
