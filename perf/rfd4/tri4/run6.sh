#!/bin/bash
# tri4 run6: the 31-seed oracle-free grade at 512 behind default-on. Seeds 10-30 for base and the stack
# (psig + hifi2 + t1w8), pooled with run4 seeds 0-9 (compile run r0 excluded), then perf/rfd4/quality.py
# (binder, land5 floor doc) and grade.py per seed.   perf/rfd4/tri4/run6.sh <card>
set -u
W=/home/ttuser/.coworker/wt/rfd4-perf-tri4; O=$W/artifacts/tri4/run6; R4=$W/artifacts/tri4/run4/design
PY=$HOME/tt-bio-dev/env/bin/python; C=${1:?card}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri4 PYTHONPATH=$W:$W/perf
mkdir -p $O/design $O/pool/base $O/pool/stack $O/grade; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
exec 9>"$HOME/spd_qb2_chip$C.lock"
stamp "wait flock chip$C"
flock -w 10800 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start $(git log --oneline -1 | cut -c1-70) dirty=$(git status --short tt_bio | wc -l)"
B=trimul_t1,pair_mm,pair_mm_tr,transition_b8,trimul_gin,trimul_gin_w8,pair_block,pair_tail,trimul_t1_dram,pair_bias_rms,pair_write_res
leg() {  # leg <name> <levers>
  TT_BIO_LEVERS="$2" timeout 5400 $PY perf/rfd4/e2e/design.py --spec $HOME/rfd4-inputs/pdl1-512.yaml \
      --weights $HOME/rfd4-bundle --out $O/design/$1 --rng upstream --seeds $(seq 10 30) > $O/design/$1.log 2>&1
  stamp "design $1 rc=$?"
}
leg base $B
leg stack $B,trimul_t1_psig,trimul_gin_hifi2,trimul_t1w8
exec 9>&-
for a in base stack; do s=$([ $a = base ] && echo base || echo ph2w8)
  for k in $(seq 0 9); do ln -sf $R4/$s/pdl1_binder_512_s${k}_r$((k+1))_generated.cif $O/pool/$a/; done
  for k in $(seq 10 30); do ln -sf $O/design/$a/pdl1_binder_512_s${k}_r$((k-9))_generated.cif $O/pool/$a/; done
done
stamp "pool base $(ls $O/pool/base | wc -l) stack $(ls $O/pool/stack | wc -l) dangling $(find -L $O/pool -type l | wc -l)"
f() { ls $O/pool/$1/pdl1_binder_512_s$2_r*_generated.cif; }
for k in $(seq 0 30); do $PY perf/rfd4/e2e/grade.py --ours $(f stack $k) --upstream $(f base $k) \
    --floor $(f base $(( (k+1)%31 ))) --out $O/grade/s$k.json >/dev/null 2>&1; done
cp $HOME/scratch/rfd4-land5/QUALITY.base.json $O/quality512.json
$PY perf/rfd4/quality.py grade --feature binder --tt $O/pool/stack --ref $O/pool/base --allow-no-oracle \
  --label "main default + trimul_t1_psig + trimul_gin_hifi2 + trimul_t1w8 vs main default, 512 PD-L1 binder, 31 seeds" \
  --out $O/quality512.json > $O/quality512.log 2>&1
stamp "quality rc=$? $(tail -1 $O/quality512.log | cut -c1-200)"
stamp end
