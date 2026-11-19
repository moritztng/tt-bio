#!/bin/bash
# bat-lever: structural grade of a lever at PopVax's 9TH6 + 120-mer shape (620 sample tokens), paired seeds.
# Designs per arm on the card, then the CPU grade: designed-chain CA RMSD per seed against the base arm's design
# at the same seed, with the seed floor from the base arm at the NEXT seed, plus the oracle-free binder grade.
#   ARMS="base: mix:+mask_at_mix" SEEDS=15 bash perf/rfd4/bat/grade620.sh <chip> <outdir> [device|grade]
set -u
C=$1; R=$2; LEG=${3:-device}; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-8edb023e00db/bin/python}
W=$HOME/rfd4-bundle; SPEC=$WT/perf/pvo_gpu/rfd4/9TH6-A-b120.yaml; N=${SEEDS:-15}
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
if [ $LEG = device ]; then
  export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:bat-lever
  exec 9>"$HOME/spd_qb2_chip$C.lock"; flock -w 21600 9 || { stamp "flock timeout"; exit 3; }
  BASE=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(\",\".join(sorted(L)))" 2>/dev/null | tail -1)
  stamp "start chip $C sha $(git rev-parse --short HEAD) seeds 0..$N"
  for arm in ${ARMS:-base:}; do name=${arm%%:*}; ex=${arm#*:}
    L=",$BASE,"; for e in ${ex//,/ }; do case $e in -*) L=${L//,${e#-},/,};; +*) L="$L${e#+},";; esac; done
    L=${L#,}; L=${L%,}
    TT_BIO_LEVERS=$L timeout 21600 $PY perf/rfd4/e2e/design.py --spec $SPEC --weights $W --rng upstream \
      --seeds $(seq 0 $N) --out $R/$name > $R/$name.log 2>&1
    stamp "$name rc=$?"
  done
  stamp DEVICEDONE
else
  # CPU only, niced: pair arm against base seed for seed, floor = base at the next seed (land12's rule).
  OURS=${OURS:?arm dir}; REF=${REF:-base}
  s() { ls $R/$1/*_s$2_r*_generated.cif 2>/dev/null | tail -1; }
  mkdir -p $R/grade-$OURS $R/pool-$OURS/tt $R/pool-$OURS/ref
  for k in $(seq 0 $N); do
    o=$(s $OURS $k); r=$(s $REF $k); f=$(s $REF $(( (k+1) % (N+1) )))
    [ -z "$o" ] || [ -z "$r" ] || [ -z "$f" ] && { echo "seed $k missing a cif" >> $R/done; continue; }
    ln -sf $o $R/pool-$OURS/tt/; ln -sf $r $R/pool-$OURS/ref/
    nice -n 19 $PY perf/rfd4/e2e/grade.py --ours $o --upstream $r --floor $f \
      --out $R/grade-$OURS/s$k.json > /dev/null 2>&1 || echo "grade $OURS s$k failed" >> $R/done
  done
  nice -n 19 $PY perf/rfd4/quality.py grade --feature binder --tt $R/pool-$OURS/tt --ref $R/pool-$OURS/ref \
    --allow-no-oracle --label "bat-lever $OURS vs $REF, 9TH6 + 120-mer, $((N+1)) seeds" \
    --out $R/quality-$OURS.json > $R/quality-$OURS.log 2>&1
  stamp "grade $OURS rc=$?"
  $PY - "$R/grade-$OURS" <<'PY' >> $R/SUMMARY.txt
import json, sys, glob, statistics
rs = [json.load(open(f)) for f in sorted(glob.glob(sys.argv[1] + "/s*.json"))]
if rs:
    d = [r["rmsd_a"] for r in rs]; fl = [r["seed_floor_a"] for r in rs]
    print(f"{sys.argv[1]}: n={len(d)} designed-chain CA RMSD median {statistics.median(d):.2f} A "
          f"(max {max(d):.2f}) against a seed floor median {statistics.median(fl):.2f} A, "
          f"ratio {statistics.median(d)/statistics.median(fl):.3f} (kill line 0.5)")
PY
fi
