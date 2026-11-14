#!/bin/bash
# z8 run1: the pair stream as bfloat8_b (`pair_bfp8z`) on top of RFD4 default. z / s vs float64 at 256 and 512
# (bf16z/zprec.py), the paired traced trunk A/B at 512 / 384 / 256 (bfp8/stack_ab.py, 30 reps, AICLK during), then
# 31 seeds at 512 per arm graded per seed (grade.py) and as a population (quality.py).
# Takes the first free qb2 card (book.py + chip flock; release gates keep priority), waits up to 10 h.
#   RUN=runN perf/rfd4/z8/run1.sh [legs]
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/perf/rfd4/z8/${RUN:-run1}; PY=$HOME/tt-bio-dev/env/bin/python
LEGS=${1:-zprec256 zprec512 ab512 ab384 ab256 design grade}
mkdir -p $O/design $O/grade; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait any card, legs $LEGS"
end=$(( $(date +%s) + 36000 )); C=
while [ -z "$C" ] && [ $(date +%s) -lt $end ]; do
  for c in 1 2 0 3; do
    [ -n "$(python3 $HOME/spd/feed/book.py active $c --except rfd4-perf-z8 2>/dev/null)" ] && continue
    exec 9>"$HOME/spd_qb2_chip$c.lock"
    flock -n 9 || { exec 9>&-; continue; }
    fuser "/dev/tenstorrent/$c" >/dev/null 2>&1 && { exec 9>&-; continue; }
    C=$c; break
  done
  [ -z "$C" ] && sleep 30
done
[ -z "$C" ] && { stamp TIMEOUT; exit 3; }
python3 $HOME/spd/feed/book.py add rfd4-perf-z8 $C 10800 --now --note "z8 zprec, A/B, 31-seed grade" >/dev/null
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-z8 PYTHONPATH=$W:$W/perf
stamp "card $C start $(cat TREE 2>/dev/null)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(\",\".join(sorted(N)))")
S=$B,pair_bfp8z
echo "base $B" >> $O/run.log
f() { ls $O/design/$1/pdl1_binder_512_s$2_r*_generated.cif | tail -1; }   # s0 also has the cold run r0
for leg in $LEGS; do
  case $leg in
    zprec*) n=${leg#zprec}; timeout 3600 $PY perf/rfd4/bf16z/zprec.py --n $n --lever pair_bfp8z --out $O/$leg.json > $O/$leg.log 2>&1 ;;
    ab*) n=${leg#ab}; TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle \
           --out $O/$leg.json --n $n --reps 30 --base $B --levers $S > $O/$leg.log 2>&1 ;;
    design) for a in base stack; do L=$([ $a = base ] && echo $B || echo $S)
              TT_BIO_LEVERS=$L timeout 5400 $PY perf/rfd4/e2e/design.py --spec $HOME/rfd4-inputs/pdl1-512.yaml \
                --weights $HOME/rfd4-bundle --out $O/design/$a --rng upstream --seeds $(seq 0 30) > $O/design/$a.log 2>&1
              stamp "design $a rc=$?"; done ;;
    grade) exec 9>&-; python3 $HOME/spd/feed/book.py rm rfd4-perf-z8
           for k in $(seq 0 30); do $PY perf/rfd4/e2e/grade.py --ours $(f stack $k) --upstream $(f base $k) \
               --floor $(f base $(( (k+1)%31 ))) --out $O/grade/s$k.json >/dev/null 2>&1; done
           cp $HOME/scratch/rfd4-land5/QUALITY.base.json $O/quality512.json
           $PY perf/rfd4/quality.py grade --feature binder --tt $O/design/stack --ref $O/design/base --allow-no-oracle \
             --label "RFD4 default + pair_bfp8z vs default, 512 PD-L1 binder, 31 seeds" \
             --out $O/quality512.json > $O/quality512.log 2>&1 ;;
  esac
  rc=$?; stamp "$leg rc=$rc"
  # z over 0.05 vs float64 is a precision trade the structure grade decides. Over 0.10, or 512 off 256 by a third,
  # is a wrong program: stop before the designs spend the card.
  case $leg in zprec*)
    [ $rc != 0 ] && { stamp "ABORT: the lever does not run"; break; }
    $PY - "$O" <<'EOF' || { stamp "ABORT: z wrong"; break; }
import glob, json, sys
z = [json.load(open(f))["pair_bfp8z"]["z"] for f in glob.glob(sys.argv[1] + "/zprec*.json")]
sys.exit(max(z) >= 0.10 or max(z) > 1.33 * min(z))
EOF
    ;;
  esac
done
exec 9>&-; python3 $HOME/spd/feed/book.py rm rfd4-perf-z8
stamp ALLDONE
