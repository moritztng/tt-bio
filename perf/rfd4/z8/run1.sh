#!/bin/bash
# z8 run1: the pair stream as bfloat8_b (`pair_bfp8z`) on top of RFD4 default. z / s vs float64 at 256 and 512
# (bf16z/zprec.py), then the paired traced trunk A/B at 512 / 384 / 256 (bfp8/stack_ab.py, 30 reps, AICLK during).
# Takes the first free qb2 card (book.py + chip flock; release gates keep priority), waits up to 6 h.
#   perf/rfd4/z8/run1.sh [legs]
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/z8/${RUN:-run1}; PY=$HOME/tt-bio-dev/env/bin/python
LEGS=${1:-zprec256 zprec512 ab512 ab384 ab256}
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait any card, legs $LEGS"
end=$(( $(date +%s) + 21600 )); C=
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
python3 $HOME/spd/feed/book.py add rfd4-perf-z8 $C 9000 --now --note "z8 run1 zprec + A/B" >/dev/null
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-z8 PYTHONPATH=$W:$W/perf
stamp "card $C start $(git log --oneline -1 | cut -c1-70)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(\",\".join(sorted(N)))")
echo "base $B" >> $O/run.log
for leg in $LEGS; do
  case $leg in
    zprec*) n=${leg#zprec}; timeout 3600 $PY perf/rfd4/bf16z/zprec.py --n $n --lever pair_bfp8z --out $O/$leg.json > $O/$leg.log 2>&1 ;;
    ab*) n=${leg#ab}; TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle \
           --out $O/$leg.json --n $n --reps 30 --base $B --levers $B,pair_bfp8z > $O/$leg.log 2>&1 ;;
  esac
  rc=$?; stamp "$leg rc=$rc"
  if [ "$leg" = zprec256 ]; then
    [ $rc != 0 ] && { stamp "ABORT: the lever does not run"; break; }
    $PY -c "import json,sys; r=json.load(open(\"$O/zprec256.json\")); sys.exit(r[\"pair_bfp8z\"][\"z\"] >= r[\"bar\"])" \
      || { stamp "ABORT: z over the bar"; break; }
  fi
done
exec 9>&-; python3 $HOME/spd/feed/book.py rm rfd4-perf-z8
stamp ALLDONE
