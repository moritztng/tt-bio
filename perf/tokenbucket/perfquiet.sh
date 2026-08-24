#!/usr/bin/env bash
# Settle the protenix-v2 perf red properly. perf_regression.py prints no loadavg, so its output
# cannot be audited for contention after the fact (its "load 4s" is model load time, not loadavg).
# This wrapper stamps the load itself, around every run, and interleaves the OFF control with the
# ON arm so a drifting box is common-mode rather than attributed to the flip.
#
# Evidence this is needed: the same leg read -16.0% at loadavg 4.1 and -61.4% at loadavg ~15-23
# while rel068 ran its own perf_regression on another card. One of those is the flip; the other is
# a co-tenant.
set -u
: "${CARD:?set CARD}"
PAIRS=${PAIRS:-4}
WT=/home/ttuser/.coworker/wt/protenix-opendde-token-bucket-flip-measure
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm
LEASE="TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:protenix-opendde-token-bucket-flip-measure"
OUT=perf/tokenbucket/perfquiet
mkdir -p $OUT
: > $OUT/readings.tsv

# Wait for a band where a 810 ms/call measurement is not dominated by a co-tenant. Up to 40 min.
t0=$SECONDS
while [ $((SECONDS-t0)) -lt 2400 ]; do
  l=$(cut -d' ' -f1 /proc/loadavg)
  awk -v a="$l" 'BEGIN{exit !(a+0<=5.0)}' && { echo "$(date -Is) load $l, measuring"; break; }
  echo "$(date -Is) load $l, waiting for <= 5"; sleep 30
done

one() {  # one <tag> <env...>
  tag=$1; shift
  pre=$(cut -d' ' -f1 /proc/loadavg)
  env $LEASE PYTHONPATH=$WT ESM_ROOT=$ESM_ROOT "$@" \
    "$PY" -u scripts/perf_regression.py --model protenix-v2 > "$OUT/$tag.log" 2>&1
  post=$(cut -d' ' -f1 /proc/loadavg)
  v=$(grep -aoE 'structures/s\s+[0-9.]+\s+[0-9.]+' "$OUT/$tag.log" | tail -1 | awk '{print $3}')
  printf '%s\t%s\t%s\t%s\n' "$tag" "${v:-NA}" "$pre" "$post" | tee -a $OUT/readings.tsv
}

for i in $(seq 0 $((PAIRS-1))); do
  # alternate within-pair order so a monotone drift cannot bias the sign
  if [ $((i % 2)) -eq 0 ]; then
    one "off_$i" TT_BIO_PROTENIX_TOKEN_BUCKET=0
    one "on_$i"  TT_BIO_PROTENIX_TOKEN_BUCKET=1
  else
    one "on_$i"  TT_BIO_PROTENIX_TOKEN_BUCKET=1
    one "off_$i" TT_BIO_PROTENIX_TOKEN_BUCKET=0
  fi
done
echo "=== perfquiet done $(date -Is)"
cat $OUT/readings.tsv
