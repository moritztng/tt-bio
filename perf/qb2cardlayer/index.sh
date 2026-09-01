#!/usr/bin/env bash
# index.sh — rebuild draws.tsv from logs/, so the table can never disagree with the runs.
# The parent task (qb2-new-hardware-baseline-crosscheck) shipped a median-of-5 note resting on a
# value that appeared in no log. Deriving the index from the logs removes that failure mode.
set -eu
OUT=/home/ttuser/.coworker/wt/qb2-card-layer-baseline-reseed/perf/qb2cardlayer
: > "$OUT/draws.tsv.new"
for f in $(ls -1 "$OUT/logs" | sort -t- -k1,1 | sort); do
  base=${f%.log}
  model=$(echo "$base" | sed -E "s/-card[0-9]+-[0-9TZ]+$//")
  card=$(echo "$base" | sed -E "s/.*-card([0-9]+)-[0-9TZ]+$/\1/")
  ts=$(echo "$base" | sed -E "s/.*-card[0-9]+-//")
  val=$(grep -oE "^\[$model\] [0-9.]+ " "$OUT/logs/$f" | awk "{print \$2}" | tail -1)
  gate=$(grep -E "^$model " "$OUT/logs/$f" | tail -1 | tr -s " ")
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$ts" "$model" "card$card" "${val:-NA}" "${gate:-NO-GATE-LINE}" "$f"
done | sort -k2,2 -k1,1 | awk -F"\t" '{ if ($2!=prev) {n=0; prev=$2} n++; printf "%s\tmodel=%s\t%s\tdraw=%d\tvalue=%s\tgate=%s\tlog=%s\n",$1,$2,$3,n,$4,$5,$6 }' > "$OUT/draws.tsv.new"
mv "$OUT/draws.tsv.new" "$OUT/draws.tsv"
wc -l < "$OUT/draws.tsv"
