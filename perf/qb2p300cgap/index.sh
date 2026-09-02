#!/usr/bin/env bash
# index.sh — rebuild draws.tsv from logs/, so the table can never disagree with the runs.
# A median-of-N typed by hand next to the logs instead of computed from them is a documented past
# failure of this exact protocol (qb2-new-hardware-baseline-crosscheck, "Defect found and fixed").
set -eu
OUT=/home/ttuser/.coworker/wt/p300c-baseline-coverage-gap/perf/qb2p300cgap
: > "$OUT/draws.tsv.new"
for f in $(ls -1 "$OUT/logs" | sort); do
  base=${f%.log}
  model=$(echo "$base" | sed -E "s/-card[0-9]+-[0-9TZ]+$//")
  card=$(echo "$base" | sed -E "s/.*-card([0-9]+)-[0-9TZ]+$/\1/")
  ts=$(echo "$base" | sed -E "s/.*-card[0-9]+-//")
  val=$(grep -oE "^\[$model\] [0-9.]+ " "$OUT/logs/$f" | awk "{print \$2}" | tail -1)
  # No trailing space in the anchor: _print_table pads the model column to 16 chars, so
  # esmc-300m-single (17) runs straight into its metric and the row reads
  # "esmc-300m-singleseq/s". Anchoring on "^$model " lost the gate column for exactly the
  # legs this task is seeding. One model per log, so the looser anchor cannot cross-match.
  gate=$(grep -E "^$model" "$OUT/logs/$f" | tail -1 | tr -s " ")
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$ts" "$model" "card$card" "${val:-NA}" "${gate:-NO-GATE-LINE}" "$f"
done | sort -k2,2 -k1,1 | awk -F"\t" '{ if ($2!=prev) {n=0; prev=$2} n++; printf "%s\tmodel=%s\t%s\tdraw=%d\tvalue=%s\tgate=%s\tlog=%s\n",$1,$2,$3,n,$4,$5,$6 }' > "$OUT/draws.tsv.new"
mv "$OUT/draws.tsv.new" "$OUT/draws.tsv"
wc -l < "$OUT/draws.tsv"
