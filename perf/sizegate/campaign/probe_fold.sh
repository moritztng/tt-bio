#!/bin/bash
# One fold at one rung on one card, with the two things the ladder recorder does not keep:
# a timestamp on every progress line and AICLK sampled DURING the fold.
#
# That pair is what separates "this rung is slow" from "this rung stalls". A slow fold prints
# steadily and its clock stays up; a stalled one goes quiet with the clock still boosted, and
# the recorder's only output for both is the same timeout.
#
# Args: <card> <model> <rung> [timeout_s]
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY=/home/ttuser/kisoji_p2_fresh/env/bin/python3
CARD=$1; MODEL=$2; RUNG=$3; TMO=${4:-7200}
OUT=$WT/perf/sizegate/probe
mkdir -p "$OUT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BASE=$OUT/$MODEL-$RUNG-card$CARD-$STAMP

"$PY" "$WT/perf/sizegate/campaign/card_health.py" "$CARD" | tee "$BASE.health" || exit 3

# AICLK sampler: resolve the node by BDF, never by UMD id (they differ on this host).
BDF=$("$PY" -c "import sys;sys.path.insert(0,'$WT/perf/sizegate/campaign');import card_health as c;print(c.bdf_for_umd($CARD))")
NODE=$("$PY" -c "import sys;sys.path.insert(0,'$WT/perf/sizegate/campaign');import card_health as c;print(c.node_for_bdf('$BDF'))")
( while :; do echo -e "$(date -u +%FT%TZ)\t$(cat $NODE/tt_aiclk 2>/dev/null || echo NA)"; sleep 15; done ) > "$BASE.aiclk" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

echo "probe $MODEL rung $RUNG card $CARD ($BDF, $NODE) timeout ${TMO}s start $(date -u +%FT%TZ)" | tee "$BASE.log"
START=$(date +%s)
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=0,$CARD \
TT_BIO_LEASE_HOLDER=worker:cov-ladder-p150a-p2 PYTHONPATH="$WT" \
  timeout "$TMO" stdbuf -oL -eL "$PY" -u -m tt_bio.main predict \
    "$WT/perf/size512/fixtures/cdk2x2_$RUNG.yaml" --model "$MODEL" --single_sequence \
    --sampling_steps 6 --diffusion_samples 1 --seed 0 \
    --out_dir "$BASE.out" 2>&1 \
  | stdbuf -oL awk '{ "date -u +%FT%TZ" | getline t; close("date -u +%FT%TZ"); print t"\t"$0; fflush() }' \
  | tee -a "$BASE.log"
RC=${PIPESTATUS[0]}
echo "probe rc=$RC wall=$(( $(date +%s) - START ))s end $(date -u +%FT%TZ)" | tee -a "$BASE.log"
kill $SAMPLER 2>/dev/null
echo "clock during fold: $(sort -k2 -n "$BASE.aiclk" | awk -F'\t' '{print $2}' | sort -n | uniq -c | tr '\n' ' ')" | tee -a "$BASE.log"
exit $RC
