#!/bin/bash
# proga-review fallback run (authorized in state/ask-16668-decision.md): drive tt-bio with the
# EXACT command line proga's TTBioOracle.predict_complex builds (ttbio_oracle.py:49-61), and the
# EXACT yaml/a3m its BoltzOracle._write_yaml writes (boltz_oracle.py:69-82), without executing
# proga's own driver on the fleet. Two candidates per model share one target, so the
# content-addressed a3m is written once and reused -- that is proga's MSA cache across a population.
# Usage: run_fallback.sh <logical_card> <outdir> <per_fold_timeout_s>
set -u
CHIP=$1 OUT=$2 TMO=$3
R=$HOME/proga-review
TREE=${TREE:-$HOME/spd/spd-gap-small/tm1e3b37664}
VENV=${VENV:-$(ls -dt $HOME/scratch/shipflow/venv-*/ | head -1)}; VENV=${VENV%/}
mkdir -p "$OUT" "$R/leases"
say() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/run.log"; }

exec 9> "/home/ttuser/spd_qb1_card$CHIP.lock"
say "waiting for card $CHIP flock"
flock -w "${WAIT:-10800}" 9 || { say "flock busy after ${WAIT:-10800}s"; exit 3; }
mkdir "$R/CLAIM_FB" 2>/dev/null || { say "another card won the race; releasing card $CHIP"; exit 0; }
say "got card $CHIP; tree $(cat $TREE/TREE_SHA) venv $VENV"
export TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=proga-review TT_BIO_LEASE_DIR=$R/leases

# every perf number here names the clock it was taken at: sample AICLK + load during the folds
( while :; do l=$(cut -d' ' -f1 /proc/loadavg); for n in /sys/class/tenstorrent/tenstorrent\!*; do
    printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "${n##*!}" "$(cat $n/tt_aiclk 2>/dev/null)" "$l"; done; sleep 5; done ) 9>&- >> "$OUT/aiclk.tsv" 2>/dev/null &
SAMP=$!

TGT=MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
B1=EKLRELAEKLWRQAEEMLKRAREN
B2=SWEEIAKRLHELAQKMLDEAKRLG
MSA=$OUT/msa; mkdir -p "$MSA"
a3m() { # content-addressed exactly like proga's _seq_hash: sha256(seq)[:16]
  h=$(printf '%s' "$1" | sha256sum | cut -c1-16); p=$MSA/$h.a3m
  if [ ! -e "$p" ]; then printf '>query\n%s\n' "$1" > "$p"; say "a3m MISS -> wrote $h.a3m"; else say "a3m HIT  -> reused $h.a3m"; fi
  echo "$p"
}
yaml() { # BoltzOracle._write_yaml, verbatim shape
  { echo "version: 1"; echo "sequences:";
    for cid_seq in "A:$2" "B:$TGT"; do cid=${cid_seq%%:*}; seq=${cid_seq#*:}
      echo "  - protein:"; echo "      id: $cid"; echo "      sequence: $seq"; echo "      msa: $(a3m "$seq")"; done; } > "$1"
}

for MODEL in boltz2 protenix-v2; do
  for C in 1 2; do
    eval "BSEQ=\$B$C"; NAME="cand${C}"
    RUN="$OUT/$MODEL/$NAME"; mkdir -p "$RUN"
    yaml "$RUN/$NAME.yaml" "$BSEQ"
    say "START $MODEL $NAME clk=$(cat /sys/class/tenstorrent/tenstorrent\!$CHIP/tt_aiclk 2>/dev/null)"
    t0=$(date +%s)
    ( cd "$RUN" && timeout -s INT "$TMO" env PYTHONPATH="$TREE" "$VENV/bin/python" -m tt_bio.main \
        predict "$RUN/$NAME.yaml" --out_dir "$RUN" --model "$MODEL" --write_pae \
        --output_format cif --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 --seed 0 ) \
      > "$RUN/predict.log" 2>&1
    rc=$?; t1=$(date +%s)
    say "END   $MODEL $NAME rc=$rc wall=$((t1-t0))s clk=$(cat /sys/class/tenstorrent/tenstorrent\!$CHIP/tt_aiclk 2>/dev/null)"
    echo -e "$MODEL\t$NAME\t$rc\t$((t1-t0))" >> "$OUT/walls.tsv"
    [ $rc -ne 0 ] && say "  (continuing to next candidate/model despite rc=$rc)"
  done
done
kill $SAMP 2>/dev/null
say "ALL DONE"; touch "$OUT/FINISHED"
