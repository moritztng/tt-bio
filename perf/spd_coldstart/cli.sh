#!/bin/bash
# Where a cold `tt-bio predict` call spends its wall, and what the persistent controller saves:
#   perf/spd_coldstart/cli.sh CHIP OUT MODEL [INPUT]
# 1. the CLI twice cold with trace/sitecustomize.py on PYTHONPATH (phase events of the parent and its spawned worker
#    in trace-<n>.jsonl); 2. `tt-bio controller` on the same chip, then the CLI three times with --controller (the
#    first loads the model, the next two reuse it); 3. for boltz2, one cold CLI against an empty TT_METAL_CACHE.
# Walls from /usr/bin/time; AICLK of every node each second in aiclk.log.
set -u
CHIP=$1 OUT=$2 M=$3 I=${4:-u100}
ROW=spd-bet-coldstart
mkdir -p "$OUT" ~/spd/locks ~/spd/$ROW/leases
[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=$ROW TT_BIO_LEASE_DIR=$HOME/spd/$ROW/leases
TRACE=$PWD/perf/spd_coldstart/trace
say() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/run.log"; }
exec 9> "${LOCK:-$HOME/spd/locks/chip$CHIP.lock}"
flock -w "${WAIT:-600}" 9 || { say "flock busy"; exit 3; }
say "start chip $CHIP tree $(cat TREE_SHA 2>/dev/null || git rev-parse --short HEAD) host $(hostname) model $M input $I"
D=$OUT/data; mkdir -p "$D/inputs"; ln -sfn ~/spd-data/msa "$D/msa"
cp ~/spd-data/inputs/l512.yaml "$D/inputs/"
cat > "$D/inputs/u100.yaml" <<'YAML'
version: 1
# ubiquitin 76 aa + a 24-mer binder, 100 tokens, no MSA
sequences:
- protein: {id: A, sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG, msa: empty}
- protein: {id: B, sequence: SPEELLKKAEELLKKGDLEGARKL, msa: empty}
YAML
( while :; do echo "$(date +%s) $(for f in /sys/class/tenstorrent/tenstorrent!*/tt_aiclk; do n=${f%/tt_aiclk}; echo -n "${n##*!}:$(cut -d' ' -f1 "$f") "; done)"; sleep 1; done ) > "$OUT/aiclk.log" 2>/dev/null &
SAMPLER=$! CTL=
trap 'kill $SAMPLER 2>/dev/null; [ -n "$CTL" ] && kill -INT $CTL 2>/dev/null' EXIT
fail=0
run() {  # name, then the command
  local n=$1; shift
  say "$n begin"
  CS_TRACE=$OUT/trace-$n.jsonl PYTHONPATH=$TRACE:$PYTHONPATH /usr/bin/time -f %e -o "$OUT/$n.time" \
    timeout -s INT 1800 "$@" > "$OUT/$n.log" 2>&1
  local rc=$?; say "$n rc=$rc wall=$(tail -1 "$OUT/$n.time" 2>/dev/null)"; [ $rc = 0 ] || fail=$rc
}
predict() { python -c 'from tt_bio.main import cli; cli()' predict "$D/inputs/$I.yaml" --model "$M" \
  --accelerator tenstorrent --diffusion_samples 1 --msa_dir "$D/msa" --msa_cache_only --output_format cif "$@"; }
for k in 1 2; do run "cold-$k" predict --out_dir "$OUT/cold-$k"; done
PORT=$((18700 + CHIP)) URL=http://127.0.0.1:$((18700 + CHIP))
say "controller begin port $PORT"
CS_TRACE=$OUT/trace-controller.jsonl PYTHONPATH=$TRACE:$PYTHONPATH \
  python -c 'from tt_bio.main import cli; cli()' controller --port $PORT > "$OUT/controller.log" 2>&1 &
CTL=$!
for _ in $(seq 120); do
  curl -sf "$URL/cluster" 2>/dev/null | grep -q '"online_workers": *[1-9]' && break; sleep 1
done
say "controller up: $(curl -sf "$URL/cluster" 2>/dev/null | head -c 300)"
for k in 1 2 3; do run "ctl-$k" predict --controller "$URL" --out_dir "$OUT/ctl-$k"; done
kill -INT $CTL; wait $CTL; CTL=
say "controller end"
if [ "$M" = boltz2 ]; then
  TT_METAL_CACHE=$OUT/jit-empty run cold-jit predict --out_dir "$OUT/cold-jit"
  rm -rf "$OUT/jit-empty"
fi
say "end rc=$fail"
date -u +%FT%TZ > "$OUT/done"
exit $fail
