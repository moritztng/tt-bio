#!/bin/bash
# Cold-call split of `tt-bio predict` on one chip: perf/spd_coldstart/run.sh CHIP OUT [MODELS] [INPUTS]
# Per model x input: split.py twice (each a fresh process, so the second reads the checkpoint from page cache and
# compiles nothing new), then the real CLI twice under /usr/bin/time, then (boltz2 u100 only) split.py against an
# empty TT_METAL_CACHE for the first-ever-call price. u100 is ubiquitin 76 aa + a 24-mer, single sequence (GitHub
# #22's shape); l512 is the SPD 512 rung with its cached MSA. AICLK of every node is sampled each second.
set -u
CHIP=$1 OUT=$2 MODELS=${3:-boltz2 protenix-v2} INPUTS=${4:-u100 l512}
ROW=spd-bet-coldstart
mkdir -p "$OUT" ~/spd/locks ~/spd/$ROW/leases
[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=$ROW TT_BIO_LEASE_DIR=$HOME/spd/$ROW/leases
say() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/run.log"; }
exec 9> "${LOCK:-$HOME/spd/locks/chip$CHIP.lock}"
flock -w "${WAIT:-600}" 9 || { say "flock busy"; exit 3; }
say "start chip $CHIP tree $(cat TREE_SHA 2>/dev/null || git rev-parse --short HEAD) host $(hostname)"
D=$OUT/data; mkdir -p "$D/inputs"; ln -sfn ~/spd-data/msa "$D/msa"
cp ~/spd-data/inputs/l512.yaml "$D/inputs/"
cat > "$D/inputs/u100.yaml" <<'EOF'
version: 1
# ubiquitin 76 aa + a 24-mer binder, 100 tokens, no MSA
sequences:
- protein: {id: A, sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG, msa: empty}
- protein: {id: B, sequence: SPEELLKKAEELLKKGDLEGARKL, msa: empty}
EOF
( while :; do echo "$(date +%s) $(for f in /sys/class/tenstorrent/tenstorrent!*/tt_aiclk; do n=${f%/tt_aiclk}; echo -n "${n##*!}:$(cut -d' ' -f1 "$f") "; done)"; sleep 1; done ) > "$OUT/aiclk.log" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
fail=0
run() {  # name, then the command
  local n=$1; shift
  say "$n begin"
  /usr/bin/time -f %e -o "$OUT/$n.time" timeout -s INT 1800 "$@" > "$OUT/$n.log" 2>&1
  local rc=$?; say "$n rc=$rc wall=$(cat "$OUT/$n.time" 2>/dev/null | tail -1)"; [ $rc = 0 ] || fail=$rc
}
for m in $MODELS; do
  for i in $INPUTS; do
    for k in 1 2; do
      run "split-$m-$i-$k" python perf/spd_coldstart/split.py --out "$OUT/split-$m-$i-$k" --chip "$CHIP" --model "$m" \
        --inputs "$i" --data "$D" --warm 1 --samples 1
    done
    for k in 1 2; do
      run "cli-$m-$i-$k" python -m tt_bio.main predict "$D/inputs/$i.yaml" --model "$m" --accelerator tenstorrent \
        --diffusion_samples 1 --msa_dir "$D/msa" --msa_cache_only --output_format cif --out_dir "$OUT/cli-$m-$i-$k"
    done
  done
done
case " $MODELS $INPUTS " in *" boltz2 "*" u100 "*)
  TT_METAL_CACHE=$OUT/jit-empty run split-boltz2-u100-coldjit python perf/spd_coldstart/split.py \
    --out "$OUT/split-boltz2-u100-coldjit" --chip "$CHIP" --model boltz2 --inputs u100 --data "$D" --warm 1 --samples 1
  rm -rf "$OUT/jit-empty";;
esac
say "end rc=$fail"
date -u +%FT%TZ > "$OUT/done"
exit $fail
