#!/usr/bin/env bash
# of3t-p10smbw ABPROOF: the two-step held-out A/B, graded on the RATE of the blown mode.
#
# `of3t-p10trainout` withdrew its own N=1 attribution: 7ohe takes ~10.8 or ~18.9 and nothing
# between, and ten device-native runs at one seed split 6 blown / 3 fine / 1 mixed, so ONE run of
# an arm measures a coin. The arms below alternate FIX and BASE on one card at one sha so host
# load and card state are shared, and every run records which variant was in the tree when it ran.
#
#   [SMBW_CARD=n] abarms.sh <pairs>
W=/home/ttuser/.coworker/wt/of3t-p10smbw
PY=/home/ttuser/tt-bio-dev/env/bin/python
S=/home/ttuser/of3t_p10trainout
L=/tmp/of3t/of3t-p10smbw/ab
O=$W/perf/of3t_p10smbw/ab
cd "$W" || exit 1
mkdir -p "$L" "$O" "$S/runs"
PAIRS=${1:-10}
CARD=${SMBW_CARD:-1}

restore () { cd "$W"; git checkout HEAD -- tt_bio/taped_ttnn.py; }
trap restore EXIT

run () {  # run <arm> <k>
  local arm=$1 k=$2 tag="smbw_$1$2"
  python3 perf/of3t_p10smbw/setarm.py "$arm" || exit 1
  local variant; variant=$(grep -q "BASE ARM" tt_bio/taped_ttnn.py && echo base || echo fix)
  [ "$variant" = "$arm" ] || { echo "FATAL: wanted $arm, tree is $variant"; exit 1; }
  echo "=== $tag ($variant) start $(date -u +%FT%TZ) clk=$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk) load=$(cut -d' ' -f1 /proc/loadavg)"
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-p10smbw \
    "$PY" perf/of3t_p10trainout/trainarm.py \
      --corpus "$S/corpus/train12" --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
      --exact off --steps 2 --seed 0 --checkpoint-every 9 --displacement-band 0.01,100 \
      --eval-corpus "$S/corpus/val" \
      --out-dir "$S/runs/$tag" --curve "$S/runs/$tag.jsonl" \
      --out "$O/arm_$tag.json" > "$L/$tag.log" 2>&1
  local rc=$?
  echo "$variant" > "$O/arm_$tag.variant"
  echo "=== $tag ($variant) done $(date -u +%FT%TZ) rc=$rc clk=$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk)"
  "$PY" perf/of3t_p10smbw/abread.py "$O/arm_$tag.json" "$variant" 2>&1 | sed 's/^/   /'
}

for k in $(seq 1 "$PAIRS"); do
  run fix "$k"
  run base "$k"
done
restore
echo "=== AB CHAIN DONE $(date -u +%FT%TZ)"
