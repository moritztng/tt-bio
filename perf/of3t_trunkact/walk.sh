#!/usr/bin/env bash
# of3t-trunkact: the forward per-block walk at padded N=384. qb1 (tt-quietbox), p150a
# Blackhole, card 0. AICLK is sampled every 4 s DURING the device arm and reported min/median/
# max; a number without a clock is not a measurement on this hardware.
#   walk.sh dev  <TAG> [--set k=v ...]      our device walk, per-block dump
#   walk.sh ref  <TAG>                      upstream 0.4.3 f64 + bf16auto, scoring the dump
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-trunkact
cd "$W"
O=/tmp/of3t/of3t-trunkact
F=/home/ttuser/of3t_frame384
mkdir -p "$O"
CARD=${TA_CARD:-0}

MODE=$1; TAG=$2; shift 2

case "$MODE" in
dev)
  CLK=$O/aiclk_${TAG}.txt
  : > "$CLK"
  ( while true; do
      /home/ttuser/.local/bin/tt-smi -s 2>/dev/null \
        | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['device_info'][$CARD]['telemetry']['aiclk'].strip())" \
        >> "$CLK" 2>/dev/null
      sleep 4
    done ) &
  SAMPLER=$!
  trap 'kill "$SAMPLER" 2>/dev/null' EXIT
  S=$(date +%s)
  echo "=== dev $TAG start $(date -u +%FT%TZ) host=$(hostname) card=$CARD ==="
  source /home/ttuser/tt-bio-dev/env/bin/activate
  TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
  TT_BIO_LEASE_HOLDER=worker:of3t-trunkact OMP_NUM_THREADS=8 \
  timeout 1800 python3 perf/of3t_trunkact/devwalk.py \
    --boundary "$F/boundary_n384.pt" --dump "$O/ours_${TAG}" \
    --report "$W/perf/of3t_trunkact/DEVWALK_${TAG}.json" "$@" 2>&1 \
    | grep -E '^\{|Traceback|rror|FAILED|placed' | tail -20
  rc=${PIPESTATUS[0]}
  E=$(date +%s)
  kill "$SAMPLER" 2>/dev/null
  echo "=== dev $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
  echo -n "AICLK during (card $CARD, p150a, MHz): "
  sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "NO SAMPLES"}'
  du -sh "$O/ours_${TAG}" 2>/dev/null
  exit $rc
  ;;
pol)
  S=$(date +%s)
  echo "=== pol $TAG start $(date -u +%FT%TZ) host=$(hostname) CPU only ==="
  PYTHONPATH=$F/ref:$F/deps OMP_NUM_THREADS=16 nice -n 5 \
  timeout 7200 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trunkact/policywalk.py \
    --tree "$F/of3pkg043" --boundary "$F/boundary_n384.pt" \
    --cap-last "$F/block47_boundary.pt" \
    --out "$W/perf/of3t_trunkact/POLICYWALK_${TAG}.json" "$@" 2>&1 \
    | grep -vE "UserWarning|warnings.warn|  from openfold3|Consider using tensor.detach" | tail -60
  rc=${PIPESTATUS[0]}
  E=$(date +%s)
  echo "=== pol $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
  exit $rc
  ;;
ref)
  S=$(date +%s)
  echo "=== ref $TAG start $(date -u +%FT%TZ) host=$(hostname) CPU only ==="
  PYTHONPATH=$F/ref:$F/deps OMP_NUM_THREADS=16 nice -n 5 \
  timeout 7200 /home/ttuser/tt-bio-dev/env/bin/python perf/of3t_trunkact/refwalk.py \
    --tree "$F/of3pkg043" --boundary "$F/boundary_n384.pt" \
    --out "$W/perf/of3t_trunkact/REFWALK_${TAG}.json" "$@" 2>&1 \
    | grep -vE "UserWarning|warnings.warn|  from openfold3|Consider using tensor.detach" | tail -60
  rc=${PIPESTATUS[0]}
  E=$(date +%s)
  echo "=== ref $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
  exit $rc
  ;;
*) echo "unknown mode $MODE"; exit 2 ;;
esac
