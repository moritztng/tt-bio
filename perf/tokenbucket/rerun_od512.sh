#!/usr/bin/env bash
# The one measurement this task still owes: OpenDDE, 512 aa, page protocol, token bucket ON vs OFF,
# paired and interleaved in one process. 8 pairs.
#
# CARD comes from the environment and has NO DEFAULT on purpose. A card grant is per launch, not
# per task, and the previous copy of this script hardcoded a card that a later launch did not hold
# (see section 10 of the state doc: four runs died on DeviceInUseError). Run it as:
#
#     CARD=<your granted card> bash perf/tokenbucket/rerun_od512.sh
#
set -u
: "${CARD:?set CARD to this launch grant, e.g. CARD=1}"
WT=/home/ttuser/.coworker/wt/protenix-opendde-token-bucket-flip-measure
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=3600 BENCHLOCK_LOAD_WAIT_S=90
LEASE="TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:protenix-opendde-token-bucket-flip-measure"

# PREFLIGHT. On 2026-08-23 this run spent 42 minutes spinning at 100 % of one core against a card
# whose ARC core had died, and only a 256x256 matmul probe told the difference between "wedged" and
# "slow". 60 s here is cheaper than 28 minutes of folding into a dead chip.
echo "$(date -Is) preflight: card $CARD"
if ! timeout 300 env $LEASE PYTHONPATH=$WT "$PY" -u perf/tokenbucket/preflight_card.py >/dev/null 2>&1; then
  echo "$(date -Is) PREFLIGHT FAILED on card $CARD (timeout, wedge, or wrong result). Not folding."
  echo "  a wedge here shows as 100 % of one core in user time, not 0 %; tt-smi -r $CARD, and if"
  echo "  that says ARC core failed to start the card needs a power cycle, not another reset."
  exit 1
fi
echo "$(date -Is) preflight OK, load $(cut -d" " -f1-3 /proc/loadavg)"

# Wait for the load band the clean runs were measured in. od298 gave a 0.070 s within-arm span at
# loadavg 2.96-6.13; od512 gave 1.43 s when load reached 13.88, which decides nothing at a 0.167 s
# margin. Up to 40 min, then fold anyway and let the per-fold stamps convict the run.
t0=$SECONDS
while [ $((SECONDS-t0)) -lt 2400 ]; do
  l=$(cut -d" " -f1 /proc/loadavg)
  awk -v a="$l" "BEGIN{exit !(a+0<=6.0)}" && { echo "$(date -Is) load $l, folding"; break; }
  echo "$(date -Is) load $l, waiting for <= 6"; sleep 20
done

bash /home/ttuser/.coworker/scripts/benchlock.sh protenix-opendde-token-bucket-flip-measure -- \
  env $LEASE PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm \
  "$PY" -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 8 \
    --ab-env TT_BIO_PROTENIX_TOKEN_BUCKET --ab-values 1,0 \
    --target perf/size512/fixtures/cdk2x2_512.yaml --msa-a3m perf/size512/fixtures/cdk2x2_512.a3m \
    --label "od512 8 pairs, page protocol, qb1 card $CARD, paired token-bucket A/B" \
    --msa-dir $WT/.msa_tokenbucket --keep-cif perf/tokenbucket/cif_od512r \
    --out perf/tokenbucket/od512_paired.json
rc=$?
echo "=== $(date -Is) od512 END rc=$rc ==="

# ACCEPTANCE. Reject the run rather than average noise into a 0.167 s decision.
[ $rc -eq 0 ] && "$PY" perf/tokenbucket/accept_od512.py perf/tokenbucket/od512_paired.json
