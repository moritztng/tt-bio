#!/usr/bin/env bash
# b2x-integrate-wh-neutral: are the two now-default-on Boltz-2 diffusion levers
# (BOLTZ2_TOKEN_DIT_SDPA, TT_BIO_ATOM_AXIS_BUCKET) correctness-neutral on Wormhole?
# Both were only ever measured on a Blackhole p300c; they ship on everywhere.
#
# Same harness, same fixtures, same protocol and the same 4649 accuracy bar the Blackhole pass
# used, so the two platforms' numbers are directly comparable.
#
#   run_wh.sh ab  [REPS]                  chip 0: the in-process A/B, the 298 aa control, and the
#                                         TT_BIO_TOKEN_BUCKET=0 leg the atom bucket should close
#   run_wh.sh cli [CHIP] [FIXTURE]        the same two arms through the real CLI on any chip
set -eu
WT=${WT:-/home/mthuening/work/wt/b2x-wh-neutral}
PY=${PY:-/home/mthuening/work/tt-bio/env/bin/python3}
OUT="$WT/perf/b2x-wh-neutral"
HOLDER=worker:b2x-integrate-wh-neutral
cd "$WT"
mkdir -p "$OUT"

case "${1:-ab}" in
ab)
  env -u BOLTZ2_TOKEN_DIT_SDPA -u TT_BIO_ATOM_AXIS_BUCKET \
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=$HOLDER \
    PYTHONPATH="$WT" "$PY" perf/b2x-flag-levers/ab_flag_levers.py \
    --out "$OUT/ab512_wh.json" --cifdir "$OUT/cif" --reps "${2:-2}" --keep-512
  mkdir -p "$OUT/cif512" "$OUT/cif298"
  mv "$OUT"/cif/512_* "$OUT/cif512/" 2>/dev/null || true
  mv "$OUT"/cif/298_* "$OUT/cif298/" 2>/dev/null || true
  "$PY" perf/b2x-flag-levers/score298.py "$OUT/cif298" --out "$OUT/control298_wh.json"
  "$PY" perf/b2x-flag-levers/domain_split.py "$OUT/cif512" --split 298 --ref base_0 \
    --out "$OUT/split512_wh.json"
  ;;
cli)
  # Each arm through the real CLI and its worker spawn rather than an in-process module global,
  # required to reproduce the SAME arm's in-process coordinates. Run it on the A/B's own chip and
  # on a sibling: coordinates that agree across two chips of one Galaxy are the stronger claim.
  CHIP=${2:-0}; FIXTURE=${3:-cdk2x2_512}; SIZE=${FIXTURE##*_}
  for arm in default off; do
    if [ "$arm" = default ]; then REFARM=AB; else REFARM=base; fi
    REF="$OUT/cif$SIZE/${SIZE}_${REFARM}_0/${FIXTURE}.cif"
    env -u BOLTZ2_TOKEN_DIT_SDPA -u TT_BIO_ATOM_AXIS_BUCKET \
      TT_VISIBLE_DEVICES="$CHIP" TT_BIO_LEASE_CARDS=0,"$CHIP" TT_BIO_LEASE_HOLDER=$HOLDER \
      PYTHONPATH="$WT" "$PY" perf/b2x-integrate/verify_default.py \
      --arm "$arm" --fixture "$FIXTURE" --ref "$REF" \
      --out "$OUT/cli_${arm}_${SIZE}_chip${CHIP}.json" \
      --keep "$OUT/cli_${arm}_${SIZE}_chip${CHIP}" \
      || echo "CLI_ARM_NONZERO $arm $FIXTURE chip$CHIP"   # a mismatch must not skip the other arm
  done
  ;;
esac
echo "WH_NEUTRAL_LEG_DONE ${1:-ab} $(date -u +%FT%TZ)"
