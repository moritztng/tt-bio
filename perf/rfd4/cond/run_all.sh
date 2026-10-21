#!/usr/bin/env bash
# Every protein condition family, one design run each, on one card.
#
#   TT_VISIBLE_DEVICES=<card> PY=<venv>/bin/python bash perf/rfd4/cond/run_all.sh [outdir]
#
# Launch it detached (nohup/setsid) with its own log: at 329 tokens a family is minutes, and
# eight of them plus eight weight loads is an hour. Each family writes <outdir>/<family>/run.json
# and one line in <outdir>/runs.jsonl, and the whole thing is resumable -- a family whose run.json
# already says rc 0 is skipped, so a relaunch finishes what a kill interrupted.
#
# partial_diffusion runs last and restarts from the binder run's first design, which is why it is
# not in the main loop.
set -u
cd "$(dirname "$0")/../../.."
PY=${PY:-python3}
OUT=${1:-$HOME/rfd4-cond-runs}
mkdir -p "$OUT"

if [ -z "${TT_VISIBLE_DEVICES:-}" ]; then
  echo "TT_VISIBLE_DEVICES is not set: this must run on the card this row leased" >&2
  exit 2
fi

"$PY" perf/rfd4/cond/families.py stage || exit 1

done_already() {
  [ -f "$OUT/$1/run.json" ] && grep -q '"rc": 0' "$OUT/$1/run.json"
}

run() {
  local family=$1; shift
  if done_already "$family"; then echo "== $family: already done, skipping"; return 0; fi
  echo "== $family: $(date -u +%FT%TZ)"
  "$PY" perf/rfd4/cond/run_designs.py --family "$family" --out "$OUT" "$@" \
    > "$OUT/$family.log" 2>&1
  echo "== $family: rc=$? $(date -u +%FT%TZ)"
}

for family in unconditional binder antibody_cdr nanobody_cdr enzyme motif_scaffold structure_prediction_folding inverse_folding; do
  run "$family"
done

parent=$(ls "$OUT"/binder/*_generated.cif 2>/dev/null | head -1)
if [ -n "$parent" ]; then
  run partial_diffusion --parent "$parent"
else
  echo "== partial_diffusion: skipped, the binder run produced no design to restart from"
fi

echo "== all families: $(date -u +%FT%TZ)"
