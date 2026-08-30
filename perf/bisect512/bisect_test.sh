#!/usr/bin/env bash
# git bisect run test: fold the OpenDDE 512 aa page fixture at the CURRENTLY CHECKED OUT commit.
# Does NOT check out anything itself -- git bisect owns HEAD.
#   exit 0   = good : plDDT >= 0.74, the pre-drop numerics (1ea1e6f3 reads 0.763315)
#   exit 1   = bad  : plDDT <  0.74, the big drop has happened (main tip reads 0.725015)
# Threshold, not equality: the two clusters are 0.7633 and ~0.7250, and an exact-equality
# predicate chases a ~2e-4 tail move instead of the transition that matters.
#   exit 125 = skip : the fold could not be scored at this commit
WT=/home/moritz/.coworker/wt/opendde-512aa-numerics-drift-bisect
PY=/home/moritz/tt-bio/env/bin/python3
OUT=$WT/.bisect-out/steps
THRESH=0.74
mkdir -p "$OUT"
cd "$WT" || exit 125
SHA=$(git rev-parse --short HEAD)
J=$OUT/$SHA.json
echo "=== $(date -u +%FT%TZ) BISECT-STEP $SHA ===" >> "$WT/.bisect-out/bisect.log"
env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
    TT_BIO_LEASE_HOLDER=worker:opendde-512aa-numerics-drift-bisect \
    PYTHONPATH=$WT "$PY" "$WT/scripts/gpu_vs_tt/tt_baseline.py" \
    --model opendde --repeat 1 \
    --target perf/size512/fixtures/cdk2x2_512.yaml \
    --msa-a3m perf/size512/fixtures/cdk2x2_512.a3m \
    --label "512 aa" --out "$J" >> "$OUT/$SHA.log" 2>&1
[ -f "$J" ] || { echo "SKIP $SHA (no json)" >> "$WT/.bisect-out/bisect.log"; exit 125; }
V=$("$PY" -c "
import json,sys
d=json.load(open('$J'))
v={f['plddt'] for f in d.get('warm_folds',[]) if f.get('plddt') is not None}
print('MIXED' if len(v)!=1 else repr(v.pop()))
" 2>/dev/null)
[ -z "$V" ] && { echo "SKIP $SHA (unparsed)" >> "$WT/.bisect-out/bisect.log"; exit 125; }
[ "$V" = "MIXED" ] && { echo "SKIP $SHA (folds disagree)" >> "$WT/.bisect-out/bisect.log"; exit 125; }
if "$PY" -c "import sys; sys.exit(0 if float('$V') < $THRESH else 1)"; then
  echo "BAD  $SHA plddt=$V" >> "$WT/.bisect-out/bisect.log"; exit 1
else
  echo "GOOD $SHA plddt=$V" >> "$WT/.bisect-out/bisect.log"; exit 0
fi
