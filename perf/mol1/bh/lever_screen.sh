#!/bin/bash
# Device ms per diffusion step at one size: the Mol-1 normal set vs that set + each arm, one process per arm.
# Usage: lever_screen.sh CARD SIZE OUTDIR INPUTS_DIR ARM...
# An arm is levers joined by "+", added to the normal set; the pseudo-levers "bf16" (--diffusion_precision bf16) and
# "m<K>" (--diffusion_samples K, a probe of the batched step only) change the run instead.
set -u
C=$1 S=$2 O=$3 IN=$4; shift 4
W=$(cd "$(dirname "$0")/../../.." && pwd); PY=/home/ttuser/tt-bio-dev/env/bin/python3
BASE=$(cd $W && PYTHONPATH=$W $PY -c "from tt_bio import protenix as P; print(\",\".join(sorted(P.NORMAL_LEVERS | P.WIDE_NORMAL | P.MOL1_BH)))" 2>/dev/null | tail -1)
mkdir -p $O; cd $IN
for A in base "$@" base; do
  LV=$BASE; X=()
  for L in ${A//+/ }; do
    case $L in
      base) ;;
      bf16) X+=(--diffusion_precision bf16) ;;
      m[0-9]*) X+=(--diffusion_samples ${L#m}) ;;
      *) LV=$LV,$L ;;
    esac
  done
  [ ${#X[@]} -gt 0 ] && X=(--cli-extra "${X[@]}")
  TT_BIO_LEVERS=$LV TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:mol1-bh-diffusion PYTHONPATH=$W \
    timeout 1200 $PY $W/perf/mol1/bh/phases.py --input mol1-$S.json --out $O/$A --rounds 0 --step-probe 50 "${X[@]}" > $O/$A.log 2>&1
  echo "$(date -u +%FT%TZ) $A rc=$? $(grep -h "^{" $O/$A.log | tail -1 | python3 -c "import json,sys; d=json.loads(sys.stdin.read() or \"{}\"); p=d.get(\"phases_s\",{}); print(p.get(\"probe_step_ms\"), d.get(\"coords_sha\"), (d.get(\"aiclk\") or {}).get(\"median\"))" 2>&1)" >> $O/screen.log
done
touch $O/DONE
