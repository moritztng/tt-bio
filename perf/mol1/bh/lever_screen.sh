#!/bin/bash
# Device ms per diffusion step at one size, the Mol-1 normal set vs that set + each candidate lever, one process per arm.
# Usage: lever_screen.sh CARD SIZE OUTDIR INPUTS_DIR LEVER...   (an arm "bf16" means --diffusion_precision bf16)
set -u
C=$1 S=$2 O=$3 IN=$4; shift 4
W=$(cd "$(dirname "$0")/../../.." && pwd); PY=/home/ttuser/tt-bio-dev/env/bin/python3
BASE=$(cd $W && PYTHONPATH=$W $PY -c "from tt_bio import protenix as P; print(\",\".join(sorted(P.NORMAL_LEVERS | P.WIDE_NORMAL | P.MOL1_BH)))" 2>/dev/null | tail -1)
mkdir -p $O; cd $IN
for L in base "$@" base; do
  if [ $L = base ]; then LV=$BASE; X=(); elif [ $L = bf16 ]; then LV=$BASE; X=(--cli-extra --diffusion_precision bf16); else LV=$BASE,$L; X=(); fi
  TT_BIO_LEVERS=$LV TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:mol1-bh-diffusion PYTHONPATH=$W \
    timeout 1200 $PY $W/perf/mol1/bh/phases.py --input mol1-$S.json --out $O/$L --rounds 0 --step-probe 50 "${X[@]}" > $O/$L.log 2>&1
  echo "$(date -u +%FT%TZ) $L rc=$? $(grep -h "^{" $O/$L.log | tail -1 | python3 -c "import json,sys; d=json.loads(sys.stdin.read() or \"{}\"); p=d.get(\"phases_s\",{}); print(p.get(\"probe_step_ms\"), d.get(\"coords_sha\"), (d.get(\"aiclk\") or {}).get(\"median\"))" 2>&1)" >> $O/screen.log
done
touch $O/DONE
