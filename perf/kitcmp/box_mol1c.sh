#!/bin/bash
# Runs ON a rented NVIDIA box for mol1-constraint goldens (state/mol1/evidence/constraint): upstream Mol-1 constraint
# branch, `mol cpred` on the six 7PZB cases (inputs/cases.json), seeds 101,102, plus base `mol pred` on the
# unconstrained case. Driven by mol1_session.sh (BOX=box_mol1c.sh INPUTS=<dir with cases.json> QB2=...).
#   markers /root/kc/mol1/{SETUP-DONE,MOL1-DONE,MOL1-FAILED}; structures under /root/kc/mol1/ref (offloaded to qb2)
set -u
SHA=9e85ad16; B=/root/kc/mol1; PY=/opt/mol1-venv/bin/python; MOL=/opt/mol1-venv/bin/mol
CK_C=33c5139f8bce74ae92bc33e46336724cc11bb720578e34f8a237834cc5e2fa5b
CK_B=d5b83a86884aa6870565b1a249e8f13c232dfc6dfa311089ab0c7868212b6bd6
OPT_CORE='opt_core @ git+https://github.com/anthropics/uplifting-biomolecular-modeling.git@f4f62fa6592ae4938d49b1757bea0cfeff9f468e#subdirectory=common/opt_core'
export MOL_ROOT_DIR=/root/moldata
mkdir -p $B/ref; cd /root/kc
say(){ echo "$(date -u +%FT%TZ) box_mol1c: $*"; }
fail(){ say "FAILED: $1"; echo "$1" > $B/MOL1-FAILED; exit 3; }
if [ ! -f $B/SETUP-DONE ]; then
  ( set -e
    apt-get update -qq && apt-get install -y -qq --no-install-recommends git wget python3.12-dev build-essential \
      libxrender1 libxext6 libsm6 kalign
    for n in 1 2 3 4; do rm -rf /root/mol1; git clone https://github.com/molence/Mol-1 /root/mol1 && break; sleep 20; done
    cd /root/mol1 && git fetch origin constraint && git checkout $SHA
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=$PATH:/root/.local/bin
    uv venv --python 3.12 --seed /opt/mol1-venv
    VIRTUAL_ENV=/opt/mol1-venv uv pip install -e "/root/mol1[cuda]" "$OPT_CORE" ninja
    mkdir -p $MOL_ROOT_DIR/checkpoint
    bash /root/mol1/scripts/database/download_mol_data.sh
    for f in mol-1-constraint mol-1-base; do
      wget -q https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/$f.pt -O $MOL_ROOT_DIR/checkpoint/$f.pt; done
    printf '%s  %s\n%s  %s\n' $CK_C $MOL_ROOT_DIR/checkpoint/mol-1-constraint.pt $CK_B $MOL_ROOT_DIR/checkpoint/mol-1-base.pt | sha256sum -c
  ) > $B/setup.log 2>&1 || fail "setup (tail $B/setup.log)"
  echo ok > $B/SETUP-DONE
fi
say "setup done"
{ $PY -m pip freeze; nvidia-smi -q; cd /root/mol1 && git log -1 --format=%H; } > $B/census.txt 2>&1
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv,noheader -l 2 >> $B/smi.csv 2>&1 &
cd /root/mol1
# One case per run so each gets its own wall time; the first run also pays the fused-LayerNorm JIT build.
python3 - <<'PY'
import json
cases = json.load(open("/root/kc/inputs/cases.json"))
for c in cases:
    json.dump([c], open(f"/root/kc/inputs/{c['name']}.json", "w"))
PY
for c in $(python3 -c 'import json;print(" ".join(c["name"] for c in json.load(open("/root/kc/inputs/cases.json"))))'); do
  for s in 101 102; do
    t0=$(date +%s.%N)
    $MOL cpred -i /root/kc/inputs/$c.json -o $B/ref/cpred -s $s > $B/cpred_${c}_$s.log 2>&1; rc=$?
    echo "{\"arm\": \"cpred\", \"case\": \"$c\", \"seed\": $s, \"rc\": $rc, \"wall_s\": $(python3 -c "import time;print(round(time.time()-$t0,1))")}" >> $B/jobs.jsonl
  done
done
for s in 101 102; do
  t0=$(date +%s.%N)
  $MOL pred -i /root/kc/inputs/c5_none.json -o $B/ref/base -s $s > $B/base_c5_$s.log 2>&1; rc=$?
  echo "{\"arm\": \"base\", \"case\": \"c5_none\", \"seed\": $s, \"rc\": $rc, \"wall_s\": $(python3 -c "import time;print(round(time.time()-$t0,1))")}" >> $B/jobs.jsonl
done
pkill -f "nvidia-smi --query-gpu" 2>/dev/null
grep -q '"rc": [1-9]' $B/jobs.jsonl && fail "a run failed (jobs.jsonl)"
echo ok > $B/MOL1-DONE; say done
