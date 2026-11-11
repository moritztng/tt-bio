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
# Upstream's Fast LayerNorm CUDA extension does not build for sm89 under torch 2.13 / CUDA 13.0
# ("Set LAYERNORM_TYPE=torch to use the PyTorch LayerNorm implementation" -- its own error), which
# killed session mol1c-gold-1 at the first fold. This run exists for the GOLDEN REFERENCE -- the
# coordinates and the constraint satisfaction -- so the stock LayerNorm is the right trade.
# ITS WALL CLOCK IS THEREFORE NOT A FAIR NVIDIA PERF NUMBER and must never be published as one:
# a timed GPU comparison needs an image whose fused kernel builds.
export LAYERNORM_TYPE=torch
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
    # Retry the install: $OPT_CORE is a git dependency and the clone fails transiently. On 2026-11-11
    # (session mol1c-gold-4) it died on "RPC failed; curl 92 HTTP/2 stream not closed cleanly", the subshell
    # carried on to the data download, and the box then ran 14 jobs that were all rc=127 for a missing `mol`.
    for n in 1 2 3 4; do
      VIRTUAL_ENV=/opt/mol1-venv uv pip install -e "/root/mol1[cuda]" "$OPT_CORE" ninja && break
      echo "install attempt $n failed, retrying"; sleep 20
    done
    # The install is the one step whose failure must stop the box: nothing after it can work without `mol`.
    [ -x "$MOL" ] || { echo "no $MOL after install"; exit 1; }
    mkdir -p $MOL_ROOT_DIR/checkpoint
    # The Shanghai OSS bucket drops TLS connections (gold-6, 15:32Z: "unexpected eof while reading" on
    # search_database.tar.gz), so every fetch retries; wget -c resumes a partial file.
    for n in 1 2 3 4 5 6; do bash /root/mol1/scripts/database/download_mol_data.sh && break
      echo "data download attempt $n failed, retrying"; sleep 30; [ $n = 6 ] && exit 1; done
    for f in mol-1-constraint mol-1-base; do
      for n in 1 2 3 4 5 6; do
        wget -q -c https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/$f.pt -O $MOL_ROOT_DIR/checkpoint/$f.pt && break
        echo "$f attempt $n failed, retrying"; sleep 30; [ $n = 6 ] && exit 1; done; done
    printf '%s  %s\n%s  %s\n' $CK_C $MOL_ROOT_DIR/checkpoint/mol-1-constraint.pt $CK_B $MOL_ROOT_DIR/checkpoint/mol-1-base.pt | sha256sum -c
  ) > $B/setup.log 2>&1 || fail "setup (tail $B/setup.log)"
  echo ok > $B/SETUP-DONE
fi
# Also guard a RESUMED box (SETUP-DONE from an earlier attempt whose venv is incomplete).
[ -x "$MOL" ] || fail "no $MOL: the dependency install did not produce the CLI (tail $B/setup.log)"
say "setup done"
{ $PY -m pip freeze; nvidia-smi -q; cd /root/mol1 && git log -1 --format=%H; } > $B/census.txt 2>&1
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv,noheader -l 2 >> $B/smi.csv 2>&1 &
cd /root/mol1
# ONE process per arm, every case in one JSON. Each `mol` process pays ~30 min of single-core startup on a
# shared-CPU 4090 host (gold-5: 13:31 init -> 14:02 first GPU work, then a 155 s fold), so a process per case
# could not fit 15 runs in the cap. Per-case walls come from the log ("Seed N completed in"). Order puts the
# cases that settle open questions first (c7 EXCLUSION vs a tight native contact, then c4, c3, c6), so a cut
# session still answers them.
python3 - <<'PY'
import json
cases = {c["name"]: c for c in json.load(open("/root/kc/inputs/cases.json"))}
order = ["c7_exclusion_strong", "c4_combined", "c3_contact", "c6_token_pairs", "c2_exclusion", "c1_site", "c5_none"]
json.dump([cases[n] for n in order if n in cases] + [c for n, c in cases.items() if n not in order],
          open("/root/kc/inputs/all.json", "w"))
if "c5_none" in cases:
    json.dump([cases["c5_none"]], open("/root/kc/inputs/c5_none.json", "w"))
PY
arm(){ local name=$1; shift; local t0; t0=$(date +%s.%N); "$@" > $B/$name.log 2>&1; local rc=$?
  echo "{\"arm\": \"$name\", \"rc\": $rc, \"wall_s\": $(python3 -c "import time;print(round(time.time()-$t0,1))")}" >> $B/jobs.jsonl; }
arm cpred $MOL cpred -i /root/kc/inputs/all.json -o $B/ref/cpred -s ${SEEDS:-101,102}
# SEEDS (BOXENV) picks the seed list; the base arm runs only when the inputs carry c5_none.
[ -f /root/kc/inputs/c5_none.json ] && arm base $MOL pred -i /root/kc/inputs/c5_none.json -o $B/ref/base -s ${SEEDS:-101,102}
pkill -f "nvidia-smi --query-gpu" 2>/dev/null
grep -q '"rc": [1-9]' $B/jobs.jsonl && fail "a run failed (jobs.jsonl)"
echo ok > $B/MOL1-DONE; say done
