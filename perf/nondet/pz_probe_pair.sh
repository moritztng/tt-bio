#!/bin/bash
# pz_probe_pair.sh — two fresh-process 256 aa folds on pc card 0 with TT_PROTENIX_PZPROBE,
# then the zero-compute DRAM readback probe. Readout table in state/protenix-v2-nondeterminism-rootcause.md.
set -u
WT=/home/moritz/.coworker/wt/protenix-v2-nondeterminism-rootcause
PY=/home/moritz/tt-bio/env/bin/python3
export TT_VISIBLE_DEVICES=0 TT_METAL_LOGGER_LEVEL=FATAL \
    TT_BIO_LEASE_HOLDER=worker:protenix-v2-nondeterminism-rootcause
for REP in d e; do
    O=$WT/perf/nondet/out/pzprobe_${REP}
    mkdir -p "$O"
    cd "$WT" || exit 1
    TT_PROTENIX_PZPROBE="$O" PYTHONPATH=. "$PY" -c 'from tt_bio.main import cli; cli()' predict \
        "$WT/perf/nondet/targets/cdk2_256.yaml" \
        --model protenix-v2 --single_sequence --seed 0 \
        --out_dir "$O" > "$O/run.log" 2>&1
    echo "EXIT=$?" >> "$O/run.log"
    cat "$O/pz_cond_probe.json"
done
cd "$WT/perf/nondet" && "$PY" dram_stability.py 2048 6 2>&1 | tee out/dram_stability.log
