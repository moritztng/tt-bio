#!/usr/bin/env bash
# Run the four RFD4 parity harnesses once a day and say so when one stops being clean.
#
#   scripts/rfd4_port/parity_nightly.sh            # once a day, cheap no-op otherwise
#   scripts/rfd4_port/parity_nightly.sh --force    # run now whatever the stamp says
#
# Why this exists. The four harnesses below are the only thing that grades seven RFD4
# features -- sequence-probability conditioning, index/coordinate groups, the conditioning
# palette, annotated-CIF prep, the recipe sampler and the CIF writer. On 2026-10-25
# `rfd4-complete-audit` found that nothing invoked any of them: no cron, no CI job, no gate.
# Run by hand that day all four were clean, so the features are fine and the SIGNAL was
# missing: a regression in any of them would have landed silently.
#
# pc only, and deliberately so: every harness needs the upstream reference environment
# (rfd4-ref-env) and the upstream clone, which exist on pc and nowhere else. On a host
# without them this exits 0 after saying why, so it is safe as a fleet-wide tick.
set -u

REPO=${REPO:-$HOME/tt-bio}
REF=${REF:-$HOME/rfd4-ref-env/venv/bin/python}
UPSTREAM=${UPSTREAM:-$HOME/rfd4-upstream/RFD4-Proteina}
# run_upstream_spec_tests.py runs UPSTREAM's own pytest files, so it needs a pytest on the path
# (the reference env has none) and a PDB mirror holding 6lyz, 1mso and 1ehz for its fixtures.
MIRROR=${MIRROR:-$HOME/rfd4-io-env/pdb-mirror}
# A pytest-only path, never a full site-packages: putting another venv's packages on
# PYTHONPATH shadows the reference env's numpy and breaks its C extension.
PYTEST_PATH=${PYTEST_PATH:-$HOME/rfd4-ref-env/pytest-venv/lib/python3.12/site-packages}
STATE=${STATE:-$HOME/.coworker/state/rfd4}
STAMP=$STATE/.parity_nightly_stamp
OUT=$STATE/parity-nightly
FORCE=${1:-}

ts() { date -u +%FT%TZ; }

for p in "$REF" "$REPO" "$UPSTREAM"; do
    [ -e "$p" ] || { echo "$(ts) not this host: $p is missing"; exit 0; }
done
if [ "$FORCE" != "--force" ] && [ -f "$STAMP" ] \
   && [ "$(date -u +%F)" = "$(date -u -r "$STAMP" +%F)" ]; then
    exit 0
fi
mkdir -p "$OUT" || exit 1
touch "$STAMP"

cd "$REPO" || exit 1
SHA=$(git rev-parse --short HEAD)
echo "$(ts) tt-bio $SHA, ref env $REF"

# name|log|command. Each harness exits non-zero on a real failure; writer_parity is run on
# the four benchmark inputs the harness can grade (its fifth, the small-molecule one, fails
# inside UPSTREAM's own write_cif and is a harness limit, not a defect of ours).
run() {  # run <name> <command...>
    local name=$1; shift
    local log=$OUT/$name.log
    if timeout 3600 "$@" > "$log" 2>&1; then
        echo "PASS $name"
    else
        echo "FAIL $name (rc=$?, $log)"
        FAILED="$FAILED $name"
    fi
}

FAILED=""
run prep_parity "$REF" -I scripts/rfd4_port/prep_parity.py
if [ -d "$MIRROR" ] && [ -d "$PYTEST_PATH" ]; then
    PDB_MIRROR_PATH=$MIRROR PYTHONPATH=$PYTEST_PATH \
        run upstream_spec_tests "$REF" scripts/rfd4_port/run_upstream_spec_tests.py
else
    echo "SKIP upstream_spec_tests (no PDB mirror at $MIRROR, or no pytest at $PYTEST_PATH)"
fi
run sampler_upstream_parity "$REF" perf/rfd4/sampler/upstream_parity.py
for cif in monomer/monomer-long ppi/ppi_all_atom/01_PD1 \
           motif_scaffolding/motif_unindexed/1BCF_AA \
           dna_binder/dna_binder_denovo_val/7m5w_sequence_context; do
    f=$UPSTREAM/data/benchmarks/$cif.cif.zst
    [ -f "$f" ] || { echo "SKIP writer_parity $cif (input missing)"; continue; }
    run "writer_$(basename "$cif")" "$REF" scripts/rfd4_port/writer_parity.py "$f"
done

{
    echo "# RFD4 parity harnesses, last run $(ts)"
    echo
    echo "tt-bio \`$SHA\`, reference env \`$REF\`. Run by"
    echo "\`scripts/rfd4_port/parity_nightly.sh\`, once a day."
    echo
    if [ -n "$FAILED" ]; then echo "**FAILING:$FAILED**"; else echo "All clean."; fi
    echo
    echo "Logs: \`$OUT/\`."
} > "$STATE/PARITY.md"

if [ -n "$FAILED" ]; then
    bash "$HOME/.coworker/tg.sh" status \
      "RFD4 parity harnesses went red on tt-bio $SHA:$FAILED. Logs in $OUT. These grade seven RFD4 features and nothing else does." 2>/dev/null
    exit 1
fi
echo "$(ts) all clean"
