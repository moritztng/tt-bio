#!/usr/bin/env bash
# The three gradient arms, on qb1. CPU only, no card.
#
# qb1 has 503 GB, so bondcov's OOM at 22 GB RSS on a 30 GB box is not a constraint and the
# crop is the stage's own 384 rather than the 256 that row had to drop to.
set -uo pipefail
W=${W:-/home/ttuser/.coworker/wt/of3t-covpaths}
S=${S:-/home/ttuser/of3t_cov}
cd "$W"
export PYTHONPATH="$S/bondcov_assets/of3pkg043:$W"   # openfold3 0.4.3, the campaign's pinned release
export OMP_NUM_THREADS=${OMP:-16}
PY=${PY:-/home/ttuser/of3t_up/venv/bin/python}
D="$S/datasets"
CACHE="$D/training_cache_with_templates_subset_4.json"
CROP=${CROP:-384}

run () {  # run <path> <index> <tag>
  echo "=== $1 idx $2 crop $CROP start $(date -u +%FT%TZ) ==="
  nice -n 19 "$PY" perf/of3t_covpaths/covpath_gradient.py \
      --path "$1" --package openfold3 \
      --data-dir "$D" --cache-file "$CACHE" \
      --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
      --stage initial_training --crop "$CROP" --index "$2" --dtype float32 \
      --rank-template "$S/bondcov_assets/batch_step003.pt" \
      --out "perf/of3t_covpaths/gradient_$3.json"
  echo "=== exit $? $(date -u +%FT%TZ) ==="
}

# Indices are positions in the dataset's own datapoint_cache, printed by covpath_probe.py
# and recorded in presence_n4.json: 24 = 1kvu, 27 = 4hj5, 30 = 5kla, 33 = 5oid.
case "${1:-all}" in
  templates)  run templates  24 templates_1kvu ;;
  nucleotide) run nucleotide 27 nucleotide_4hj5_dna ;;
  rna)        run nucleotide 30 nucleotide_5kla_rna ;;
  disabled)   run disabled   33 disabled_5oid ;;
  control)    # the control's control: the nucleotide edit on a protein-only target must be inert
              run nucleotide 24 nucleotide_1kvu_CONTROL ;;
  all)        run templates  24 templates_1kvu
              run nucleotide 27 nucleotide_4hj5_dna
              run disabled   33 disabled_5oid ;;
  *) echo "usage: $0 {templates|nucleotide|rna|disabled|control|all}"; exit 2 ;;
esac
