#!/bin/bash
# Capture Mol-1 upstream goldens + e2e reference predictions on CPU fp32 (mol1-ref).
#   bash run_goldens.sh [--cycles N] [--ship HOST:DIR] [group ...]
#   groups: msa nomsa template_json extras templates formats (default: all)
# Inputs: cases/<group>.json. Outputs: $GOLDEN_ROOT/<group>/{goldens,pred}/... and one log per group.
# --cycles overrides the upstream N_cycle (20); --ship rsyncs each finished group to HOST:DIR/<group>.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=${ROOT:-$HOME/mol1-upstream/mol1-ref}   # what setup_env.sh built
V=$ROOT/venv/bin/python
export MOL_ROOT_DIR=$ROOT/data
export PATH=$ROOT/kalign/bin:$PATH   # template hits are aligned by the kalign on PATH
GOLDEN_ROOT=${GOLDEN_ROOT:-$HOME/mol1-goldens}
SEED=${SEED:-101}
CYCLES=() SHIP=
while [[ ${1:-} == --* ]]; do
  case $1 in --cycles) CYCLES=(--model.N_cycle "$2");; --ship) SHIP=$2;; *) echo "unknown $1"; exit 2;; esac
  shift 2
done
mkdir -p "$GOLDEN_ROOT/inputs"
# the FILE_ ligand of the modcov case: caffeine with an ETKDG conformer, fixed seed
[ -s "$GOLDEN_ROOT/inputs/caffeine.sdf" ] || $V -I -c "
from rdkit import Chem; from rdkit.Chem import AllChem
m=Chem.AddHs(Chem.MolFromSmiles('Cn1cnc2c1c(=O)n(C)c(=O)n2C')); AllChem.EmbedMolecule(m, randomSeed=7)
m=Chem.RemoveHs(m); m.SetProp('_Name','caffeine'); Chem.MolToMolFile(m, '$GOLDEN_ROOT/inputs/caffeine.sdf')"
declare -A FLAGS=([msa]="--use_msa true" [nomsa]="--use_msa false" [template_json]="--use_msa false --use_template true"
              [extras]="--use_msa true --use_rna_msa true" [templates]="--use_msa true --use_template true"
              [formats]="--use_msa true --use_seeds_in_json true")
for g in "${@:-msa nomsa template_json extras templates formats}"; do
  for grp in $g; do
    out="$GOLDEN_ROOT/$grp"; mkdir -p "$out"
    echo "== $grp $(date -u +%FT%TZ)"
    # cases name paths as @MOL1@/examples/... and @GOLDEN@/inputs/...; resolve them for this host
    sed -e "s#@MOL1@#$ROOT/Mol-1#g" -e "s#@GOLDEN@#$GOLDEN_ROOT#g" "$HERE/cases/$grp.json" > "$out/cases.json"
    $V -I "$HERE/capture.py" --golden_root "$out/goldens" --threads ${THREADS:-8} -- \
      --input_json_path "$out/cases.json" --dump_dir "$out/pred" \
      --load_checkpoint_path "$MOL_ROOT_DIR/checkpoint/mol-1-base.pt" \
      --seeds $SEED --use_seeds_in_json false --dtype fp32 --device cpu \
      --triangle_attention torch --triangle_multiplicative torch \
      --need_atom_confidence true ${FLAGS[$grp]} "${CYCLES[@]}" > "$out/run.log" 2>&1 \
      && echo "   ok $grp" || echo "   FAIL $grp (see $out/run.log)"
    [ -z "$SHIP" ] || rsync -a "$out" "$SHIP/" || echo "   SHIP FAIL $grp"
  done
done
echo GOLDENS_DONE
