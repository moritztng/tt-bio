#!/bin/bash
# Capture Mol-1 upstream goldens + e2e reference predictions on CPU fp32 (mol1-ref).
#   bash run_goldens.sh [group ...]     groups: msa nomsa template_json (default: all)
# Inputs: cases/<group>.json. Outputs: $GOLDEN_ROOT/<group>/{goldens,pred}/... and one log per group.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=${ROOT:-$HOME/mol1-upstream/mol1-ref}   # what setup_env.sh built
V=$ROOT/venv/bin/python
export MOL_ROOT_DIR=$ROOT/data
GOLDEN_ROOT=${GOLDEN_ROOT:-$HOME/mol1-goldens}
SEED=${SEED:-101}
mkdir -p "$GOLDEN_ROOT/inputs"
# the FILE_ ligand of the modcov case: caffeine with an ETKDG conformer, fixed seed
[ -s "$GOLDEN_ROOT/inputs/caffeine.sdf" ] || $V -I -c "
from rdkit import Chem; from rdkit.Chem import AllChem
m=Chem.AddHs(Chem.MolFromSmiles('Cn1cnc2c1c(=O)n(C)c(=O)n2C')); AllChem.EmbedMolecule(m, randomSeed=7)
m=Chem.RemoveHs(m); m.SetProp('_Name','caffeine'); Chem.MolToMolFile(m, '$GOLDEN_ROOT/inputs/caffeine.sdf')"
declare -A FLAGS=([msa]="--use_msa true" [nomsa]="--use_msa false" [template_json]="--use_msa false --use_template true")
for g in "${@:-msa nomsa template_json}"; do
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
      --need_atom_confidence true ${FLAGS[$grp]} > "$out/run.log" 2>&1 \
      && echo "   ok $grp" || echo "   FAIL $grp (see $out/run.log)"
  done
done
echo GOLDENS_DONE
