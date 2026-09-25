#!/bin/bash
# The non-predict surfaces (embed, saprot, affinity, design) over np_inputs/ on one pinned whglx
# chip: run_np.sh <card> [surface ...]. One process per case, so a refusal or crash in one case
# is that case's result and not the end of the run. Each case writes <case>.log ending in
# EXIT=<code> WALL=<s>.
set -u
cd "$(dirname "$0")/../.."
C=$1; shift
SURFACES=${*:-embed saprot affinity design}
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:mgx-matrix
export TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxm
export TT_METAL_LOGGER_LEVEL=FATAL TT_BIO_LEASE_TIMEOUT=${TT_BIO_LEASE_TIMEOUT:-1800}
PY=$HOME/env/bin/python
I=perf/mgx_matrix/np_inputs
O=perf/mgx_matrix/out_np
UBQ=MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG

case_() {  # case_ <surface/model/case> <cli args...>
  local tag=$1; shift
  mkdir -p "$O/$tag"
  local start; start=$(date +%s)
  $PY -m tt_bio.main "$@" > "$O/$tag.log" 2>&1
  echo "EXIT=$? WALL=$(( $(date +%s) - start ))s" >> "$O/$tag.log"
}

for s in $SURFACES; do case $s in
embed)
  for m in esmc-300m esmc-600m esmc-6b; do
    for f in good.fasta unusual.fasta empty_record.fasta badchar.fasta long2100.fasta mapping.yaml; do
      case_ "embed/$m/${f%.*}" embed "$I/$f" --model $m --out_dir "$O/embed/$m/${f%.*}"
    done
    case_ "embed/$m/bare_string" embed "$UBQ" --model $m --out_dir "$O/embed/$m/bare_string"
    case_ "embed/$m/logits" embed "$I/good.fasta" --model $m --logits --out_dir "$O/embed/$m/logits"
  done ;;
saprot)
  for m in saprot-35m saprot-650m saprot-1.3b; do
    for f in good.fasta unusual.fasta empty_record.fasta badchar.fasta; do
      case_ "saprot/$m/${f%.*}" saprot "$I/$f" --model $m --out_dir "$O/saprot/$m/${f%.*}"
    done
    # --structure: the matching ubiquitin fold from the predict matrix, and a structure of a
    # different sequence (the docs say it is refused).
    ref=$(ls perf/mgx_matrix/out/protenix-v2/*_results_*/structures/base.cif 2>/dev/null | head -1)
    printf ">base\n%s\n" "$UBQ" > "$O/ubq.fasta"
    case_ "saprot/$m/structure_match" saprot "$O/ubq.fasta" --model $m --structure "$ref" --out_dir "$O/saprot/$m/structure_match"
    case_ "saprot/$m/structure_mismatch" saprot "$O/ubq.fasta" --model $m --structure examples/ground_truth_structures/prot.cif --out_dir "$O/saprot/$m/structure_mismatch"
  done ;;
affinity)
  for f in "$I"/aff_*.yaml; do
    b=$(basename "$f" .yaml)
    case_ "affinity/nesso1/$b" affinity "$f" --model nesso1 --out_dir "$O/affinity/nesso1/$b"
  done ;;
design)
  case_ design/boltzgen/bg_protein design "$I/bg_protein.yaml" --model boltzgen --num_designs 2 --budget 1 --out_dir "$O/design/boltzgen/bg_protein"
  case_ design/boltzgen/bg_peptide design "$I/bg_peptide.yaml" --model boltzgen --protocol peptide-anything --num_designs 2 --budget 1 --out_dir "$O/design/boltzgen/bg_peptide"
  case_ design/boltzgen/bg_cyclic_peptide design "$I/bg_cyclic_peptide.yaml" --model boltzgen --protocol peptide-anything --num_designs 2 --budget 1 --out_dir "$O/design/boltzgen/bg_cyclic_peptide"
  case_ design/boltzgen/bg_small_molecule design "$I/bg_small_molecule.yaml" --model boltzgen --protocol protein-small_molecule --num_designs 2 --budget 1 --out_dir "$O/design/boltzgen/bg_small_molecule"
  case_ design/rfd3/binder design "$I/rfd3_binder.json" --model rfd3 --from_pdb --num_designs 1 --out_dir "$O/design/rfd3/binder"
  case_ design/rfd3/motif design "$I/rfd3_motif.json" --model rfd3 --from_pdb --num_designs 1 --out_dir "$O/design/rfd3/motif"
  case_ design/pxdesign/pdl1 design "$I/px_pdl1.yaml" --model pxdesign --num_designs 1 --out_dir "$O/design/pxdesign/pdl1"
  case_ design/pxdesign/nohotspot design "$I/px_nohotspot.yaml" --model pxdesign --num_designs 1 --out_dir "$O/design/pxdesign/nohotspot"
  ;;
esac; done
echo "NP_DONE $(date -u +%FT%TZ)" >> "$O/np.done"
