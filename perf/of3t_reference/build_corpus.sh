#!/usr/bin/env bash
# Build the reference corpus with upstream OpenFold3's OWN preprocessing scripts.
#
# Reference stack: github.com/aqlaboratory/openfold-3 tag v0.5.0 =
# c4771653c5d0a3ebb0b3af71b05efd64bc44ee86 (LEDGER R1-AMENDED). tt-bio's production pin stays
# 0.4.3; the update-rule constants are identical between the two.
#
# Nothing here reimplements their pipeline: the two scripts are theirs, unmodified. What this file
# adds is the argument set that actually works, which cost three passes to find.
set -euo pipefail

ROOT=${ROOT:-$HOME/.coworker/scratch/of3t-reference}
UPSTREAM=${UPSTREAM:-$ROOT/upstream050}
PY=${PY:-$HOME/of3-upstream-venv/bin/python}
CCD=${CCD:-$HOME/common/components.cif}
ENTRIES=${ENTRIES:-"1ubq 6qz8 5nw3 7sgb 4hhb 6lu7"}

export PYTHONPATH="$UPSTREAM"
export PATH="$ROOT/bin:$PATH"

mkdir -p "$ROOT/data/cif" "$ROOT/bin"

if [ ! -x "$ROOT/bin/mmseqs" ]; then
  curl -sfL https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -o /tmp/mmseqs.tar.gz
  tar -xzf /tmp/mmseqs.tar.gz -C /tmp && cp /tmp/mmseqs/bin/mmseqs "$ROOT/bin/"
  rm -rf /tmp/mmseqs /tmp/mmseqs.tar.gz
fi

for p in $ENTRIES; do
  [ -s "$ROOT/data/cif/$p.cif" ] || \
    curl -sfL -o "$ROOT/data/cif/$p.cif" "https://files.rcsb.org/download/${p^^}.cif"
done

"$PY" "$UPSTREAM/scripts/data_preprocessing/preprocess_pdb_of3.py" \
  --cif-dir "$ROOT/data/cif" --ccd-path "$CCD" --out-dir "$ROOT/data/prep" \
  --output-format npz --num-workers 0 --log-level WARNING

# Alignment representatives. Matching is exact string equality on the sequence, so sequences must
# be joined across the FASTA's wrapped lines. 0.5.0 requires moltype-annotated headers,
# ">{msa_id}|{moltype}" with moltype protein or rna (io/sequence/fasta.py), and only polymer chains
# that carry an MSA belong in the file: DNA and ligand chains have no alignment representative by
# design. Each representative gets a single-sequence MSA; the file stem must be a key of
# msa.max_seq_counts (dataset_config_components.py) or parse_msas_direct reads zero files and
# msa.py raises IndexError on an empty list instead of falling back to single sequence.
"$PY" "$(dirname "$0")/make_representatives.py" "$ROOT/data/prep"

# --preprocessed-dir wants structure_files/, not the output root: given the root it consolidates
# zero FASTAs and mmseqs dies with "query createdb died".
# --allow-missing-alignment is NOT passed on purpose. It sets filter_missing_alignment=False, which
# skips representative assignment entirely, leaving every alignment_representative_id None and
# killing the dataset later on Path(None).
"$PY" "$UPSTREAM/scripts/data_preprocessing/create_pdb-weighted_training_dataset_cache.py" \
  --metadata-cache "$ROOT/data/prep/metadata.json" \
  --preprocessed-dir "$ROOT/data/prep/structure_files" \
  --alignment-representatives-fasta "$ROOT/data/prep/alignment_representatives.fasta" \
  --output "$ROOT/data/prep/training_cache.json" \
  --dataset-name of3t-ref --log-level WARNING

"$PY" -c "
import json
d = json.load(open('$ROOT/data/prep/training_cache.json'))['structure_data']
for k in sorted(d):
    print(k, {c: v['alignment_representative_id'] for c, v in d[k]['chains'].items()})
"
