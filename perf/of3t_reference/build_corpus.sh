#!/usr/bin/env bash
# Build the reference corpus with upstream openfold3 0.4.3's OWN preprocessing scripts.
#
# Upstream: github.com/aqlaboratory/openfold-3 tag 0.4.3 = 0bb17be5199846e806b6347b6e17c6249c88ff1b
# Nothing here is a reimplementation: the two scripts are theirs, unmodified. What this file adds
# is the argument set that actually works, which cost three passes to find.
set -euo pipefail

ROOT=${ROOT:-$HOME/.coworker/scratch/of3t-reference}
UPSTREAM=${UPSTREAM:-$ROOT/upstream}
PY=${PY:-$HOME/of3-upstream-venv/bin/python}
CCD=${CCD:-$HOME/common/components.cif}
ENTRIES=${ENTRIES:-"1ubq 6qz8 5nw3 7sgb"}

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

# Alignment representatives. Matching is exact string equality on the sequence
# (caches/filtering.py:489), so headers must be <pdb_id>_<chain_id> and sequences must be joined
# across the FASTA's wrapped lines. Each representative gets a single-sequence MSA; the file stem
# must be a key of msa.max_seq_counts (dataset_config_components.py:80) or parse_msas_direct reads
# zero files and msa.py:547 raises IndexError on an empty list.
"$PY" - "$ROOT" <<'PY'
import shutil, sys
from pathlib import Path
root = Path(sys.argv[1]) / "data/prep"
aln = root / "alignments"
shutil.rmtree(aln, ignore_errors=True)
aln.mkdir()
out = []
for fa in sorted(root.glob("structure_files/*/*.fasta")):
    pdb = fa.parent.name
    recs, hdr, seq = [], None, []
    for line in fa.read_text().splitlines():
        if line.startswith(">"):
            if hdr:
                recs.append((hdr, "".join(seq)))
            hdr, seq = f"{pdb}_{line[1:].strip()}", []
        elif line.strip():
            seq.append(line.strip())
    if hdr:
        recs.append((hdr, "".join(seq)))
    for h, s in recs:
        out.append(f">{h}\n{s}")
        d = aln / h
        d.mkdir(exist_ok=True)
        (d / "uniref90_hits.a3m").write_text(f">{h}\n{s}\n")
(root / "alignment_representatives.fasta").write_text("\n".join(out) + "\n")
print(f"{len(out)} alignment representatives")
PY

# --preprocessed-dir wants structure_files/, not the output root: given the root it consolidates
# zero FASTAs and mmseqs dies with "query createdb died".
# --allow-missing-alignment is NOT passed on purpose. It sets filter_missing_alignment=False, which
# skips representative assignment entirely, leaving every alignment_representative_id None and
# killing the dataset later in msa.py:537 on Path(None).
"$PY" "$UPSTREAM/scripts/data_preprocessing/create_pdb-weighted_training_dataset_cache.py" \
  --metadata-cache "$ROOT/data/prep/metadata.json" \
  --preprocessed-dir "$ROOT/data/prep/structure_files" \
  --alignment-representatives-fasta "$ROOT/data/prep/alignment_representatives.fasta" \
  --output "$ROOT/data/prep/training_cache.json" \
  --dataset-name of3t-ref --log-level WARNING

"$PY" -c "
import json, sys
d = json.load(open('$ROOT/data/prep/training_cache.json'))['structure_data']
for k in sorted(d):
    reps = {c: v['alignment_representative_id'] for c, v in d[k]['chains'].items()}
    print(k, reps)
"
