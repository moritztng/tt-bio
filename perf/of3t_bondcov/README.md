# §6's last uncovered loss term: fired

`bond` is the one OpenFold3 loss term the campaign never saw contribute anything. Two rows
narrowed why. `of3t-auxheads` read the predicate off the loss itself — `bond_mask =
token_bonds * (is_polymer[..., None, :] * is_ligand[..., None])`, so it is a **polymer–ligand**
term — and measured the consequence on the 8-structure corpus: `bond_loss` 0.0,
`‖g(bond=4) − g(bond=0)‖² = 0.0` exactly, 0 of 4,170 tensors moved. `of3t-orchestrator/bondcov`
then found targets satisfying the predicate at mmCIF level, and left one step open: an
annotation is not a feature tensor.

This row closes that step.

## The corpus

**4G5J** (afatinib covalently bound to EGFR Cys797, one covale link) and **4BYH** (ASN–NAG, two).
Both are already in OpenFold3's own `training_cache_with_templates.json` — 180,975 structures — so
this selects from upstream's corpus rather than adding to it. `build_of3_subset.py --ids
4g5j,4byh` builds it; the structures are fetched from the public `s3://openfold3-data` bucket and
none is committed here.

    corpus-digest sha256 e20b564af303d16ecf155cbc283ae773f7a5df2c3e02d655d3c80d8bdf505975

## Files

| file | what it does |
|---|---|
| `bond_mask_probe.py` | featurises every datapoint and reports `token_bonds`, `is_polymer`/`is_ligand` on the partners, and `bond_mask`, computed with upstream's own expression |
| `make_evidence.py` | turns a `bond_coverage.py` report into the record `stage_loss_coverage.py --demonstrated` reads |
| `bond_mask_*.json` | the mask measurement |
| `bond_gradient_*.json` | one forward, two losses differing only in the bond weight, two backwards |

The gradient instrument is `perf/of3t_auxheads/bond_coverage.py`, unchanged in what it computes:
reading the non-zero the same way that row read the zero is the point. It gained `--package`,
`--cache-file` and a `bond_mask` line, because `token_bonds` alone answers a different question.

## Reproducing

    python scripts/of3_port/build_of3_subset.py --target-dir <D> --split train --ids 4g5j,4byh
    python perf/of3t_bondcov/bond_mask_probe.py --data-dir <D> \
        --cache-file <D>/training_cache_with_templates_subset_2.json \
        --stage finetune_1 --crop 384 --out bond_mask.json
    python perf/of3t_auxheads/bond_coverage.py --package openfold3 --data-dir <D> \
        --cache-file <D>/training_cache_with_templates_subset_2.json \
        --checkpoint <of3-p2-155k.pt> --stage finetune_1 --crop 384 --index 12 \
        --rank-template <batch_step003.pt> --out bond_gradient.json

`--index 12` is 4G5J chain 1. The index-to-target map is the dataset's own `datapoint_cache`
and both scripts print it.

No bond is synthesised anywhere. A zero would have been the finding.
