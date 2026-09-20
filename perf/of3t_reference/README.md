# OpenFold3 training reference

The frozen reference the OF3T campaign compares against: upstream OpenFold3's own training stack,
pinned, run on a corpus built with their own preprocessing scripts.

Upstream is `github.com/aqlaboratory/openfold-3` tag **0.4.3**, commit
`0bb17be5199846e806b6347b6e17c6249c88ff1b` — the version `NOTICE:45` pins.

## Files

| file | what it does |
|---|---|
| `build_corpus.sh` | runs their `preprocess_pdb_of3.py` and `create_pdb-weighted_training_dataset_cache.py` over a handful of RCSB entries, and writes the alignment representatives those scripts need |
| `stage_config.yml` | their `examples/training_yamls/initial_training.yml`, with the four single-box changes listed in the file header |
| `build_batch.py` | freezes N batches through their `WeightedPDBDataset` and collator, one `.pt` per step, sha256 per file, plus `batches_manifest.json` |

## Running it

```sh
bash build_corpus.sh
PYTHONPATH=<openfold3-0.4.3-checkout> python build_batch.py \
    --config stage_config.yml --corpus <corpus>/data/prep --out <corpus>/bundle/batches --steps 20
```

## Things that cost a pass each

* `create_pdb-weighted_training_dataset_cache.py --preprocessed-dir` wants the `structure_files/`
  subdirectory, not the preprocessing output root. Given the root it consolidates zero FASTAs and
  mmseqs exits 1 with `Error: query createdb died`, which reads like a broken mmseqs install.
* `--allow-missing-alignment` does not tolerate chains without an MSA. It skips
  alignment-representative assignment entirely, so every `alignment_representative_id` is `None`
  and the dataset dies later on `Path(None)`. Supply a representative for every polymer chain and
  leave the flag off.
* Representative matching is exact string equality on the sequence, so the FASTA needs
  `<pdb_id>_<chain_id>` headers and sequences joined across wrapped lines.
* An MSA file's stem must be one of the keys in `msa.max_seq_counts`; anything else is not read,
  and an empty MSA set raises `IndexError` rather than falling back to single sequence.

## The first optimizer step does not move the weights

Their scheduler is built with `last_epoch=-1`, so step 1 runs at **lr = 0**. Weights after step 1
equal the initial weights on a correct implementation and on a broken one alike, and step 2 moves
them by ~1.8e-6 relative. Compare step-1 **gradients**, not step-1 weights.

Full write-up, including what is measured and what is still owed:
`~/.coworker/state/of3t-reference.md`.
