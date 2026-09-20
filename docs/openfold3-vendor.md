# The vendored OpenFold3 data pipeline

tt-bio runs OpenFold3's neural network as an independent ttnn reimplementation, but it
uses OpenFold3's own host-side data pipeline unchanged: the query schema, CCD lookup,
tokenization, and structure, conformer, MSA and template featurization. That code lives
under `tt_bio/_vendor/openfold3/` and comes from
[aqlaboratory/openfold-3](https://github.com/aqlaboratory/openfold-3), Apache-2.0,
Copyright 2026 AlQuraishi Laboratory. The licence is kept at
`tt_bio/_vendor/openfold3/LICENSE`.

This page records what was changed and why, because 15 of those files are not
upstream's file any more.

## Provenance is per file, not per tree

`scripts/of3_port/audit_vendor_provenance.py` compares every vendored file against
several upstream releases and prints which ones it matches. Current state, 108 files:

| | files |
|---|---|
| identical to upstream (after the import rewrite) | 93 |
| carrying tt-bio changes | 15 |

Of the 15, seven are import plumbing (below) and eight carry older tt-bio changes, five
of which expose the `af3_spec_*` flags.

The tree is mostly 0.4.3, but not entirely, so "the tree is 0.4.3" would be wrong:

- `core/utils/relpos.py` is from **0.4.5**, which added the cyclic-offset helpers.
- `core/data/framework/single_datasets/inference.py` is the one file that matches 0.4.3
  and no later release, so it is what pins the rest.

Run the audit after any upstream bump; it is the thing that keeps this page honest.

## The import rewrite

Every vendored file has `openfold3.` rewritten to `tt_bio._vendor.openfold3.`. That is a
plain textual substitution and is reversible, which is how the audit above can compare
against upstream at all.

One place needed more than that. `core/data/framework/__init__.py` imports its
submodules by rebuilding the module name out of filesystem path components
(`path.parts[-6:-1]`), which hardcodes upstream's package depth and tries to import a
bare `openfold3` from inside `tt_bio/_vendor/`. There is no import statement to rewrite,
so the substitution cannot see it. It now derives the prefix from `__package__`, which
imports the same modules wherever the package sits.

## Optional dependencies are imported lazily

The training half of the pipeline needs `lmdb`, `boto3` and `kalign`. Two of its modules,
`primitives/caches/format.py` and `pipelines/preprocessing/template.py`, are also on the
**inference** path, so importing them eagerly would make every `tt-bio predict` user
install training-only packages. Those imports are deferred instead: 38 diff lines across
five files, all import plumbing, none of it reaching a tensor. Upstream uses the same
idiom (`format.py` defers `convert_datacache_to_lmdb`) and so does this tree
(`logging_utils.py` defers `memory_profiler`).

To use the training pipeline, install them:

```
pip install lmdb boto3 pytorch_lightning
```

## The behavioural changes, all flag-gated and all off by default

These are the ones that can move a number. Each defaults to reproducing the OpenFold3
preview2 checkpoint exactly, and each can be switched to the AF3 spec that upstream
v0.5.0 moved to.

| flag | default | what it changes |
|---|---|---|
| `af3_spec_deletion_value` | `False` | MSA `deletion_value` scale. Preview2 emits `atan(d/3) * 8/pi`; the AF3 spec is `2/pi`, exactly 4x smaller. Preview2 trained for all 155k of its steps on the larger value. |
| `af3_spec_profile_columns` | `False` | Which columns the MSA profile averages over. |
| `af3_spec_main_msa_dedup` | `False` | Deduplicates the concatenated main MSA on the sequence array, order preserving. Preview2 trained on the redundant array. |
| `af3_spec_uppercase_msa` | `False` | Whether a3m lowercase insertion columns are uppercased before the sequence array is built. |

They are keyed off the checkpoint rather than simply fixed because feeding preview2 the
corrected value is an input-distribution shift on a shipped, parity-gated model, and
feeding a model trained the other way the preview2 value is the same error mirrored.

## Reproducibility

`scripts/of3_port/batch_digest.py` hashes every feature of every sample. With the
4-structure corpus from `scripts/of3_port/build_of3_subset.py`, the vendored pipeline and
upstream 0.4.3 produce the same digest, and so do two different hosts.

One caveat, and it is a real constraint rather than a footnote: **pin RDKit**. 45 of the
46 features are byte-identical across hosts regardless of version. The 46th, `ref_pos`,
is the reference conformer RDKit embeds with ETKDGv3, and it changes with the RDKit
version even at a fixed seed. Measured: RDKit 2025.09.6 and 2026.03.6 give different
`ref_pos` on the same input, and pinning both hosts to 2025.09.6 makes the whole batch
byte-identical.
