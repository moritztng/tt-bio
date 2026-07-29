# AbAg-XM release provenance (closeout spec 1.5)

Every number in this release derives from 492 accepted fold records (164 targets x
3 generators) in `progress.jsonl`, verified by `scripts/abag_xm_acceptance.py`
(492/492 accepted, 2026-07-29). This file pins every input those folds consumed:
model weights, engine code, the generation command line, and the labeling tools.

## 1. Model checkpoints

sha256 over the resolved weight files on tt-quietbox (qb1); every hash verified
identical on tt-quietbox2 (qb2) and against the current `main` LFS etag of the
HF repo (checked 2026-07-30), so the pins resolve to publicly fetchable bytes.

| generator | HF repo | file | sha256 | repo revision used |
|---|---|---|---|---|
| protenix-v2 | TMF001/protenix-v2-weights (Apache-2.0) | `protenix-v2.pt` | `8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599` | content identical at 653edab2 (current main) |
| boltz2 | moritztng/boltz-2 (MIT) | `boltz2_conf.ckpt` | `090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1` | content identical at e98526ea (current main) |
| opendde-abag | aurekaresearch/OpenDDE | `opendde_abag.pt` | `5cf37441ddef2a2f148b81dd4a218ad274f996fecaf17dec901ab6cf1351713d` | snapshot `eddd563ce96571f784012edd8f045181c8f8627d` (current main) |

`boltz2_aff.ckpt` (sha256 `dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e`)
is present in the cache but is the affinity head — no fold in this release loads it.

## 2. tt-bio engine code

492 accepted records pin `tt_bio_commit`; 60 distinct commits cover them (the
campaign folded continuously while tt-bio development proceeded). Enumeration +
diff evidence (all computable in the tt-bio repo):

**The three model-engine files are bit-identical across all 60 commits** —
`tt_bio/protenix.py` blob `18f9bc7ae139`, `tt_bio/opendde.py` blob
`2f315784b18d`, `tt_bio/boltz2.py` blob `d8afa8be8d05`. The model numerics that
produced the slab never changed during the campaign. Verified by
`git rev-parse <commit>:<file>` for every commit (equivalent to empty
`git diff a b -- <files>` for all 1,770 pairs).

The 60 commits fall into 5 `tt_bio` tree groups; the differences are confined to
`tt_bio/tenstorrent.py` (runner infrastructure) and `tt_bio/main.py`
(serve/CLI orchestration) and are all campaign-hardening changes — the device
watchdog docstring, the mmseqs/pixi-binary resolution fix, and the dead-worker
run detector. None touches a model file:

| `tt_bio` tree | commits | folds | engine files |
|---|---|---|---|
| `3dc9db33353b` | 52 | 422 | identical (see above) |
| `c0ab5ed19602` | 3 | 3 | identical |
| `1936ccd045dd` | 2 | 55 | identical |
| `99c4d506f103` | 2 | 6 | identical |
| `8ed580cc0fe0` | 1 | 5 | identical |

(Commit-to-fold mapping: `git log` over the accepted records; the five folds
recovered from `.fold_provenance.json` sidecars additionally pin
`tt_bio_tree = 3dc9db33353bfe4535948c2a8cea4b894118bc14` and a per-fold MSA sha
in `progress.jsonl`.)

One fold (9yc5 / protenix-v2) ran at commit `24e0cc0d`, which was never pushed
to origin. Its parent `ae294a51` is in origin history, its `tt_bio` tree is
`3dc9db33353b` (identical to the majority group), and its five engine-file
blobs were verified byte-identical to the table above directly from the
tt-quietbox object store. Its engine state is fully pinned by tree hash.

## 3. Generation command line (all folds, all generators)

Driven by `scripts/abag_xm_generate.py` (on this branch). Per fold:

```
python -m tt_bio.main predict <target>.yaml --model <protenix-v2|boltz2|opendde-abag> \
    --out_dir ~/abag_xm/tier_a/<model> --diffusion_samples 50 --max_parallel_samples <mps> \
    --msa_dir ~/abag_xm/msa_cache --msa_db_path ~/.boltz/msa_db \
    --seed 42 --override --write_pae [--host_threads <cores/concurrent_folds>]
```

with `TT_VISIBLE_DEVICES` pinning one physical card per fold. Fixed inputs:
50 diffusion samples, seed 42 for every fold of every target (cross-generator
fairness: sample k of generator A and sample k of generator B start from the
same seed), shared MSA cache keyed by sequence hash, shared MSA DB. The YAMLs
(same antibody heavy/light + antigen sequences for all three generators) are
inputs under `~/abag_xm/tier_a/yaml/` on the campaign hosts.

`--max_parallel_samples` (card batching, not model config): boltz2 164/164 at
mps=5 (gate-enforced; batching was found numerics-relevant for boltz2 and
standardized at 5); opendde-abag 137 at mps=5, 27 at mps=3; protenix-v2 136 at
mps=5, 28 at mps=3. mps changes device-side scheduling only; the acceptance
gate rejects any boltz2 fold not at mps=5.

## 4. Labeling + environment tools

| tool | version | used for |
|---|---|---|
| DockQ | 2.1.3 | DockQ per model-native pair (`dockq` in labels.parquet) |
| ANARCI | 2026.2.13.2 | IMGT numbering for CDR-H3 boundaries (`cdr_h3_rmsd`) |
| tmtools | 0.3.0 | TM-align wrapper (`tm_rmsd`, `tm_score`) |
| MMseqs2 | 18.8cc5c | MSA search (via localcolabfold `colabfold_search`; version-matched sibling resolution per `tt_bio/main.py::_find_mmseqs`) |
| hosts | Ubuntu 22.04.5 LTS, kernel 6.8.0-136-generic | both campaign hosts |
| tt-kmd | 2.8.0 | Tenstorrent kernel driver, both hosts |

Label generation: `scripts/abag_xm_labels_campaign.py` (label venv
`~/.abag_xm_label_venv` on qb1) over the native CIFs in
`~/abag_xm/ground_truth/` (collected from the internal data lake, see the
dataset card's ground-truth section).

## 5. What is NOT pinned (and why)

- The MSA database (`~/.boltz/msa_db`, multi-TB): the per-fold consumed MSA
  bytes are pinned by `msa_sha` where recorded (all folds after 2026-07-27 and
  the 5 sidecar-recovered folds); the DB itself is the public
  ColabFold/mmseqs set, reproducible by re-fetch.
- `seed` is not in the progress records; it is the driver constant `--seed 42`
  recorded here and in `.fold_provenance.json` sidecars from 2026-07-28 on.

The dataset card (`docs/abag-xm-dataset-card.md`) links this file as the
release's provenance of record.
