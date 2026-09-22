# Batch and data-order digests

Produced by `scripts/of3_port/batch_digest.py` and `scripts/of3_port/data_order_digest.py`
over the corpus `scripts/of3_port/build_of3_subset.py` fetches. sha256 over dtype, shape
and raw bytes of every feature of every sample, folded in emission order.

## Validation split, 4 pinned structures (7ohe DNA, 7kud RNA, 7vus protein+ligand, 7fb8 multimer)

| run | digest |
|---|---|
| vendored, pc, run 1 | `fc55b78ab8b1d954050a2686b1d53c45b689948ef5b25a4a3128ebb165bcc046` |
| vendored, pc, run 2 | identical |
| upstream openfold3 0.4.3, pc | identical |
| vendored, tt-quietbox2, RDKit pinned | identical |

Controls, which must move and do: `n_templates` 4 to 2 gives
`c4ee72c844d56affb6936de6d4c81650ff66944da5eae57430b6c2e4dc2d7ac2`; seed 42 to 1234
gives `e73b82616ee3c5ddfcb3acd9bd7180f4b96f1a8d8faaf06453d3b8a9779f2e68`, and `ref_pos`
is the only one of the 46 features responsible.

## Training split, 8 sampled structures (1kvu 1wyc 1xbs 210l 2wig 3wnm 4ky2 5ron, seed 42)

`WeightedPDBDataset` over the 292 chain/interface datapoints those produce.

| crop | digest | matches upstream 0.4.3 | matches tt-quietbox2 |
|---|---|---|---|
| none | `15ced5667408bbd154b60b4f2305a7decb8c4e98028bdf9159f1e23c92040e35` | yes | yes |
| 384 (initial_training) | `fbf70aab6bb12bf833e82915dabf18e725020816294200b061633f910e9efdd4` | yes | yes |
| 640 (finetune_1) | `c953f30da82528f2ebc92b5ae1b844b0c2830f6c1d9f192eeb55069671154f37` | | |
| 768 (finetune_2, finetune_3) | `cc3c7cd098158a8c736f51f0a779f6a66357a18abe5b97c34410077d6c2fdf40` | | |

Every run above reported 0 retries. A non-zero count refuses to print a digest at all;
see `install_retry_guard`.

## Data order

`OF3DistributedSampler`, epoch_len 64, seed 42. Per-rank batch is 1, so this is the
batch sequence.

| epoch | digest |
|---|---|
| 0 | `354d097c9d05abab16f2df42372456c25660abf57b73c734803146563bcdd388` |
| 1 | `9fc7fb74b0e75c9eed9dfb5a370cf3e88155fbd99cba34cb5e2f39658449953d` |
| 2 | `d06abd6d9bcb1161ec7810840b0398deee89d51f3f98efffc6e36bb9ffc68451` |

Epoch 0 reproduces exactly on a second construction, and the three epochs differ from
each other, which they must.

## Corpus

| split | files | size | corpus-digest |
|---|---|---|---|
| val | 132 | 4.8 MB | `320373cde5cdebe50cf614dbcf5b2c4881f7d844c1694e738e480257096b52ca` |
| train | 436 | 74.2 MB | `8a2018fd0af07262299f512eebba02fbf6f6921be0ede9e16d272a6205935bdf` |

Per-file hashes in `MANIFEST.val.sha256` and `MANIFEST.train.sha256`.

## Environments

| | pc | tt-quietbox2 |
|---|---|---|
| python | 3.11 | 3.11 |
| numpy | 2.4.6 | 1.26.4 |
| torch | 2.13.0+cpu | 2.14.0+cu130 |
| pytorch_lightning | 2.6.5 | 2.6.6 |
| biotite | 1.6.0 | 1.6.0 |
| rdkit | 2025.09.6 | 2025.09.6 (pinned down from 2026.03.6) |

RDKit is the one that has to match. Before pinning it the two hosts disagreed, on
`ref_pos` and nothing else.
