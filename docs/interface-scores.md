# Interface scores

`tt_bio.interface_scores` computes, for every chain pair of a folded complex, the scores binder
campaigns rank by: ipSAE, ipTM, interface pAE, pDockQ, pDockQ2 and LIS. They are arithmetic on
the PAE matrix, the pLDDT and the coordinates, so they need no device. `tt-bio predict
--write_pae` adds them to each Boltz-2 results row, and `tt-bio score` computes them for a fold
you already have.

## Which ipSAE

There are several published readings of ipSAE, so a number has to say which it is. This one is
the script Adaptyv used to rank the Nipah competition: `ipsae.py` from
[adaptyvbio/nipah_ipsae_pipeline](https://github.com/adaptyvbio/nipah_ipsae_pipeline) at commit
`80e0d56`, which is Dunbrack's version 3 of 2025-04-06. We match it exactly, including its quirks,
because a score that is "more correct" than the one participants are compared by is a different
score. `tests/test_interface_scores.py` runs that script unmodified on the same files and holds
every column it prints to half a unit of its last printed digit.

The choices, where the reference leaves room for one:

| | choice | why |
|---|---|---|
| PAE and distance cutoffs | 15 A and 15 A | The script has no default. Dunbrack's examples pass 10/10; Adaptyv's notebook passes 15/15, and that is the competition's number. |
| Which direction | both, plus `ipsae` (max) and `ipsae_min` (min) | The script prints the max. Adaptyv's competition page cites `ipSAE_min` as the best single predictor of binding. Both are returned and neither is chosen for you. |
| Chain-pair ipTM | the model's own, by real chain order | The script maps chain letters to the model's matrix by alphabet position, which swaps the two directions when a binder is declared as B before target A. The max over both directions is unaffected. |
| `interface_pae` | mean PAE over both interchain blocks, no cutoff | Not in the script. This is BindCraft's `i_pae` before its /31 scaling. |

Distances are CB-CB (CA for glycine), a PAE pair counts when strictly below the cutoff, and a
contact counts at 8 A or less, all as in the script. The definitions carry a version,
`ipsae-v3-nipah80e0d56.1`, in every output. Anything that would change a number for the same
inputs bumps it.

## The distribution, not only the number

In the Nipah competition, Boltz-2 ipSAE varied with hardware and seed, and not by the same amount
for every design. A fixed seed makes a number reproducible. It does not say whether a design's
score is stable.

So with `--diffusion_samples K`, every sample is scored against its own structure and the row
carries `interface_score_distribution`: each sample's value in rank order, with mean, standard
deviation and range per pair and metric. The standard deviation is the design's stability figure.
The top-ranked sample stays in `interface_scores` as the point value.

At inference Boltz-2's only random step is the diffusion noise, so K samples of one fold and K
separately seeded folds draw from the same distribution, and the samples share one trunk pass.
That follows from the code. It has not been measured on a device yet.

## Other ipSAE numbers in tt-bio

- **BoltzGen's design filter** reports `design_ipsae_min`. It uses the same definition with a
  smaller floor on d0 (19 residues instead of 27). On the same PAE the two agree to 6e-8 when the
  best residue has 27 or more partners under the cutoff, and differ by at most 0.0075 below that.
  The larger difference is the input: BoltzGen scores its own refold with all target chains
  pooled. It is a design-time score. The number to compare and cite is this module's.
- **`scripts/abag_pae_metrics.py`** computes a different quantity under the name `ipsae` for one
  finished research campaign. It is not served.

## Against a GPU

Tenstorrent arithmetic differs from CUDA's, so a Boltz-2 fold here does not print the same
digits as the same fold on a GPU. The scoring arithmetic itself matches the reference script as
described above; how closely the folds agree is a separate, measured question.
