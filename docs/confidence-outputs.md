# Confidence matrices: PAE, PDE and contact probabilities

`tt-bio predict --write_pae` writes two files next to each predicted structure:

* `<name>_pae.npz`, the matrices, readable with `numpy.load`;
* `<name>_pae.json`, a sidecar naming every array in the npz with its shape, dtype and units,
  and saying why any array the model cannot produce is missing.

Every array belongs to the best-ranked sample, the one written as `<name>.cif`. The token axis is
the written structure's tokens in file order, padding removed: one row per residue, and one per
atom for a ligand or other atomised component.

## Arrays

| key | shape | dtype | units | meaning |
|---|---|---|---|---|
| `pae` | [N, N] | float32 | Å | expected aligned error of token j when the structure is aligned on token i (row i, column j). Not symmetric |
| `pde` | [N, N] | float32 | Å | expected error in the distance between tokens i and j |
| `contact_probs` | [N, N] | float32 | probability | probability that the representative atoms of i and j are within `contact_cutoff_A`. Symmetric, diagonal 1 |
| `contact_cutoff_A` | scalar | float32 | Å | the distance `contact_probs` is actually at (see below) |

pTM, ipTM and chain-pair ipTM are scalars and go into `results.json` as before.

## Per model

| model | `pae` | `pde` | `contact_probs` |
|---|---|---|---|
| `boltz2` | yes | yes | yes, cutoff 7.81 Å |
| `opendde`, `opendde-abag` | yes | yes | yes, cutoff 8.00 Å |
| `openfold3`, `openbind` | yes | yes | yes, cutoff 7.94 Å |
| `esmfold2`, `esmfold2-fast` | yes | yes | yes, cutoff 7.94 Å |
| `rf3` | yes | yes | yes, cutoff 7.71 Å |
| `protenix-v1`, `protenix-v2` | yes | yes | no: tt-bio does not run their distogram head |
| `af2ig` | yes | no: AF2 has no PDE head | yes, cutoff 7.94 Å |

`af2ig`'s matrix is AF2's raw predicted aligned error. The `interface_pae` scalar it already
reports averages the symmetrised matrix, `(pae + pae.T) / 2`, over binder rows and target columns.

## How contact probabilities are computed

From the model's own distogram head and nothing else: the softmax over its distance bins, summed
over every bin whose upper edge is at most the cutoff. This is the rule upstream OpenDDE and
OpenFold3 use. A distogram's bin edges rarely fall exactly on 8 Å, so the cutoff you get is the
largest edge at or below the one you asked for, and that value is written as `contact_cutoff_A`
in the npz and in the sidecar, with the full list of bin edges. Pass `--contact_cutoff 12` for a
looser definition.

The representative atom is the one each model's distogram was trained on: Cβ, Cα for glycine,
and the atom itself for an atomised token.

## Reading them

```python
import json, numpy as np
z = np.load("out/structures/complex_pae.npz")
meta = json.load(open("out/structures/complex_pae.json"))
pae, contacts = z["pae"], z.get("contact_probs")
print(meta["arrays"], meta["absent"], meta.get("contact_cutoff_A"))
```

The `pae` key and file name are the ones earlier releases wrote, so existing readers keep
working.

## Cost

These matrices are opt-in because they grow with N² per fold. On a 164-token complex the flag
added no measurable time or memory for Boltz-2, OpenFold3 or ESMFold-2 (within a second of the
same fold without it). At 1,536 tokens the contact-probability step takes about 0.9 s and 340 MB
of host memory. The structure is byte-identical with and without the flag: none of these heads
feeds the coordinates. Measurements and upstream comparisons are in `perf/fdx_confidence/`.
