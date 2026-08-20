# Nesso-1 port parity artifacts

One directory per fixture. Everything here is committed so
`scripts/nesso1_port/parity_gate.py` runs with no upstream `nesso` install and no
Tenstorrent card.

```
tyr48/
  tyr48.yaml            48-aa protein + tyrosine SMILES, affinity binder B -> 61 tokens
  processed/            upstream preprocessing output: structure npz, record json,
                        ESM-2 embedding, RDKit conformer pickle
  standard_aa_mols.pkl  the 21 standard-AA RDKit mols the featurizer reads, sliced out
                        of the 413 MB ccd.pkl so that file never has to be committed
  ref_feats.pt          all 25 featurizer tensors, the bit-exact target
  ref_out.json          the affinity scalars the upstream CLI wrote
  meta.json             package versions, feature/activation key lists, sha256s
  ref_acts.pt           33 per-module activations, GITIGNORED (29 MB, regenerable)
```

Refresh with:

```
<ref_venv>/bin/python scripts/nesso1_port/capture_ref.py --fixture .../tyr48
```

## Two things about this fixture that are easy to get wrong

**Featurization samples.** `process_atom_features` calls
`center_random_augmentation`, which applies a random roto-translation to every
conformer ref_pos, drawn from the **global torch RNG** rather than from the
`RandomState(idx)` the featurizer is handed. Featurization is a sampling step.
Capture and gate both call `torch.manual_seed(meta["feat_seed"])` immediately
before fetching the item so reference and port share the draws. Compare without
that and the number means nothing.

**RDKit version changes the input.** ETKDG conformer coordinates are
version-dependent: 2025.09.6 and 2026.03.5 give the same 13 atoms in the same
order but coordinates up to 1.85 A apart for this ligand, which moved
`affinity_pred_value` by 0.0007. That is small next to the 0.058 run-to-run spread
the GPU reference measured, but it is not zero. RDKit mol pickles are also not
forward-compatible: 2025.09.6 cannot read a 2026.03.5 pickle. The capture is
therefore taken with the RDKit the tt-bio runtime ships, and the gate conformer
leg fails loudly on drift instead of silently comparing two different inputs.
