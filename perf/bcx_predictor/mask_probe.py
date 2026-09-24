"""Does BindCraft 2's real design state produce a masked AF2 fold, and can it be asked not to?

tt_bio/af2.py:24-28 documents that the ttnn AF2 trunk serves only all-ones masks and
af2.py:385 / :509 assert it. BindCraft 2 pads the DESIGN chain to length_bucket_size, and
those pad residues are masked, so the default campaign hands the trunk a masked fold.
length_bucket_size is a compile-reuse device, not part of the model: at 1 the complex is
unpadded, which is the regime the trunk serves and also the reference's own semantics.
"""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import bc2_state as B
from bindcraft.af2 import pad_design_chains, protein_state_shapes, concatenate_chain_arrays
from bindcraft.protein import real_residue_weights, has_residue_flag, ResidueFlags

def probe(bucket, overrides=()):
    s = B.campaign_settings(overrides=overrides)
    ds, states, losses = B.design_state(s)
    padded = pad_design_chains(states, bucket, 0)
    out = {}
    for state_name, chain_names, chain_lengths in protein_state_shapes(padded):
        a = concatenate_chain_arrays(chain_names, padded[state_name], "flags", "sequence")
        m = np.asarray(real_residue_weights(a["flags"]))
        d = np.asarray(has_residue_flag(a["flags"], ResidueFlags.DESIGN))
        out[state_name] = {"chains": list(chain_names), "chain_lengths": list(chain_lengths),
                           "n_total": int(m.shape[0]),
                           "seq_mask_all_ones": bool((m == 1).all()),
                           "n_masked_residues": int((m == 0).sum()),
                           "n_total_mod_32": int(m.shape[0]) % 32,
                           "n_design_residues": int(d.sum())}
    return out

blob = {"bucket_32_default": probe(32),
        "bucket_1_unpadded": probe(1, overrides=["length_bucket_size=1"])}
(HERE / "mask_probe.json").write_text(json.dumps(blob, indent=1))
print(json.dumps(blob, indent=1))
