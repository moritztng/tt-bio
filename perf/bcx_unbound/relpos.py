"""What residue numbering does each instrument hand the monomer model?

`_predict_complex` renumbers a multi-chain complex for a monomer model
(af2.py:305-306, `monomer_chain_break_indices`). The gradient path
(`_compiled_sequence_gradients` -> `predict_complex_arrays`, af2.py:337-345) builds
asym_id, entity_id and seq_mask the same way and never calls it. model_1_ptm is a
monomer model (af2.py:420) and the design complex has two chains, so the branch is live.

CPU only, no device: this reads the inputs the two paths construct, not a prediction.
"""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_predictor")):
    sys.path.insert(0, p)
import numpy as np, jax.numpy as jnp
import bc2_state as B
from bindcraft.af2 import (monomer_chain_break_indices, pad_design_chains, padded_prediction_complex,
                           alphafold_model_family, cyclic_sequence_offsets, interface_asym_ids)
from bindcraft.prediction import concatenate_chain_arrays, residue_chain_ids
from bindcraft.protein import real_residue_weights

out = {"why": "predict renumbers for a monomer model; sequence_gradients does not"}
for seed in (0, 100):
    s = B.campaign_settings(overrides=[] if seed == 0 else [f"campaign_seed={seed}"])
    _ds, states, _ = B.design_state(s)
    pc = states["hPDL1"]
    names = tuple(sorted(pc))
    entry = {"chains": {n: len(pc[n]) for n in names},
             "model_family": list(alphafold_model_family("model_1_ptm")[:1])}

    # --- the gradient path: pad_design_chains, then concatenate. No renumbering.
    gpad = pad_design_chains({"hPDL1": pc}, 32, 0)["hPDL1"]
    glen = tuple(len(gpad[n]) for n in names)
    gri = np.asarray(concatenate_chain_arrays(names, gpad, "residue_index")["residue_index"])

    # --- the predict path: no chain padding (target_pad_length 0), then renumber.
    plen = tuple(len(pc[n]) for n in names)
    pri_raw = np.asarray(concatenate_chain_arrays(names, pc, "residue_index")["residue_index"])
    pri = np.asarray(monomer_chain_break_indices(plen, jnp.asarray(pri_raw)))

    def junction(ri, lengths):
        j = int(np.cumsum(lengths[:-1])[0])
        return {"index_at_junction": j, "last_of_chain0": int(ri[j - 1]),
                "first_of_chain1": int(ri[j]), "step": int(ri[j]) - int(ri[j - 1]),
                "abs_offset_across": abs(int(ri[j]) - int(ri[j - 1]))}
    entry["gradient_path"] = {"n": int(gri.shape[0]), "chain_lengths": list(glen),
                              **junction(gri, glen)}
    entry["predict_path"] = {"n": int(pri.shape[0]), "chain_lengths": list(plen),
                             **junction(pri, plen)}
    entry["predict_path_before_renumber"] = junction(pri_raw, plen)
    out[f"seed{seed}"] = entry
    print(seed, json.dumps(entry, indent=1), flush=True)
(HERE / "relpos.json").write_text(json.dumps(out, indent=1))
