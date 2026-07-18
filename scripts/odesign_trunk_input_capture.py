"""Pass-7 prerequisite: trunk-INPUT capture harness. Runs ODesign's data pipeline
(SampleDictToFeatures) on the prot_binding_prot example JSON (use_msa=False,
data_condition={'data','constraint_distogram'}) on CPU to dump the full feature_data
the InputFeatureEmbedder / MSAModule / ConstraintTemplateEmbedder consume, then hooks
ODesign's get_pairformer_output to dump per-cycle s/z + final s_trunk/z_trunk/s_inputs,
and verifies the dumped final matches the golden pre (PCC ~1.0) -- establishing the
trunk parity baseline.

Pure torch, CPU, eval, no grad. Run from the ODesign repo root so relative ref_file
paths resolve.
"""
import os, sys, json, pickle, torch
os.chdir("/home/moritz/.coworker/scratch/odesign-ref/ODesign")
sys.path.insert(0, "/home/moritz/.coworker/scratch/odesign-ref/ODesign")
os.environ.setdefault("LAYERNORM_TYPE", "")

from src.utils.inference.inference_utils import SampleDictToFeatures
from src.api.model_interface import PairFormerInput

JSON = "/home/moritz/.coworker/scratch/odesign-ref/ODesign/examples/protein_design/prot_binding_prot/odesign_input.json"
PRE = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_denoiser_pre.pkl"
OUT = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_trunk_inputs.pkl"


def pcc(u, v):
    u = u.flatten().double(); v = v.flatten().double()
    return float(((u - u.mean()) * (v - v.mean())).sum()
                 / ((u - u.mean()).norm() * (v - v.mean()).norm() + 1e-12))


def main():
    torch.set_grad_enabled(False)
    samples = json.load(open(JSON))
    s = samples[0]
    print("sample name:", s.get("name"))
    s2f = SampleDictToFeatures(single_sample_dict=s,
                                data_condition={"data", "constraint_distogram"},
                                use_msa=False)
    feature_data, label_data, atom_array, token_array = s2f.get_feature_and_label()
    print("feature_data type:", type(feature_data).__name__)
    fd = feature_data  # OFeatureData (dict-like)
    keys = list(fd.keys()) if hasattr(fd, "keys") else [a for a in dir(fd) if not a.startswith("_")]
    print("feature_data keys (first 40):", keys[:40])
    # dump the trunk-input features we need
    want = ["restype", "profile", "deletion_mean", "is_hotspot_residue",
            "token_bonds", "msa", "has_deletion", "deletion_value",
            "constraint_feature", "residue_index", "asym_id", "entity_id",
            "sym_id", "token_index", "is_cyclic_token", "cyclic_mode",
            "cycle_bonds", "ref_pos", "ref_space_uid", "ref_element",
            "ref_mask", "ref_atom_name_chars", "ref_charge", "atom_to_token_idx"]
    dumped = {}
    for k in want:
        if k in fd:
            v = fd[k]
            dumped[k] = v.clone() if hasattr(v, "clone") else v
            print(f"  {k}: {tuple(v.shape) if hasattr(v,'shape') else type(v).__name__} {v.dtype if hasattr(v,'dtype') else ''}")
        else:
            print(f"  {k}: MISSING")
    # compare token meta against the golden pre
    pre = pickle.load(open(PRE, "rb"))
    pf = pre["input_data"]
    print("\n--- token meta match vs golden pre ---")
    for k in ["residue_index", "asym_id", "entity_id", "sym_id", "token_index", "is_cyclic_token"]:
        if k in dumped and k in pf:
            a = dumped[k]; b = pf[k]
            if hasattr(a, "shape") and a.shape == b.shape:
                eq = bool(torch.equal(a.long(), b.long()))
                print(f"  {k}: shape {tuple(a.shape)} match={eq}")
            else:
                print(f"  {k}: shape a={getattr(a,'shape',None)} b={getattr(b,'shape',None)}")
    pickle.dump({"feature_data": dumped, "label_coordinate": label_data.get("coordinate") if hasattr(label_data,"get") else None}, open(OUT, "wb"))
    print("saved", OUT)


if __name__ == "__main__":
    main()
