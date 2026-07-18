"""Pass-7 baseline: run ODesign's own InputFeatureEmbedder on the captured
trunk inputs (scripts/odesign_trunk_input_capture.py) on CPU, and verify the
produced s_inputs matches the golden pre's s_inputs (PCC ~1.0). This confirms
the data pipeline + InputFeatureEmbedder reproduce the golden -- the trunk
parity baseline. Pure torch, CPU, eval, no grad.
"""
import os, sys, pickle, torch
sys.path.insert(0, "/home/moritz/.coworker/scratch/odesign-ref/ODesign")
os.environ.setdefault("LAYERNORM_TYPE", "")
from src.api._base import DictAccessMixin
from src.model.modules.embedders import InputFeatureEmbedder

CKPT = "/home/moritz/.coworker/scratch/odesign-ref/ckpt/odesign_base_prot_flex.pt"
INP = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_trunk_inputs.pkl"
PRE = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_denoiser_pre.pkl"
OUT = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_s_inputs_ref.pkl"


def pcc(u, v):
    u = u.flatten().double(); v = v.flatten().double()
    return float(((u - u.mean()) * (v - v.mean())).sum()
                 / ((u - u.mean()).norm() * (v - v.mean()).norm() + 1e-12))


class P(DictAccessMixin):
    def __init__(self, d): self.__dict__.update(d)


def main():
    torch.set_grad_enabled(False)
    ck = torch.load(CKPT, map_location="cpu", weights_only=True); ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    ie_sd = {k[len("input_embedder."):]: v for k, v in sd.items() if k.startswith("input_embedder.")}
    ie = InputFeatureEmbedder(c_atom=128, c_atompair=16, c_token=384)
    missing, unexpected = ie.load_state_dict(ie_sd, strict=False)
    print("InputFeatureEmbedder missing:", len(missing), "unexpected:", len(unexpected))
    if missing: print("  missing[:5]:", missing[:5])
    if unexpected: print("  unexpected[:5]:", unexpected[:5])
    ie.eval()

    inp = pickle.load(open(INP, "rb"))
    fd = inp["feature_data"]
    pre = pickle.load(open(PRE, "rb"))
    # build a PairFormerInput with everything the encoder + embedder need
    keys = ["restype", "profile", "deletion_mean", "is_hotspot_residue",
            "ref_pos", "ref_mask", "ref_element", "ref_atom_name_chars",
            "ref_charge", "ref_space_uid", "atom_to_token_idx"]
    pfi = P({k: fd[k] for k in keys if k in fd})
    s_inputs = ie(pfi, inplace_safe=False, chunk_size=None)
    print("produced s_inputs:", tuple(s_inputs.shape), s_inputs.dtype)
    g = pre["s_inputs"].float()
    print("golden  s_inputs:", tuple(g.shape), g.dtype)
    p = pcc(s_inputs, g); m = float((s_inputs - g).abs().max())
    print(f"s_inputs PCC vs golden: {p:.6f}  maxerr {m:.4e}")
    pickle.dump({"s_inputs_ref": s_inputs.clone()}, open(OUT, "wb"))
    print("saved", OUT)
    print("BASELINE", "PASS" if p >= 0.999 else "CHECK", "(bar PCC >= 0.999)")


if __name__ == "__main__":
    main()
