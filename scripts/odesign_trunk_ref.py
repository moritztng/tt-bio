"""Pass-6 trunk-input CPU reference: run ODesign's own RelativePositionEncoding
(with the cyclic-peptide offset logic), the trunk-init linears (s_init / z_init),
and the ConstraintTemplateEmbedder FRONT-END (distogram binning + v_ij projection)
on the golden pre's token meta + a synthetic constraint distogram, and dump the
intermediates so the ttnn port can be diffed against a known-correct reference.

Pure torch, CPU, eval, no grad. Same methodology as passes 1-5 (CPU-fp32 ref =
ODesign's own modules run fresh). The 2-block PairformerStack INSIDE the
ConstraintTemplateEmbedder is deferred to pass 7 (triangle-attention head geometry
needs a dedicated port); only the front-end (binning + v_ij) is captured here.
"""
import os, sys, pickle, torch
import torch.nn.functional as F

sys.path.insert(0, "/home/moritz/.coworker/scratch/odesign-ref/ODesign")
os.environ.setdefault("LAYERNORM_TYPE", "")
from src.api._base import DictAccessMixin
from src.model.modules.embedders import RelativePositionEncoding
from src.model.modules.pairformer import ConstraintTemplateEmbedder
from src.model.modules.primitives import LinearNoBias


class P(DictAccessMixin):
    def __init__(self, d): self.__dict__.update(d)

CKPT = "/home/moritz/.coworker/scratch/odesign-ref/ckpt/odesign_base_prot_flex.pt"
PRE = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_denoiser_pre.pkl"
OUT = "/home/moritz/.coworker/scratch/odesign-ref/golden/odesign_trunk_input_ref.pkl"

R_MAX, S_MAX, C_Z, C_S, C_S_INPUTS = 32, 2, 128, 384, 453


def pcc(u, v):
    u = u.flatten().double(); v = v.flatten().double()
    return float(((u - u.mean()) * (v - v.mean())).sum()
                 / ((u - u.mean()).norm() * (v - v.mean()).norm() + 1e-12))


def load_sd():
    ck = torch.load(CKPT, map_location="cpu", weights_only=True); ck = ck.get("model", ck)
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}


def make_pairformer_input(feat):
    """Build the minimal dict-access object the ODesign RPE expects."""
    return P({k: feat[k] for k in
              ["asym_id", "residue_index", "entity_id", "sym_id", "token_index", "is_cyclic_token"]})


def trunk_init(sd, s_inputs, relpe, token_bonds):
    """s_init = LinearNoBias_sinit(s_inputs); z_init = zinit1(s_init)[...,None,:]
    + zinit2(s_init)[...,None,:,:] + relpe + token_bond_linear(token_bonds[...,None])."""
    sinit_w = sd["linear_no_bias_sinit.weight"]            # (c_s, c_s_inputs)
    z1_w = sd["linear_no_bias_zinit1.weight"]               # (c_z, c_s)
    z2_w = sd["linear_no_bias_zinit2.weight"]               # (c_z, c_s)
    tb_w = sd["linear_no_bias_token_bond.weight"]           # (c_z, 1)
    s_init = F.linear(s_inputs, sinit_w)                    # (N, c_s)
    z_init = (F.linear(s_init, z1_w)[..., None, :]
              + F.linear(s_init, z2_w)[..., None, :, :])    # (N, N, c_z)
    z_init = z_init + relpe
    z_init = z_init + F.linear(token_bonds.unsqueeze(-1), tb_w)
    return s_init, z_init


def synth_cyclic_meta(n=12):
    """A single cyclic chain of n residues: asym_id=0, entity_id=0, sym_id=0,
    residue_index=0..n-1, token_index=0..n-1, is_cyclic_token=1. Exercises the
    cyclic-peptide offset branch of ODesign's RPE (genuinely-new code)."""
    return {
        "asym_id": torch.zeros(n, dtype=torch.long),
        "residue_index": torch.arange(n, dtype=torch.long),
        "entity_id": torch.zeros(n, dtype=torch.long),
        "sym_id": torch.zeros(n, dtype=torch.long),
        "token_index": torch.arange(n, dtype=torch.long),
        "is_cyclic_token": torch.ones(n, dtype=torch.long),
    }


def synth_constraint_feature(ref_pos, a2t, n_token):
    """Token-centroid distance map (N_token, N_token, 1) as a plausible constraint
    distogram input (ODesign's constraint_feature is (N,N,1) -- a per-pair distance;
    `cf > boundaries` broadcasts (N,N,1) vs (38,) -> (N,N,38), sum dim=-1 -> (N,N)).
    Exact values don't matter for parity (same input to both sides); made realistic
    from the atom coords."""
    centre = torch.zeros(n_token, 3)
    counts = torch.zeros(n_token, 1)
    for a in range(ref_pos.shape[0]):
        centre[a2t[a]] += ref_pos[a]; counts[a2t[a]] += 1
    centre = centre / counts.clamp(min=1)
    d = torch.cdist(centre, centre)                          # (N,N)
    return d.unsqueeze(-1)                                    # (N,N,1)


def main():
    torch.set_grad_enabled(False)
    sd = load_sd()
    pre = pickle.load(open(PRE, "rb"))
    feat = pre["input_data"]
    s_inputs = pre["s_inputs"].float()
    n_token = s_inputs.shape[0]
    print("N_token =", n_token)

    # --- RelativePositionEncoding (ODesign, with cyclic offset) ---
    rpe = RelativePositionEncoding(r_max=R_MAX, s_max=S_MAX, c_z=C_Z)
    rpe.linear_no_bias.weight.data = sd["relative_position_encoding.linear_no_bias.weight"].clone()
    rpe.eval()
    pfi = make_pairformer_input(feat)
    relpe = rpe(pfi)                                          # (N, N, c_z)
    print("relpe", tuple(relpe.shape), "finite:", bool(torch.isfinite(relpe).all()))

    # cyclic path on a synthetic cyclic peptide
    cyc_pfi = make_pairformer_input(synth_cyclic_meta(12))
    relpe_cyc = rpe(cyc_pfi)
    print("relpe_cyc", tuple(relpe_cyc.shape), "finite:", bool(torch.isfinite(relpe_cyc).all()))

    # --- trunk init linears (s_init, z_init) ---
    token_bonds = torch.zeros(n_token, n_token)              # no covalent bonds in this example
    s_init, z_init = trunk_init(sd, s_inputs, relpe, token_bonds)
    print("s_init", tuple(s_init.shape), "z_init", tuple(z_init.shape))

    # --- ConstraintTemplateEmbedder front-end (binning + v_ij projection) ---
    cte = ConstraintTemplateEmbedder(n_blocks=2, c=64, c_z=C_Z, dropout=0.25)
    cte_sd = {k[len("constraint_distogram_embedder."):]: v
              for k, v in sd.items() if k.startswith("constraint_distogram_embedder.")}
    missing, unexpected = cte.load_state_dict(cte_sd, strict=False)
    print("CTE missing:", len(missing), "unexpected:", len(unexpected))
    cte.eval()
    cf = synth_constraint_feature(feat["ref_pos"].float(),
                                  feat["atom_to_token_idx"].long(), n_token)
    # build a PairFormerInput with just constraint_feature (front-end only needs that + z)
    pfi_c = P({"constraint_feature": cf})
    # replicate the front-end (lines 1548-1560) explicitly so we capture v_ij WITHOUT
    # running the 2-block PairformerStack (deferred to pass 7)
    boundaries = torch.linspace(cte.min_bin, cte.max_bin, cte.no_bins - 1)
    true_bins = torch.sum(cf > boundaries, dim=-1)
    distogram = F.one_hot(true_bins, cte.no_bins).to(z_init.dtype)
    v_ij = cte.linear_no_bias_z(cte.layernorm_z(z_init)) + cte.linear_no_bias_a(distogram)
    print("distogram", tuple(distogram.shape), "v_ij", tuple(v_ij.shape),
          "finite:", bool(torch.isfinite(v_ij).all()))

    out = {
        "relpe": relpe.clone(), "relpe_cyc": relpe_cyc.clone(),
        "s_init": s_init.clone(), "z_init": z_init.clone(),
        "distogram": distogram.clone(), "v_ij": v_ij.clone(),
        "constraint_feature": cf.clone(),
        "token_meta": {k: feat[k].clone() for k in
                       ["asym_id", "residue_index", "entity_id", "sym_id",
                        "token_index", "is_cyclic_token"]},
        "s_inputs": s_inputs.clone(),
        "cyclic_meta": synth_cyclic_meta(12),
    }
    pickle.dump(out, open(OUT, "wb"))
    print("saved", OUT)
    # self-consistency: relpe dim check
    print("relpe last-dim:", relpe.shape[-1], "(expect", C_Z, ")")
    print("v_ij last-dim:", v_ij.shape[-1], "(expect", 64, ")")


if __name__ == "__main__":
    main()
