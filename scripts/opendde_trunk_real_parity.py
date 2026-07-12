"""One-cycle real-input parity gate for OpenDDE's residue trunk."""
import os
import sys
import types

os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")

import torch
import torch.nn.functional as F
import ttnn

from scripts.opendde_real_seam_parity import (
    SRC, _full_construct_features, _residue_trunk, report)
from tt_bio.opendde import OpenDDE, load_opendde_checkpoint
from tt_bio.tenstorrent import get_device


def main() -> None:
    feats = _full_construct_features()
    state_dict = load_opendde_checkpoint()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    model = OpenDDE(state_dict, ckc, dev)
    p = model._protenix
    got_s_inputs, got_s, got_z, fi, _, _ = _residue_trunk(
        model, feats, n_cycles=1)

    sys.path.insert(0, SRC)
    sys.modules.setdefault("optree", types.ModuleType("optree"))
    from opendde.config.inference import build_inference_config
    from opendde.model.opendde import OpenDDE as ReferenceOpenDDE

    cfg = build_inference_config(fill_required_with_null=True)
    cfg.triangle_multiplicative = "torch"
    cfg.triangle_attention = "torch"
    cfg.enable_efficient_fusion = False
    reference = ReferenceOpenDDE(cfg).eval()
    reference.load_state_dict(state_dict, strict=True)

    ref_feats = dict(feats)
    ref_feats["relp"] = p._generate_relp(feats)
    ref_feats["d_lm"] = fi["d"].reshape(
        fi["nb"], fi["nq"], fi["nk"], 3)
    ref_feats["v_lm"] = fi["v"].reshape(
        fi["nb"], fi["nq"], fi["nk"], 1)
    ref_feats["pad_info"] = {"mask_trunked": fi["mt"].bool()}
    ref_s_inputs, ref_s, ref_z = reference.get_pairformer_output(
        ref_feats, N_cycle=1, inplace_safe=False, chunk_size=None)

    ref_s_init = reference.linear_no_bias_sinit(ref_s_inputs)
    ref_z_init = (
        reference.linear_no_bias_zinit1(ref_s_init)[:, None]
        + reference.linear_no_bias_zinit2(ref_s_init)[None]
        + reference.relative_position_encoding(ref_feats["relp"])
        + reference.linear_no_bias_token_bond(
            ref_feats["token_bonds"].unsqueeze(-1)))
    tt_s_init, tt_z_init = p.trunk.trunk_input(
        p.trunk._up(ref_s_inputs), p.trunk._up(ref_feats["relp"]),
        p.trunk._up(ref_feats["token_bonds"].unsqueeze(-1)))

    ref_z_pre_template = ref_z_init + reference.linear_no_bias_z_cycle(
        reference.layernorm_z_cycle(torch.zeros_like(ref_z_init)))
    n_token = ref_z_init.shape[0]
    tt_z_init_4d = ttnn.reshape(
        tt_z_init, (1, n_token, n_token, p.trunk.C_Z))
    tt_z_pre_template = ttnn.add(
        tt_z_init_4d,
        p.trunk._lin(
            p.trunk._ln(
                ttnn.mul(tt_z_init_4d, 0.0),
                "layernorm_z_cycle.weight", "layernorm_z_cycle.bias"),
            "linear_no_bias_z_cycle.weight"))

    ref_z_pre_msa = ref_z_pre_template + reference.template_embedder(
        ref_feats, ref_z_pre_template,
        triangle_multiplicative=cfg.triangle_multiplicative,
        triangle_attention=cfg.triangle_attention,
        inplace_safe=False, chunk_size=None)
    asym = ref_feats["asym_id"]
    multichain_mask = (asym[:, None] == asym[None, :]).float()
    pair_mask = torch.ones(n_token, n_token)
    nt = ref_feats["template_aatype"].shape[0]
    template_features = []
    for template_index in range(nt):
        distogram = (ref_feats["template_distogram"][template_index]
                     * multichain_mask[..., None] * pair_mask[..., None])
        pseudo_beta = (ref_feats["template_pseudo_beta_mask"][template_index]
                       * multichain_mask * pair_mask).unsqueeze(-1)
        aatype = F.one_hot(
            ref_feats["template_aatype"][template_index].long(), 32).float()
        aatype_i = aatype[None].expand(n_token, n_token, 32)
        aatype_j = aatype[:, None].expand(n_token, n_token, 32)
        unit_vector = (ref_feats["template_unit_vector"][template_index]
                       * multichain_mask[..., None] * pair_mask[..., None])
        backbone = (ref_feats["template_backbone_frame_mask"][template_index]
                    * multichain_mask * pair_mask).unsqueeze(-1)
        template_features.append(torch.cat([
            distogram, pseudo_beta, aatype_i, aatype_j,
            unit_vector, backbone], -1))
    tt_z_pre_msa = ttnn.add(
        tt_z_pre_template,
        p.trunk._template(tt_z_pre_template, template_features, n_token, nt))
    ref_z_msa = reference.msa_module(
        ref_feats, ref_z_pre_msa, ref_s_inputs, pair_mask=None,
        triangle_multiplicative=cfg.triangle_multiplicative,
        triangle_attention=cfg.triangle_attention,
        inplace_safe=False, chunk_size=None)
    msa = F.one_hot(ref_feats["msa"].long(), 32).float()
    msa_inputs = torch.cat([
        msa, ref_feats["has_deletion"].unsqueeze(-1),
        ref_feats["deletion_value"].unsqueeze(-1)], -1).unsqueeze(0)
    tt_m = ttnn.add(
        p.trunk._lin(p.trunk._up(msa_inputs),
                     "msa_module.linear_no_bias_m.weight"),
        p.trunk._lin(p.trunk._up(ref_s_inputs),
                     "msa_module.linear_no_bias_s.weight"))
    tt_z_msa = p.trunk._msa(tt_z_pre_msa, tt_m)

    tt_s_pre_pf = ttnn.add(
        tt_s_init,
        p.trunk._lin(
            p.trunk._ln(ttnn.mul(tt_s_init, 0.0),
                        "layernorm_s.weight", "layernorm_s.bias"),
            "linear_no_bias_s.weight"))
    tt_s_pf, tt_z_pf = p.trunk.PF(
        ttnn.reshape(tt_s_pre_pf, (1, n_token, 384)), tt_z_msa)

    report("residue_input_embedder.s_inputs", got_s_inputs, ref_s_inputs)
    report("residue_trunk.s_init", p._to_host(tt_s_init), ref_s_init)
    report("residue_trunk.z_init", p._to_host(tt_z_init), ref_z_init)
    report("residue_trunk.cycle0.z_pre_template",
           p._to_host(tt_z_pre_template, tuple(ref_z_pre_template.shape)),
           ref_z_pre_template)
    report("residue_trunk.cycle0.z_pre_msa",
           p._to_host(tt_z_pre_msa, tuple(ref_z_pre_msa.shape)), ref_z_pre_msa)
    report("residue_trunk.cycle0.z_post_msa",
           p._to_host(tt_z_msa, tuple(ref_z_msa.shape)), ref_z_msa)
    report("residue_trunk.cycle0.s_post_pairformer",
           p._to_host(tt_s_pf, tuple(ref_s.shape)), ref_s)
    report("residue_trunk.cycle0.z_post_pairformer",
           p._to_host(tt_z_pf, tuple(ref_z.shape)), ref_z)
    report("residue_trunk.cycle0.s", got_s, ref_s)
    report("residue_trunk.cycle0.z", got_z, ref_z)


if __name__ == "__main__":
    main()
