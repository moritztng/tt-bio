"""Real-activation parity bisect for OpenDDE's structural-token seam.

Captures the residue trunk output on the complete 7ROA input with its real MSA,
then feeds identical activations and checkpoint weights to the upstream CPU
modules and the ttnn port. Reports PCC and norm ratio after the expander and
after each of the four structural refiner blocks.
"""
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")

import torch
import ttnn

from tt_bio.opendde import (OpenDDE, load_opendde_checkpoint,
                            route_opendde_weights)
from tt_bio.opendde_data import build_structural_token_features
from tt_bio.protenix_data import build_complex_features
from tt_bio.tenstorrent import get_device

torch.set_grad_enabled(False)

ROOT = Path(__file__).resolve().parent.parent
SRC = os.environ.get("OPENDDE_SRC", "/tmp/opendde-src")


def pcc_ratio(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    got = got.detach().float().reshape(-1)
    ref = ref.detach().float().reshape(-1)
    n = got.numel()
    sx = sy = sxx = syy = sxy = 0.0
    for start in range(0, n, 1_000_000):
        x = got[start:start + 1_000_000].double()
        y = ref[start:start + 1_000_000].double()
        sx += float(x.sum()); sy += float(y.sum())
        sxx += float(torch.dot(x, x)); syy += float(torch.dot(y, y))
        sxy += float(torch.dot(x, y))
    covariance = sxy - sx * sy / n
    variance_x = sxx - sx * sx / n
    variance_y = syy - sy * sy / n
    pcc = covariance / max((variance_x * variance_y) ** 0.5, 1e-12)
    ratio = (sxx / max(syy, 1e-12)) ** 0.5
    return pcc, ratio


def report(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    pcc, ratio = pcc_ratio(got, ref)
    print(f"{name:38s} PCC={pcc:.6f} norm_ratio={ratio:.6f}", flush=True)


def _full_construct_features() -> dict:
    a3m = (ROOT / "examples" / "msa" / "seq2.a3m").read_text()
    query = next(line.strip() for line in a3m.splitlines()
                 if line.strip() and not line.startswith(">"))
    return build_complex_features([(query.replace("X", "M"), a3m, "protein")])


def _residue_trunk(model: OpenDDE, feats: dict, n_cycles: int = 10):
    p = model._protenix
    tt = p._tt
    fi = p._atom_feat_inputs(feats)
    n_atom, n_token = fi["N"], fi["NT"]
    mt, selection = fi["mt"], fi["S"]
    mean_selection = selection.t() / (selection.t().sum(-1, keepdim=True) + 1e-6)
    deletion_mean = feats["deletion_mean"]
    if deletion_mean.dim() == 1:
        deletion_mean = deletion_mean.reshape(-1, 1)
    s_inputs_tt = p.input_aae(
        tt(feats["ref_pos"]), tt(fi["ref_charge_asinh"]),
        tt(feats["ref_mask"].reshape(n_atom, 1)), tt(fi["f_in"]),
        tt(fi["d"]), tt(fi["v"]), tt(fi["invd"]), mt,
        tt(mean_selection), tt(feats["restype"]), tt(feats["profile"]),
        tt(deletion_mean))
    s_inputs = p._to_host(s_inputs_tt)[:n_token]
    relp = feats["relp"] if "relp" in feats else p._generate_relp(feats)
    s_tt, z_tt = p.trunk(
        feats, s_inputs, relp, feats["token_bonds"], n_cycles=n_cycles)
    s = p._to_host(s_tt, (n_token, s_tt.shape[-1]))
    z = p._to_host(z_tt, (n_token, n_token, p.trunk.C_Z))
    mt_dev = tt(mt.reshape(-1, 1).float())
    c_l = p._to_host(p.diff_feat.c_l(
        tt(feats["ref_pos"]), tt(fi["ref_charge_asinh"]),
        tt(feats["ref_mask"].reshape(n_atom, 1)), tt(fi["f_in"])),
        (n_atom, 128))
    p_lm = p._to_host(p.diff_feat.p_lm(
        tt(fi["d"]), tt(fi["v"]), tt(fi["invd"]), mt_dev),
        (fi["nb"], fi["nq"], fi["nk"], 16))
    return s_inputs, s, z, fi, c_l, p_lm


def _load_reference_modules(state_dict: dict):
    sys.path.insert(0, SRC)
    # optree is only used by an unrelated debug helper in opendde.model.utils.
    sys.modules.setdefault("optree", types.ModuleType("optree"))
    from opendde.model.modules.diffusion import DiffusionConditioning
    from opendde.model.modules.pairformer import PairformerStack
    from opendde.model.modules.transformer import AtomAttentionEncoder
    from opendde.model.modules.structural_tokens import StructuralTokenExpander

    expander = StructuralTokenExpander(
        c_s=384, c_z=384, c_s_inputs=449, n_roles=7,
        init_mode="scratch", role_init_std=0.02,
        pair_feature_init_std=0.02, attention_bias_init=0.1,
        pair_projection_mode="full", pair_chunk_size=128).eval()
    routed = route_opendde_weights(state_dict)
    expander.load_state_dict(routed["expander"], strict=True)

    refiner = PairformerStack(
        n_blocks=4, n_heads=8, c_z=384, c_s=384,
        num_intermediate_factor=2, blocks_per_ckpt=None,
        hidden_scale_up=True).eval()
    prefix = "structural_token_refiner."
    refiner_state = {key[len(prefix):]: value for key, value in state_dict.items()
                     if key.startswith(prefix)}
    refiner.load_state_dict(refiner_state, strict=True)

    conditioning = DiffusionConditioning(
        sigma_data=16.0, c_z=384, c_z_pair_diffusion=128,
        c_s=384, c_s_inputs=449).eval()
    prefix = "diffusion_module.diffusion_conditioning."
    conditioning_state = {key[len(prefix):]: value for key, value in state_dict.items()
                          if key.startswith(prefix)}
    conditioning.load_state_dict(conditioning_state, strict=True)

    atom_encoder = AtomAttentionEncoder(
        n_blocks=3, n_heads=4, c_atom=128, c_atompair=16, c_token=768,
        has_coords=True, c_s=384, c_z=128, blocks_per_ckpt=None).eval()
    prefix = "diffusion_module.atom_attention_encoder."
    atom_state = {key[len(prefix):]: value for key, value in state_dict.items()
                  if key.startswith(prefix)}
    atom_encoder.load_state_dict(atom_state, strict=True)
    return expander, refiner, conditioning, atom_encoder


def _upload_refiner_inputs(p, s, z, bias):
    s_tt = ttnn.reshape(p._tt(s), (1, s.shape[0], s.shape[1]))
    z_tt = ttnn.reshape(p._tt(z), (1, z.shape[0], z.shape[1], z.shape[2]))
    bias_tt = ttnn.reshape(p._tt(bias), (1, 1, bias.shape[0], bias.shape[1]))
    return s_tt, z_tt, bias_tt


def main() -> None:
    feats = _full_construct_features()
    print(f"real input: N_res={feats['restype'].shape[0]} "
          f"N_msa={feats['msa'].shape[0]}", flush=True)
    state_dict = load_opendde_checkpoint()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    model = OpenDDE(state_dict, ckc, dev)
    p = model._protenix
    # Upstream receives the original residue features plus structural annotations.
    ifd = {**feats, **build_structural_token_features(feats)}
    s_inputs, s_residue, z_residue, fi, got_c_l, got_p_lm = _residue_trunk(model, feats)
    print(f"captured residue trunk: s={tuple(s_residue.shape)} "
          f"z={tuple(z_residue.shape)}", flush=True)

    ref_expander, ref_refiner, ref_conditioning, ref_atom_encoder = (
        _load_reference_modules(state_dict))
    ref_si, ref_s, ref_z, ref_pair = ref_expander(
        ifd, s_inputs, s_residue, z_residue)
    ref_bias = ref_pair["structural_pair_attn_bias"]

    tt_si, tt_s, tt_z, tt_bias = model.expander(
        ifd, s_inputs, s_residue, z_residue)
    n_struct = ref_s.shape[0]
    got_si = p._to_host(tt_si, tuple(ref_si.shape))
    got_s = p._to_host(tt_s, tuple(ref_s.shape))
    got_z = p._to_host(tt_z, tuple(ref_z.shape))
    got_bias = p._to_host(tt_bias, tuple(ref_bias.shape))
    report("expander.s_inputs", got_si, ref_si)
    report("expander.s", got_s, ref_s)
    report("expander.z", got_z, ref_z)
    report("expander.attn_bias", got_bias, ref_bias)

    # Reproduce the P4 routing bug for block 0: the structural bias was sent to
    # both triangle attentions as well as the single AttentionPairBias.
    ref_s0, ref_z0 = ref_refiner.blocks[0](
        ref_s.clone(), ref_z.clone(), pair_mask=None,
        extra_attn_bias=ref_bias)
    legacy_s, legacy_z, legacy_bias = _upload_refiner_inputs(
        p, ref_s, ref_z, ref_bias)
    legacy_s, legacy_z = model.refiner.blocks[0](
        legacy_s, legacy_z, attn_mask_start=legacy_bias,
        attn_mask_end=legacy_bias)
    report("refiner.block0.legacy.s", p._to_host(legacy_s, tuple(ref_s0.shape)), ref_s0)
    report("refiner.block0.legacy.z", p._to_host(legacy_z, tuple(ref_z0.shape)), ref_z0)

    # Correct routing: the structural bias belongs only in AttentionPairBias.
    current_ref_s, current_ref_z = ref_s, ref_z
    current_tt_s, current_tt_z, current_bias = _upload_refiner_inputs(
        p, ref_s, ref_z, ref_bias)
    for block_index, (ref_block, tt_block) in enumerate(
            zip(ref_refiner.blocks, model.refiner.blocks)):
        current_ref_s, current_ref_z = ref_block(
            current_ref_s, current_ref_z, pair_mask=None,
            extra_attn_bias=ref_bias)
        current_tt_s, current_tt_z = tt_block(
            current_tt_s, current_tt_z, extra_attn_bias=current_bias)
        got_s = p._to_host(current_tt_s, tuple(current_ref_s.shape))
        got_z = p._to_host(current_tt_z, tuple(current_ref_z.shape))
        report(f"refiner.block{block_index}.correct.s", got_s, current_ref_s)
        report(f"refiner.block{block_index}.correct.z", got_z, current_ref_z)

    parent = ifd["parent_residue_idx"]
    relp_struct = p._generate_relp({
        "asym_id": feats["asym_id"].index_select(0, parent),
        "residue_index": feats["residue_index"].index_select(0, parent),
        "entity_id": feats["entity_id"].index_select(0, parent),
        "sym_id": feats["sym_id"].index_select(0, parent),
        "token_index": ifd["structural_token_index"],
    })
    ref_pair_z = ref_conditioning.prepare_cache(
        relp_struct, current_ref_z, inplace_safe=False)
    got_pair_z = p._diffusion_pair_cond(
        p._tt(current_ref_z), relp_struct).reshape(ref_pair_z.shape)
    report("diffusion_conditioning.pair_z", got_pair_z, ref_pair_z)

    d_lm = fi["d"].reshape(fi["nb"], fi["nq"], fi["nk"], 3)
    v_lm = fi["v"].reshape(fi["nb"], fi["nq"], fi["nk"], 1)
    ref_p_lm, ref_c_l = ref_atom_encoder.prepare_cache(
        feats["ref_pos"], feats["ref_charge"], feats["ref_mask"],
        feats["ref_element"], feats["ref_atom_name_chars"],
        ifd["atom_to_structural_token_idx"], d_lm, v_lm,
        {"mask_trunked": fi["mt"].bool()}, r_l=None, z=None)
    report("atom_encoder.cache.c_l", got_c_l, ref_c_l)
    report("atom_encoder.cache.p_lm", got_p_lm, ref_p_lm)
    ref_p_with_z = ref_atom_encoder._add_token_pair_context_to_atom_pair(
        ref_p_lm.clone(), ref_pair_z,
        ifd["atom_to_structural_token_idx"])
    ref_z_term = ref_p_with_z - ref_p_lm
    got_z_term = p._plm_z_term(
        ref_pair_z, ifd["atom_to_structural_token_idx"],
        fi["nb"], fi["nq"], fi["nk"])
    report("atom_encoder.structural_z_term", got_z_term, ref_z_term)

    base_bias = p.diffusion._dit_pair_biases(ref_pair_z)[0]
    expected_bias = base_bias + ref_bias.float().unsqueeze(0)
    report("diffusion_transformer.block0.legacy_bias", base_bias, expected_bias)
    report("diffusion_transformer.block0.correct_bias", expected_bias, expected_bias)

    print(f"structural tokens: {n_struct}", flush=True)


if __name__ == "__main__":
    main()
