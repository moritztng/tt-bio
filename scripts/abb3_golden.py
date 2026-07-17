"""Capture ABodyBuilder3 reference structure-module intermediates (the "golden") for
on-device component-by-component PCC parity.

Runs the vendored reference ``StructureModule`` on the paired 6yio H0-L0 Fv with the
real ``plddt-loss`` checkpoint and dumps, per IPA block, the exact (s, z, rot_mats, trans,
mask) inputs and the IPA update delta, plus the BackboneUpdate (s -> 6-dim) inputs, the
final single state feeding the pLDDT head, the pLDDT logits, and the atom14 positions.
The ttnn IPA / structure-module port PCC-gates against this golden (PCC > 0.98 per
component) — real weights, real inputs, not synthetic.

Run with the tt-bio env (CPU is fine — this is the reference, no device needed):
    python scripts/abb3_golden.py [--out ~/abb3_golden.pkl]

The checkpoint is resolved via tt_bio.abodybuilder3_weights (set $ABB3_CKPT to skip the
Zenodo download).
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from tt_bio.abodybuilder3_weights import ABB3_CONFIG, ensure_abb3_weights
from tt_bio._vendor.abodybuilder3.openfold.model.structure_module import StructureModule
from abodybuilder3_reference import string_to_input, EXAMPLE_HEAVY, EXAMPLE_LIGHT


def _capture_intermediates(model: StructureModule, heavy: str, light: str, device: str):
    """Monkeypatch each block submodule's forward to capture (args, output), run the
    structure-module forward, and return {block_i: {...}, "final": {...}}."""
    gold = {"blocks": [], "config": dict(ABB3_CONFIG)}
    ipa = model.ipa_layers
    bb = model.bb_update_layers
    ln = model.layer_norm_ipa_layers
    trans = model.transition_layers
    ang = model.angle_resnet_layers

    # IPA: capture (s, z, rot_mats, trans, mask) -> delta
    orig_ipa = [m.forward for m in ipa]

    def make_ipa_hook(i):
        def _fwd(s, z, r, mask=None, **kw):
            out = orig_ipa[i](s, z, r, mask, **kw)
            gold["blocks"].append({
                "ipa_s_in": s.detach().cpu().clone(),
                "ipa_z_in": z.detach().cpu().clone(),
                "ipa_rot_mats": r.get_rots().get_rot_mats().detach().cpu().clone(),
                "ipa_trans": r.get_trans().detach().cpu().clone(),
                "ipa_mask": mask.detach().cpu().clone() if mask is not None else None,
                "ipa_delta": out.detach().cpu().clone(),
            })
            return out
        return _fwd

    for i, m in enumerate(ipa):
        m.forward = make_ipa_hook(i)

    # LayerNorm after IPA: capture (post-ipa s) -> transition input
    orig_ln = [m.forward for m in ln]

    def make_ln_hook(i):
        def _fwd(x):
            out = orig_ln[i](x)
            gold["blocks"][i]["ln_s_in"] = x.detach().cpu().clone()
            gold["blocks"][i]["ln_s_out"] = out.detach().cpu().clone()  # transition input
            return out
        return _fwd

    for i, m in enumerate(ln):
        m.forward = make_ln_hook(i)

    # Transition: capture (layernorm out) -> transition out (bb_update input)
    orig_trans = [m.forward for m in trans]

    def make_trans_hook(i):
        def _fwd(s):
            out = orig_trans[i](s)
            gold["blocks"][i]["trans_s_in"] = s.detach().cpu().clone()
            gold["blocks"][i]["trans_s_out"] = out.detach().cpu().clone()  # == bb_s_in
            return out
        return _fwd

    for i, m in enumerate(trans):
        m.forward = make_trans_hook(i)

    # BackboneUpdate: capture s (post-transition) -> 6-dim update vector
    orig_bb = [m.forward for m in bb]

    def make_bb_hook(i):
        def _fwd(s):
            out = orig_bb[i](s)
            gold["blocks"][i]["bb_s_in"] = s.detach().cpu().clone()
            gold["blocks"][i]["bb_update"] = out.detach().cpu().clone()
            return out
        return _fwd

    for i, m in enumerate(bb):
        m.forward = make_bb_hook(i)

    # AngleResnet: capture (s, s_initial) -> (unnormalized_angles, angles)
    orig_ang = [m.forward for m in ang]

    def make_ang_hook(i):
        def _fwd(s, s_initial):
            unnorm, norm = orig_ang[i](s, s_initial)
            gold["blocks"][i]["ang_s_in"] = s.detach().cpu().clone()
            gold["blocks"][i]["ang_s_initial"] = s_initial.detach().cpu().clone()
            gold["blocks"][i]["ang_unnorm"] = unnorm.detach().cpu().clone()
            gold["blocks"][i]["ang_norm"] = norm.detach().cpu().clone()
            return unnorm, norm
        return _fwd

    for i, m in enumerate(ang):
        m.forward = make_ang_hook(i)

    inp = string_to_input(heavy, light, device)
    with torch.no_grad():
        out = model({"single": inp["single"], "pair": inp["pair"]}, inp["aatype"])

    gold["final"] = {
        "single": out["single"].detach().cpu().clone(),     # pLDDT head input
        "plddt_logits": out["plddt"].detach().cpu().clone(),  # (1, N, 50)
        "atom14": out["positions"][-1].detach().cpu().clone(),  # (1, N, 14, 3)
        "aatype": inp["aatype"].squeeze(0).cpu().clone() if inp["aatype"].dim() > 1 else inp["aatype"].cpu().clone(),
        "is_heavy": inp["is_heavy"].cpu().clone(),
        "n_res": inp["aatype"].shape[-1],
    }
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heavy", default=EXAMPLE_HEAVY)
    ap.add_argument("--light", default=EXAMPLE_LIGHT)
    ap.add_argument("--out", default=os.path.expanduser("~/abb3_golden.pkl"))
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    cache = Path(args.cache or os.environ.get("TT_BIO_CACHE", Path.home() / ".ttbio"))
    weights = ensure_abb3_weights(cache)
    state_dict = torch.load(weights, map_location="cpu", weights_only=True)
    model = StructureModule(**ABB3_CONFIG)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    gold = _capture_intermediates(model, args.heavy, args.light, "cpu")
    with open(args.out, "wb") as f:
        pickle.dump(gold, f)
    n = gold["final"]["n_res"]
    print(f"ABodyBuilder3 golden: {n} residues, {len(gold['blocks'])} IPA blocks")
    b0 = gold["blocks"][0]
    print(f"  block 0: ipa_s_in {tuple(b0['ipa_s_in'].shape)} ipa_z_in {tuple(b0['ipa_z_in'].shape)} "
          f"ipa_rot_mats {tuple(b0['ipa_rot_mats'].shape)} ipa_delta {tuple(b0['ipa_delta'].shape)}")
    print(f"           ln_s_out {tuple(b0['ln_s_out'].shape)} trans_s_out {tuple(b0['trans_s_out'].shape)} "
          f"bb_update {tuple(b0['bb_update'].shape)} ang_norm {tuple(b0['ang_norm'].shape)}")
    print(f"  final single {tuple(gold['final']['single'].shape)} "
          f"plddt_logits {tuple(gold['final']['plddt_logits'].shape)} "
          f"atom14 {tuple(gold['final']['atom14'].shape)}")
    print(f"  wrote -> {args.out}")


if __name__ == "__main__":
    main()
