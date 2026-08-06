"""S3 acceptance check: every tensor the de-fixtured accuracy path derives host-side
vs the 1UBQ golden fixture it replaces.

Reports exact-match / maxdiff / PCC per tensor and the device-run ``ai`` PCC. The
fixture is used HERE only -- ``scripts/of3_fold_rmsd.py`` itself reads no golden.

Run with the device env:
  PYTHONPATH=$WT TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=<p150 descriptor> \
  TT_BIO_LEASE_HOLDER=worker:tt-bio-openfold3-p13-msa-templates-e2e \
  /home/ttuser/tt-bio-dev/env/bin/python3 scripts/of3_host_prep_check.py
"""
import os
import pickle
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
GOLD = os.path.expanduser("~/of3_ref_out.pkl")
QUERY = os.path.join(REPO, "tests/fixtures/of3_ubiquitin_query.json")
CA_MASK = os.path.join(REPO, "tests/fixtures/of3_ubiquitin_ca_mask.npy")
MSA_DIR = os.path.join(REPO, ".artifacts/msa")


def pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def main():
    import hashlib
    from pathlib import Path

    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    from tt_bio.openfold3_data import build_openfold3_features
    from tt_bio.openfold3_host_prep import (
        derive_block_aux, derive_relpos, derive_template_feat, ref_atom_embed,
        run_input_atom_encoder,
    )
    from tt_bio.openfold3_weights import _sub

    torch.manual_seed(0)
    np.random.seed(0)
    query = next(iter(InferenceQuerySet.from_json(QUERY).queries.values()))
    for chain in query.chains:
        if chain.molecule_type.name == "PROTEIN" and not chain.main_msa_file_paths:
            seq_hash = hashlib.sha256(chain.sequence.encode()).hexdigest()[:16]
            chain.main_msa_file_paths = [
                Path(MSA_DIR) / "of3" / seq_hash / "colabfold_main.a3m"
            ]
    features = build_openfold3_features(query)
    I = pickle.load(open(GOLD, "rb"))["intermediates"]

    te = derive_template_feat(features)
    te_ref = I["template_embedder_real"]["feat"]
    for k, v in te_ref.items():
        d = (te[k] - v.float()).abs().max()
        print(f"te[{k}]: maxdiff={float(d):.6f} {'EXACT' if float(d) == 0 else ''}")

    relpos = derive_relpos(features)
    relpos_ref = I["input_embedder_real"]["relpos"].float()
    print(f"relpos: shape {tuple(relpos.shape)} vs {tuple(relpos_ref.shape)} "
          f"maxdiff={float((relpos - relpos_ref).abs().max()):.6f}")

    aux = derive_block_aux(features)
    xl = I["diffusion_module_xlout_real"]
    dec = I["diffusion_decoder_real"]
    at = I["input_embedder_atom_transformer_real"]
    conf = I["confidence_heads_real"]
    print(f"n_atom {aux['n_atom']} vs fixture {xl['n_atom']}; n_token {aux['n_token']} vs "
          f"{xl['n_token']}; nb {aux['nb']} vs {xl['nb']}; NP {aux['NP']} vs {xl['NP']}")
    for name, mine, ref in [
        ("atom_mask", aux["atom_mask"], dec["atom_mask"].float()),
        ("atom_to_token_index", aux["atom_to_token_index"].float(), dec["atom_to_token_index"].float()),
        ("npe_q_indices", aux["npe_q_indices"].float(), xl["npe_q_indices"].float()),
        ("npe_k_indices", aux["npe_k_indices"].float(), xl["npe_k_indices"].float()),
        ("zij_mask", aux["zij_mask"], xl["zij_mask"].float()),
        ("key_block_idxs", aux["key_block_idxs"].float(), dec["key_block_idxs"].float()),
        ("invalid_mask", aux["invalid_mask"].float(), dec["invalid_mask"].float()),
        ("mask_trunked", aux["mask_trunked"], dec["mask_trunked"].float()),
        ("atom_to_token_mean", aux["atom_to_token_mean"], at["atom_to_token_mean"].float()),
        ("max_atom_per_token_mask", aux["max_atom_per_token_mask"], conf["max_atom_per_token_mask"].float()),
    ]:
        same = torch.equal(mine, ref)
        d = float((mine - ref).abs().max())
        print(f"{name}: {'EXACT' if same else f'maxdiff={d:.6f}'}")

    ca_ref = np.load(CA_MASK).astype(bool)
    ca_mine = aux["ca_mask"].numpy()
    print(f"ca_mask: {'EXACT' if (ca_mine == ca_ref).all() else 'MISMATCH'} "
          f"({int(ca_mine.sum())} vs {int(ca_ref.sum())} Ca atoms)")

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    cl0, plm0 = ref_atom_embed(
        _sub(sd, "diffusion_module.atom_attn_enc.ref_atom_feature_embedder"), features)
    cl0_ref, plm0_ref = xl["cl0"].float(), xl["plm0"].float()
    print(f"cl0: pcc={pcc(cl0, cl0_ref):.6f} maxdiff={float((cl0 - cl0_ref).abs().max()):.6f}")
    print(f"plm0: pcc={pcc(plm0, plm0_ref):.6f} maxdiff={float((plm0 - plm0_ref).abs().max()):.6f}")

    import ttnn
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    ai = run_input_atom_encoder(dev, ckc, sd, features, aux)
    ai_ref = I["input_embedder_atom_enc_real"]["out"][0].float()
    print(f"ai: shape {tuple(ai.shape)} vs {tuple(ai_ref.shape)} "
          f"pcc={pcc(ai, ai_ref):.6f} maxdiff={float((ai - ai_ref).abs().max()):.6f}")


if __name__ == "__main__":
    main()
