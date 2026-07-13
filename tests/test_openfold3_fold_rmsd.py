"""OpenFold3 searched-MSA + confidence-ranked 1UBQ accuracy gate."""
import hashlib
import os
import pickle

import numpy as np
import pytest
import torch

_CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
_GOLD = os.path.expanduser("~/of3_ref_out.pkl")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_QUERY = os.path.join(_REPO, "tests/fixtures/of3_ubiquitin_query.json")
_GT_PDB = os.path.join(_REPO, "examples/ground_truth_structures/ubiquitin.pdb")
_CA_MASK = os.path.join(_REPO, "tests/fixtures/of3_ubiquitin_ca_mask.npy")
_SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
_MSA_DIR = os.path.expanduser(os.environ.get("OF3_MSA_DIR", "~/.boltz/msa"))
_MSA = os.path.join(_MSA_DIR, hashlib.sha256(_SEQ.encode()).hexdigest()[:16] + ".a3m")
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (_CKPT, _GOLD, _QUERY, _GT_PDB, _CA_MASK, _MSA)),
    reason="OF3 checkpoint, golden, 1UBQ, or cached ubiquitin MSA missing")


def test_of3_msa_confidence_selected_fold_rmsd():
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import InferenceQuerySet
    from tt_bio.openfold3_data import (
        build_openfold3_features, make_openfold3_msa_features, resolve_openfold3_msas,
    )
    from tt_bio.openfold3_fold import OpenFold3, kabsch_rmsd, load_pdb_ca

    torch.manual_seed(0)
    np.random.seed(0)
    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    I = pickle.load(open(_GOLD, "rb"))["intermediates"]
    ie = I["input_embedder_real"]
    te = I["template_embedder_real"]["feat"]
    xl = I["diffusion_module_xlout_real"]
    dec = I["diffusion_decoder_real"]
    atom_transformer = I["input_embedder_atom_transformer_real"]
    atom_encoder = I["input_embedder_atom_enc_real"]
    conf_g = I["confidence_heads_real"]
    tr = I["trunk_real"]

    query = next(iter(InferenceQuerySet.from_json(_QUERY).queries.values()))
    resolve_openfold3_msas(query, _MSA_DIR, target_id="ubiquitin")
    features = build_openfold3_features(query)
    msa_feat = make_openfold3_msa_features(features, max_sequences=1024, seed=0)
    s_input = torch.cat([
        atom_encoder["out"][0].float(), features["restype"], features["profile"],
        features["deletion_mean"].unsqueeze(-1)], dim=-1)

    n_atom, n_token, nb, NP = xl["n_atom"], xl["n_token"], xl["nb"], xl["NP"]
    dm_aux = dict(
        cl0=xl["cl0"], plm0=xl["plm0"], atom_mask=dec["atom_mask"],
        atom_to_token_index=dec["atom_to_token_index"],
        npe_q_indices=xl["npe_q_indices"], npe_k_indices=xl["npe_k_indices"],
        zij_mask=xl["zij_mask"], key_block_idxs=dec["key_block_idxs"],
        invalid_mask=dec["invalid_mask"], mask_trunked=dec["mask_trunked"],
        atom_to_token_mean=atom_transformer["atom_to_token_mean"], nb=nb, NP=NP)
    ca_mask = np.load(_CA_MASK).astype(bool)
    a2t = dec["atom_to_token_index"].long()
    polymer_token = (features["is_protein"] | features["is_rna"] | features["is_dna"]).bool()
    confidence_aux = dict(
        representative_atom_indices=torch.from_numpy(np.flatnonzero(ca_mask)).long(),
        max_atom_per_token_mask=conf_g["max_atom_per_token_mask"],
        atom_array=features["atom_array"], asym_id=features["asym_id"],
        atom_to_token_index=a2t, atom_mask=features["atom_mask"].bool(),
        polymer_mask=polymer_token[a2t])

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    result = OpenFold3(sd, ckc, num_cycles=tr["num_cycles"]).fold(
        template_feat=te, msa_feat=msa_feat, s_input=s_input, relpos=ie["relpos"],
        token_bonds=features["token_bonds"], token_mask=features["token_mask"],
        dm_aux_host=dm_aux, n_atom=n_atom, n_token=n_token,
        no_rollout_steps=200, seed=1234, no_samples=5,
        confidence_aux_host=confidence_aux)

    assert msa_feat.shape[0] == 1024
    assert all(torch.isfinite(sample).all() for sample in result.samples)
    assert result.best_index == max(
        range(5), key=lambda i: result.confidence[i]["ranking_score"])
    gt_ca = load_pdb_ca(_GT_PDB)
    rmsds = [kabsch_rmsd(sample[ca_mask].double(), gt_ca) for sample in result.samples]
    selected = rmsds[result.best_index]
    print(f"OF3 MSA+confidence: selected={selected:.3f} A best={min(rmsds):.3f} A "
          f"median={sorted(rmsds)[2]:.3f} A index={result.best_index}")
    assert min(rmsds) < 10.0, f"MSA ensemble best RMSD {min(rmsds):.2f} A regressed"
    assert selected < 12.0, f"confidence-selected RMSD {selected:.2f} A regressed"
