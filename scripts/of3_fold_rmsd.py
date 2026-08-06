"""OpenFold3 end-to-end MSA + confidence-selection accuracy gate.

The benchmark reuses the production MSA search/cache stage, featurizes the resulting
alignment with the vendored OF3 pipeline, runs the full 200-step x 5-sample device
fold, and reports both oracle best-of-5 and the sample selected by the OF3 ranking
score.

Fully de-fixtured (P13/S3): every model input is derived from the featurized query --
template feature block, relpos, atom-encoder output ``ai``, and all diffusion-module
block/mask auxiliaries come from ``tt_bio.openfold3_host_prep`` (validated against the
retired golden fixture by ``scripts/of3_host_prep_check.py``). No golden intermediates
appear anywhere in the accuracy path.
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tt_bio.openfold3_fold import OpenFold3, kabsch_rmsd, load_pdb_ca

_CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
_QUERY = os.path.join(REPO, "tests/fixtures/of3_ubiquitin_query.json")
_GT_PDB = os.path.join(REPO, "examples/ground_truth_structures/ubiquitin.pdb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rollout-steps", type=int, default=200)
    ap.add_argument("--no-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--msa-dir", default=os.path.join(REPO, ".artifacts/msa"))
    ap.add_argument("--msa-db-path", default=None)
    ap.add_argument("--msa-server-url", default="https://api.colabfold.com")
    ap.add_argument("--query-json", default=_QUERY)
    ap.add_argument("--gt-pdb", default=_GT_PDB)
    ap.add_argument("--target-name", default="ubiquitin(1UBQ)")
    args = ap.parse_args()

    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import InferenceQuerySet
    from tt_bio.openfold3_data import (
        build_openfold3_features, make_openfold3_msa_features, resolve_openfold3_msas,
    )
    from tt_bio.openfold3_host_prep import (
        derive_block_aux, derive_relpos, derive_template_feat, ref_atom_embed,
        run_input_atom_encoder,
    )
    from tt_bio.openfold3_weights import _sub

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)

    torch.manual_seed(0)
    np.random.seed(0)
    query = next(iter(InferenceQuerySet.from_json(args.query_json).queries.values()))
    resolve_openfold3_msas(
        query, args.msa_dir, target_id=args.target_name.split("(")[0],
        msa_db_path=args.msa_db_path, msa_server_url=args.msa_server_url,
    )
    features = build_openfold3_features(query)
    msa_feat = make_openfold3_msa_features(features, max_sequences=1024, seed=0)

    aux = derive_block_aux(features)
    template_feat = derive_template_feat(features)
    relpos = derive_relpos(features)

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)

    ai = run_input_atom_encoder(dev, ckc, sd, features, aux)
    s_input = torch.cat(
        [ai, features["restype"], features["profile"], features["deletion_mean"].unsqueeze(-1)],
        dim=-1,
    )
    cl0, plm0 = ref_atom_embed(
        _sub(sd, "diffusion_module.atom_attn_enc.ref_atom_feature_embedder"), features)

    n_atom = aux["n_atom"]; n_token = aux["n_token"]
    dm_aux_host = dict(
        cl0=cl0, plm0=plm0, atom_mask=aux["atom_mask"],
        atom_to_token_index=aux["atom_to_token_index"],
        npe_q_indices=aux["npe_q_indices"], npe_k_indices=aux["npe_k_indices"],
        zij_mask=aux["zij_mask"], key_block_idxs=aux["key_block_idxs"],
        invalid_mask=aux["invalid_mask"], mask_trunked=aux["mask_trunked"],
        atom_to_token_mean=aux["atom_to_token_mean"], nb=aux["nb"], NP=aux["NP"])

    ca_mask = aux["ca_mask"]
    representative_atom_indices = torch.from_numpy(np.flatnonzero(ca_mask.numpy())).long()
    atom_to_token = aux["atom_to_token_index"].long()
    polymer_token = (features["is_protein"] | features["is_rna"] | features["is_dna"]).bool()
    confidence_aux = dict(
        representative_atom_indices=representative_atom_indices,
        max_atom_per_token_mask=aux["max_atom_per_token_mask"],
        atom_array=features["atom_array"], asym_id=features["asym_id"],
        atom_to_token_index=atom_to_token, atom_mask=features["atom_mask"].bool(),
        polymer_mask=polymer_token[atom_to_token],
    )

    raw_rows = int(features["msa"].shape[0])
    print(f"MSA: raw_rows={raw_rows} selected_rows={msa_feat.shape[0]} cache={args.msa_dir}")
    model = OpenFold3(sd, ckc, num_cycles=4)
    result = model.fold(
        template_feat=template_feat, msa_feat=msa_feat, s_input=s_input,
        relpos=relpos, token_bonds=features["token_bonds"],
        token_mask=features["token_mask"], dm_aux_host=dm_aux_host,
        n_atom=n_atom, n_token=n_token, no_rollout_steps=args.no_rollout_steps,
        seed=args.seed, no_samples=args.no_samples, confidence_aux_host=confidence_aux)

    gt_ca = load_pdb_ca(args.gt_pdb)
    rmsds = []
    for i, sample in enumerate(result.samples):
        if not torch.isfinite(sample).all():
            raise RuntimeError(f"sample {i} has non-finite coordinates")
        rmsd = kabsch_rmsd(sample[ca_mask].double(), gt_ca)
        rmsds.append(rmsd)
        c = result.confidence[i]
        ranking, ptm = c["ranking_score"], c["ptm"]
        plddt, disorder = c["plddt"], c["disorder"]
        print(f"sample {i}: RMSD={rmsd:.4f} A ranking={ranking:.6f} "
              f"pTM={ptm:.6f} pLDDT={plddt:.6f} disorder={disorder:.6f}")

    selected = rmsds[result.best_index]
    ordered = sorted(rmsds)
    print(f"RESULT: selected_sample={result.best_index} selected_RMSD={selected:.4f} A "
          f"oracle_best={ordered[0]:.4f} A median={ordered[len(ordered)//2]:.4f} A "
          f"max={ordered[-1]:.4f} A")
    print(f"CONFIG: MSA rows={msa_feat.shape[0]} rollout_steps={args.no_rollout_steps} "
          f"samples={args.no_samples} seed={args.seed} target={args.target_name}")


if __name__ == "__main__":
    main()
