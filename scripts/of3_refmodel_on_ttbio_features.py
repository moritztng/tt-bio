"""Run C (decisive host-side bisect): the upstream reference OpenFold3 model, driven
directly, fed tt-bio's OWN featurization of 1UBQ (vendored pipeline + cached ColabFold
MSA, production settings).

Decision:
  - lands ~1 A   -> tt-bio host featurization is exonerated (incl. the subsample_main
                    draw, modulo row selection); the 7-9 A defect is device-side
                    (conditioning / sampler / decoder chain).
  - lands 7-9 A  -> the defect is in build_openfold3_features itself; diff the batch
                    against the reference featurizer tensor-by-tensor.

Variants:
  --subsample-main {0,1}  production default is 1 (the bare MSASettings bug); 0 = the
                          upstream inference setting (plan S4).

Run with the CPU reference venv:
    /tmp/of3-venv/bin/python scripts/of3_refmodel_on_ttbio_features.py
"""
import argparse
import os
import sys

import numpy as np
import torch

OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
WT = os.environ.get(
    "WT", "/home/ttuser/.coworker/wt/tt-bio-openfold3-p13-msa-templates-e2e"
)
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
GT = os.path.join(WT, "examples/ground_truth_structures/ubiquitin.pdb")
MSA_FILE = os.path.join(
    WT, ".artifacts/msa/of3/233b4b0b8c461609/colabfold_main.a3m"
)
sys.path.insert(0, WT)
sys.path.insert(0, OF3_REF)


def kabsch_rmsd(pred_ca, gt_ca):
    p = pred_ca.double() - pred_ca.double().mean(0)
    g = gt_ca.double() - gt_ca.double().mean(0)
    u, _, vt = torch.linalg.svd(p.t() @ g)
    d = torch.sign(torch.det(vt.t() @ u.t()))
    s = torch.eye(3, dtype=torch.float64)
    s[2, 2] = d
    p_aligned = p @ (vt.t() @ s @ u.t()).t()
    return float(torch.sqrt(((p_aligned - g) ** 2).sum(-1).mean()))


def load_pdb_ca(pdb_path):
    pts, seen = [], set()
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            uid = (line[21].strip(), int(line[22:26].strip()))
            if uid in seen:
                continue
            seen.add(uid)
            pts.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return torch.tensor(pts, dtype=torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample-main", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.projects.of3_all_atom.model import OpenFold3

    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.dataset_config_components import (
        MSASettings,
    )
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    from tt_bio.openfold3_data import build_openfold3_features

    query_json = {
        "queries": {
            "ubiquitin": {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
                        "main_msa_file_paths": [MSA_FILE],
                    }
                ]
            }
        }
    }
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False
    ) as f:
        json.dump(query_json, f)
        qpath = f.name
    query = next(iter(InferenceQuerySet.from_json(qpath).queries.values()))

    msa_settings = MSASettings(subsample_main=bool(args.subsample_main))
    features = build_openfold3_features(query, msa_settings=msa_settings)
    print(
        f"features: msa rows={int(features['msa'].shape[0])} "
        f"n_token={int(features['token_mask'].shape[0])} "
        f"n_atom={int(features['atom_mask'].shape[0])} subsample_main={args.subsample_main}"
    )

    batch = {}
    for k, v in features.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0)
    # scalar-ish entries the model may need
    for k in ("num_paired_seqs",):
        if k in features and not torch.is_tensor(features[k]):
            batch[k] = torch.tensor([features[k]])

    C.settings.memory.eval.use_triton_triangle_kernels = False
    C.settings.memory.eval.use_deepspeed_evo_attention = False
    C.settings.memory.eval.use_cueq_triangle_kernels = False

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = OpenFold3(config=C).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 2 or unexpected:
        print("  missing:", missing[:5], " unexpected:", unexpected[:5])

    with torch.no_grad():
        s_input, s_trunk, z_trunk = model.run_trunk(batch, num_cycles=4)
        print(
            f"trunk: s_input std {float(s_input.std()):.4f} "
            f"s std {float(s_trunk.std()):.4f} z std {float(z_trunk.std()):.4f}"
        )
        # Mirror OpenFold3.forward's eval branch: sample dim at axis 1 on both the
        # representations and every batch tensor (except ref_space_uid_to_perm).
        from openfold3.core.utils.tensor_utils import tensor_tree_map

        s_input = s_input.unsqueeze(1)
        s_trunk = s_trunk.unsqueeze(1)
        z_trunk = z_trunk.unsqueeze(1)
        perm = batch.pop("ref_space_uid_to_perm", None)
        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)
        if perm is not None:
            batch["ref_space_uid_to_perm"] = perm
        out = model._rollout(batch, s_input, s_trunk, z_trunk)

    xl = out["atom_positions_predicted"].float()
    xl = xl.reshape(-1, xl.shape[-2], xl.shape[-1])  # [n_samples, n_atom, 3]
    print("samples:", tuple(xl.shape))

    atom_array = features["atom_array"]
    ca_mask = torch.from_numpy(atom_array.atom_name == "CA")
    gt_ca = load_pdb_ca(GT)
    assert int(ca_mask.sum()) == gt_ca.shape[0], (int(ca_mask.sum()), gt_ca.shape)

    rmsds = []
    for i in range(xl.shape[0]):
        r = kabsch_rmsd(xl[i][ca_mask].double(), gt_ca)
        rmsds.append(r)
        print(f"sample {i}: RMSD={r:.4f} A")
    rmsds_s = sorted(rmsds)
    print(
        f"RESULT-C(subsample_main={args.subsample_main}): best={rmsds_s[0]:.4f} "
        f"median={rmsds_s[len(rmsds_s)//2]:.4f} worst={rmsds_s[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
