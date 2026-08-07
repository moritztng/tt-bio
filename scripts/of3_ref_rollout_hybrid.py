"""P13/H0+H1 (cross-stack bisect): reference CPU _rollout fed trunk outputs from
different sources, all on tt-bio's featurization of 1UBQ.

  --src s1cpu    H0: ~/p13_s1_trunk_msa.pt  (CPU reference trunk on tt-bio's exact
                   1024-row draw, constant-m) -> reference rollout. Expect ~1 A
                   (exonerates tt-bio row selection + constant-m).
  --src device   H1: ~/p13_h1_device_trunk.pt (device glue+trunk, fixture-hybrid
                   s_input) -> reference rollout. ~1 A => device trunk functionally
                   sufficient, defect strictly in the device sampler chain.

Run with the CPU reference venv:
    /tmp/of3-venv/bin/python scripts/of3_ref_rollout_hybrid.py --src device
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

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


def build_batch():
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    from tt_bio.openfold3_data import build_openfold3_features

    torch.manual_seed(0)
    np.random.seed(0)
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
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(query_json, f)
        qpath = f.name
    query = next(iter(InferenceQuerySet.from_json(qpath).queries.values()))
    features = build_openfold3_features(query)
    assert int(features["msa"].shape[0]) == 2734, "featurization drift vs S0"
    batch = {k: v.unsqueeze(0) for k, v in features.items() if torch.is_tensor(v)}
    return features, batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=["s1cpu", "device"], required=True)
    args = ap.parse_args()

    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.projects.of3_all_atom.model import OpenFold3
    from openfold3.core.utils.tensor_utils import tensor_tree_map

    features, batch = build_batch()

    if args.src == "s1cpu":
        d = torch.load(os.path.expanduser("~/p13_s1_trunk_msa.pt"),
                       map_location="cpu", weights_only=False)
        s_input, s_trunk, z_trunk = d["s_input_ref"], d["s_trunk_fixed"], d["z_trunk_fixed"]
    else:
        d = torch.load(os.path.expanduser("~/p13_h1_device_trunk.pt"),
                       map_location="cpu", weights_only=False)
        s_input, s_trunk, z_trunk = d["s_input"], d["s_trunk"], d["z_trunk"]

    C.settings.memory.eval.use_triton_triangle_kernels = False
    C.settings.memory.eval.use_deepspeed_evo_attention = False
    C.settings.memory.eval.use_cueq_triangle_kernels = False

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = OpenFold3(config=C).eval()
    model.load_state_dict(sd, strict=False)

    # Mirror OpenFold3.forward's eval branch: sample dim at axis 1.
    s_input = s_input.unsqueeze(0).unsqueeze(0)
    s_trunk = s_trunk.unsqueeze(0).unsqueeze(0)
    z_trunk = z_trunk.unsqueeze(0).unsqueeze(0)
    perm = batch.pop("ref_space_uid_to_perm", None)
    batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)
    if perm is not None:
        batch["ref_space_uid_to_perm"] = perm

    torch.manual_seed(1234)
    with torch.no_grad():
        out = model._rollout(batch, s_input, s_trunk, z_trunk)

    xl = out["atom_positions_predicted"].float()
    xl = xl.reshape(-1, xl.shape[-2], xl.shape[-1])
    atom_array = features["atom_array"]
    ca_mask = torch.from_numpy(atom_array.atom_name == "CA")
    gt_ca = load_pdb_ca(GT)
    assert int(ca_mask.sum()) == gt_ca.shape[0]

    rmsds = []
    for i in range(xl.shape[0]):
        r = kabsch_rmsd(xl[i][ca_mask].double(), gt_ca)
        rmsds.append(r)
        print(f"sample {i}: RMSD={r:.4f} A")
    rmsds_s = sorted(rmsds)
    print(
        f"RESULT-H(src={args.src}): best={rmsds_s[0]:.4f} "
        f"median={rmsds_s[len(rmsds_s)//2]:.4f} worst={rmsds_s[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
