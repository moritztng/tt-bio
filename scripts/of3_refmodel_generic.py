"""Run C generalized: the upstream reference OpenFold3 model on CPU, driven directly,
fed tt-bio's OWN featurization of an arbitrary query json (vendored pipeline, current
inference defaults). Discriminates host-featurization defects from device-port defects
for any target.

Run with the combined env:
  PYTHONPATH=/tmp/p13/pylibs:$WT:/tmp/of3-ref /home/ttuser/tt-bio-dev/env/bin/python3 \
    scripts/of3_refmodel_generic.py --query-json /tmp/p13_query_7XI5.json \
    --gt-cif ~/of3-bench/7xi5-assembly1.cif
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
sys.path.insert(0, WT)
sys.path.insert(0, OF3_REF)
sys.path.insert(0, os.path.join(WT, "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-json", required=True)
    ap.add_argument("--gt-cif", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-trunk", default=None,
                    help="dump reference s_input/s_trunk/z_trunk + atom-encoder "
                         "cl/plm/ai (pre-rollout batch) to this .pt path")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.projects.of3_all_atom.model import OpenFold3

    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    from tt_bio.openfold3_data import build_openfold3_features

    query = next(iter(InferenceQuerySet.from_json(args.query_json).queries.values()))
    features = build_openfold3_features(query)
    print(
        f"features: msa rows={int(features['msa'].shape[0])} "
        f"n_token={int(features['token_mask'].shape[0])} "
        f"n_atom={int(features['atom_mask'].shape[0])}"
    )

    batch = {}
    for k, v in features.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0)
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

    with torch.no_grad():
        s_input, s_trunk, z_trunk = model.run_trunk(batch, num_cycles=4)
        print(
            f"trunk: s_input std {float(s_input.std()):.4f} "
            f"s std {float(s_trunk.std()):.4f} z std {float(z_trunk.std()):.4f}"
        )
        if args.dump_trunk:
            dm_rafe = model.diffusion_module.atom_attn_enc.ref_atom_feature_embedder
            cl0, plm0 = dm_rafe(batch=batch, n_query=32, n_key=128)
            ai, _, _, _ = model.input_embedder.atom_attn_enc(batch=batch)
            torch.save(
                {"s_input": s_input[0], "s_trunk": s_trunk[0], "z_trunk": z_trunk[0],
                 "cl0": cl0[0], "plm0": plm0[0], "ai": ai[0]},
                args.dump_trunk)
            print(f"dumped reference trunk/conditioning tensors -> {args.dump_trunk}")
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
    xl = xl.reshape(-1, xl.shape[-2], xl.shape[-1])

    from of3_bench_s5 import aligned_ca_rmsd

    import biotite.sequence as _seq

    atom_array = features["atom_array"]
    ca_mask = torch.from_numpy(atom_array.atom_name == "CA")
    pred_ca_seq = "".join(
        _seq.ProteinSequence.convert_letter_3to1(r)
        for r in atom_array.res_name[ca_mask.numpy()]
    )
    rmsds = []
    for i in range(xl.shape[0]):
        r, n_aln = aligned_ca_rmsd(xl[i][ca_mask], pred_ca_seq, args.gt_cif)
        rmsds.append(r)
        print(f"sample {i}: RMSD={r:.4f} A (aligned {n_aln} Ca)")
    rmsds_s = sorted(rmsds)
    print(
        f"RESULT-C-generic: best={rmsds_s[0]:.4f} "
        f"median={rmsds_s[len(rmsds_s)//2]:.4f} worst={rmsds_s[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
