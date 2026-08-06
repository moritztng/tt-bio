"""P13/H1 device leg: dump the device glue + trunk outputs exactly as the S0 fold
produced them (fixture-hybrid s_input, fixture relpos, empty fixture template block,
tt-bio 1024-row MSA draw), for the cross-stack hybrid experiment:

  device (s_input, s_trunk, z_trunk) -> reference CPU _rollout -> RMSD.

If that lands ~1 A the device trunk outputs are functionally sufficient and the 7-9 A
defect lives strictly in the device sampler chain (conditioning / DiffusionModule /
decoder / host rollout maths). If it lands 7-9 A the trunk outputs, despite
PCC 0.9994, are functionally broken.

Run with the tt-bio device env (card 0):
  PYTHONPATH=$WT TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=<p150 descriptor> \
    TT_BIO_LEASE_HOLDER=worker:tt-bio-openfold3-p13-msa-templates-e2e \
    /home/ttuser/tt-bio-dev/env/bin/python3 scripts/of3_device_trunk_dump.py
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
MSA_DIR = os.path.join(REPO, ".artifacts/msa")
OUT = os.path.expanduser("~/p13_h1_device_trunk.pt")


def main():
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    from tt_bio.openfold3_data import (
        build_openfold3_features,
        make_openfold3_msa_features,
    )
    from tt_bio.openfold3 import InputEmbedderGlue
    from tt_bio.openfold3_trunk import OF3Trunk
    from tt_bio.openfold3_weights import _sub

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    I = pickle.load(open(GOLD, "rb"))["intermediates"]
    ie = I["input_embedder_real"]
    te = I["template_embedder_real"]["feat"]
    atom_enc = I["input_embedder_atom_enc_real"]

    torch.manual_seed(0)
    np.random.seed(0)
    query = next(iter(InferenceQuerySet.from_json(QUERY).queries.values()))
    # Cache is warm: attach the cached MSA directly (no RNG, no tt_bio.main import).
    import hashlib
    from pathlib import Path

    for chain in query.chains:
        if chain.molecule_type.name != "PROTEIN" or chain.main_msa_file_paths:
            continue
        seq_hash = hashlib.sha256(chain.sequence.encode()).hexdigest()[:16]
        chain.main_msa_file_paths = [
            Path(MSA_DIR) / "of3" / seq_hash / "colabfold_main.a3m"
        ]
    features = build_openfold3_features(query)
    assert int(features["msa"].shape[0]) == 2734, "featurization drift vs S0"
    msa_feat = make_openfold3_msa_features(features, max_sequences=1024, seed=0)

    ai = atom_enc["out"][0].float()
    s_input = torch.cat(
        [ai, features["restype"], features["profile"], features["deletion_mean"].unsqueeze(-1)],
        dim=-1,
    )

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)

    def ft(x):
        return ttnn.from_torch(x.float().unsqueeze(0), layout=ttnn.TILE_LAYOUT,
                               device=dev, dtype=ttnn.bfloat16)

    glue = InputEmbedderGlue(_sub(sd, "input_embedder"), ckc)
    tb_d = ttnn.from_torch(
        features["token_bonds"].float().unsqueeze(0).unsqueeze(-1),
        layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    s_init_d, z_init_d = glue(ft(s_input), ft(ie["relpos"]), tb_d)

    trunk = OF3Trunk(sd, ckc, num_cycles=4)
    tmpl_d = {k: ttnn.from_torch(v.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                 dtype=ttnn.bfloat16) for k, v in te.items()}
    s_trunk_d, z_trunk_d = trunk(s_init_d, z_init_d, tmpl_d, ft(msa_feat), ft(s_input))

    out = {
        "s_input": s_input,
        "s_init": ttnn.to_torch(s_init_d).float().reshape(s_input.shape[0], -1),
        "z_init": ttnn.to_torch(z_init_d).float().reshape(76, 76, -1),
        "s_trunk": ttnn.to_torch(s_trunk_d).float().reshape(76, -1),
        "z_trunk": ttnn.to_torch(z_trunk_d).float().reshape(76, 76, -1),
    }
    for k, v in out.items():
        print(k, tuple(v.shape), f"std={float(v.std()):.4f}")
    torch.save(out, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
