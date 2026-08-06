"""Device-vs-reference PCC probe for an arbitrary target: runs the de-fixtured device
input pipeline (host_prep ai/cl0/plm0, glue, trunk) on the query and PCCs every
conditioning tensor against a reference dump produced by
``scripts/of3_refmodel_generic.py --dump-trunk``.

  PYTHONPATH=$WT TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=<p150 descriptor> \
  TT_BIO_LEASE_HOLDER=worker:tt-bio-openfold3-p13-msa-templates-e2e \
  /home/ttuser/tt-bio-dev/env/bin/python3 scripts/of3_trunk_pcc_generic.py \
    --query-json /tmp/p13_query_7XI5.json --ref-dump /tmp/p13_7xi5_ref_trunk.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")


def pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-json", required=True)
    ap.add_argument("--ref-dump", required=True)
    args = ap.parse_args()

    import ttnn

    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    from tt_bio.openfold3 import InputEmbedderGlue
    from tt_bio.openfold3_data import build_openfold3_features, make_openfold3_msa_features
    from tt_bio.openfold3_host_prep import (
        derive_block_aux, derive_relpos, derive_template_feat, ref_atom_embed,
        run_input_atom_encoder,
    )
    from tt_bio.openfold3_trunk import OF3Trunk
    from tt_bio.openfold3_weights import _sub
    from tt_bio.tenstorrent import get_device

    torch.manual_seed(0)
    np.random.seed(0)
    query = next(iter(InferenceQuerySet.from_json(args.query_json).queries.values()))
    features = build_openfold3_features(query)
    msa_feat = make_openfold3_msa_features(features, max_sequences=1024, seed=0)
    aux = derive_block_aux(features)
    template_feat = derive_template_feat(features)
    relpos = derive_relpos(features)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)

    ai = run_input_atom_encoder(dev, ckc, sd, features, aux)
    s_input = torch.cat(
        [ai, features["restype"], features["profile"],
         features["deletion_mean"].unsqueeze(-1)], dim=-1)
    cl0, plm0 = ref_atom_embed(
        _sub(sd, "diffusion_module.atom_attn_enc.ref_atom_feature_embedder"), features)

    ref = torch.load(args.ref_dump, map_location="cpu", weights_only=False)
    print(f"ai:      pcc={pcc(ai, ref['ai'].float()):.6f}")
    print(f"cl0:     pcc={pcc(cl0, ref['cl0'].float()):.6f}")
    print(f"plm0:    pcc={pcc(plm0, ref['plm0'].float()):.6f}")
    print(f"s_input: pcc={pcc(s_input, ref['s_input'].float()):.6f}")

    def ft(x):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)

    glue = InputEmbedderGlue(_sub(sd, "input_embedder"), ckc)
    trunk = OF3Trunk(sd, ckc, num_cycles=4)
    tb = features["token_bonds"]
    tb_d = ttnn.from_torch(tb.float().unsqueeze(0).unsqueeze(-1),
                           layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    s_init_d, z_init_d = glue(ft(s_input.unsqueeze(0)), ft(relpos.unsqueeze(0)), tb_d)
    tmpl_d = {k: ft(v) for k, v in template_feat.items()}
    s_trunk_d, z_trunk_d = trunk(
        s_init_d, z_init_d, tmpl_d, ft(msa_feat.unsqueeze(0)), ft(s_input.unsqueeze(0)))
    n_token = aux["n_token"]
    s_trunk = ttnn.to_torch(s_trunk_d).float().reshape(n_token, -1)
    z_trunk = ttnn.to_torch(z_trunk_d).float().reshape(n_token, n_token, -1)
    print(f"s_trunk: pcc={pcc(s_trunk, ref['s_trunk'].float()):.6f}")
    print(f"z_trunk: pcc={pcc(z_trunk, ref['z_trunk'].float()):.6f}")


if __name__ == "__main__":
    main()
