"""7XI5 s-track bisect: per-cycle, per-stage PCC of the device trunk vs the reference
trunk, in one process (combined env).

Reference capture: forward hooks on ``model.msa_module`` and
``model.pairformer_stack`` during ``run_trunk(num_cycles=4)`` give, per cycle,
z-after-MSA-module (pairformer input z), s-after-glue (pairformer input s), and
s/z-after-pairformer. Device capture: the ``OF3Trunk`` cycle loop is replicated
manually with the same submodules so the identical intermediates are pulled off the
device each cycle.

  PYTHONPATH=/tmp/p13/pylibs:$WT:/tmp/of3-ref TT_VISIBLE_DEVICES=0 \
  TT_MESH_GRAPH_DESC_PATH=<p150 descriptor> \
  TT_BIO_LEASE_HOLDER=worker:tt-bio-openfold3-p13-msa-templates-e2e \
  /home/ttuser/tt-bio-dev/env/bin/python3 scripts/of3_strunk_cycle_bisect.py \
    --query-json /tmp/p13_query_7XI5.json
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
OF3_REF = os.environ.get("OF3_REF", "/tmp/of3-ref")
sys.path.insert(0, OF3_REF)

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
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    from openfold3.projects.of3_all_atom.config.model_config import model_config as C
    from openfold3.projects.of3_all_atom.model import OpenFold3 as RefOpenFold3

    from tt_bio._vendor.openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    from tt_bio.openfold3_data import build_openfold3_features, make_openfold3_msa_features
    from tt_bio.openfold3_host_prep import (
        derive_block_aux, derive_relpos, derive_template_feat, run_input_atom_encoder,
    )

    query = next(iter(InferenceQuerySet.from_json(args.query_json).queries.values()))
    features = build_openfold3_features(query)
    msa_feat = make_openfold3_msa_features(features, max_sequences=1024, seed=0)
    aux = derive_block_aux(features)
    template_feat = derive_template_feat(features)
    relpos = derive_relpos(features)
    n_token = aux["n_token"]

    batch = {}
    for k, v in features.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0)

    C.settings.memory.eval.use_triton_triangle_kernels = False
    C.settings.memory.eval.use_deepspeed_evo_attention = False
    C.settings.memory.eval.use_cueq_triangle_kernels = False

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    ref_model = RefOpenFold3(config=C).eval()
    ref_model.load_state_dict(sd, strict=False)

    # ---- reference per-cycle capture -------------------------------------
    ref_cap = {"z_msa": [], "s_glue": [], "s_pf": [], "z_pf": []}

    def pf_pre(_m, _args, kwargs):
        ref_cap["s_glue"].append(kwargs["s"].detach()[0].clone())
        ref_cap["z_msa"].append(kwargs["z"].detach()[0].clone())

    def pf_post(_m, _args, out):
        s, z = out
        ref_cap["s_pf"].append(s.detach()[0].clone())
        ref_cap["z_pf"].append(z.detach()[0].clone())

    ref_model.pairformer_stack.register_forward_pre_hook(pf_pre, with_kwargs=True)
    ref_model.pairformer_stack.register_forward_hook(pf_post)

    with torch.no_grad():
        s_input_ref, s_trunk_ref, z_trunk_ref = ref_model.run_trunk(batch, num_cycles=4)
    print(f"reference: {len(ref_cap['s_pf'])} cycles captured, "
          f"s_trunk std {float(s_trunk_ref.std()):.2f}")

    # ---- device per-cycle capture ----------------------------------------
    import ttnn

    from tt_bio.openfold3 import InputEmbedderGlue
    from tt_bio.openfold3_trunk import OF3Trunk
    from tt_bio.openfold3_weights import _sub
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)

    ai = run_input_atom_encoder(dev, ckc, sd, features, aux)
    s_input = torch.cat(
        [ai, features["restype"], features["profile"],
         features["deletion_mean"].unsqueeze(-1)], dim=-1)

    def ft(x):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)

    glue = InputEmbedderGlue(_sub(sd, "input_embedder"), ckc)
    tb_d = ft(features["token_bonds"].unsqueeze(0).unsqueeze(-1))
    s_init_d, z_init_d = glue(ft(s_input.unsqueeze(0)), ft(relpos.unsqueeze(0)), tb_d)

    trunk = OF3Trunk(sd, ckc, num_cycles=4)
    tmpl_d = {k: ft(v) for k, v in template_feat.items()}
    msa_d = ft(msa_feat.unsqueeze(0))
    s_input_d = ft(s_input.unsqueeze(0))

    def host(t, shape):
        return ttnn.to_torch(t).float().reshape(*shape)

    s = trunk._zeros_like(s_init_d)
    z = trunk._zeros_like(z_init_d)
    m = trunk.msa_embedder(msa_d, s_input_d)
    print(f"{'cycle':>5} {'z_msa':>9} {'s_glue':>9} {'s_pf':>9} {'z_pf':>9}")
    for cyc in range(4):
        z = trunk.glue.glue_z(z, z_init_d)
        z_tmpl = trunk.template(tmpl_d, z)
        z = ttnn.add(z, z_tmpl)
        ttnn.deallocate(z_tmpl)
        z = trunk.msa_module(m, z)[1]
        z_msa_h = host(z, (n_token, n_token, 128))
        s = trunk.glue.glue_s(s, s_init_d)
        s_glue_h = host(s, (n_token, 384))
        s, z = trunk.pairformer(s, z)
        s_pf_h = host(s, (n_token, 384))
        z_pf_h = host(z, (n_token, n_token, 128))
        print(f"{cyc:>5} {pcc(z_msa_h, ref_cap['z_msa'][cyc]):>9.5f} "
              f"{pcc(s_glue_h, ref_cap['s_glue'][cyc]):>9.5f} "
              f"{pcc(s_pf_h, ref_cap['s_pf'][cyc]):>9.5f} "
              f"{pcc(z_pf_h, ref_cap['z_pf'][cyc]):>9.5f}")
    print(f"final: s_trunk pcc={pcc(s_pf_h, s_trunk_ref[0].float()):.6f} "
          f"z_trunk pcc={pcc(z_pf_h, z_trunk_ref[0].float()):.6f}")


if __name__ == "__main__":
    main()
