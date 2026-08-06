"""P13/S1 device leg: OF3Trunk on the searched-MSA inputs, PCC vs the CPU reference.

Loads ~/p13_s1_trunk_msa.pt (written by scripts/of3_trunk_msa_golden.py), runs the device
OF3Trunk on the IDENTICAL tensors the reference "fixed" pass consumed (reference
s_init/z_init/s_input, tt-bio's exact msa_feat draw, the empty 4-slot template block from
the fixture -- verified equal to the pipeline's dummy-template output on 1UBQ), and reports
PCC(s_trunk), PCC(z_trunk) against the fp32 CPU reference. The draw-sensitivity PCCs from
the golden script are the noise floor for the verdict.

Run with the tt-bio device env:
  PYTHONPATH=$WT TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=<p150 descriptor> \
    TT_BIO_LEASE_HOLDER=worker:tt-bio-openfold3-p13-msa-templates-e2e \
    /home/ttuser/tt-bio-dev/env/bin/python3 scripts/of3_trunk_msa_device_pcc.py
"""
import os
import pickle
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
GOLD = os.path.expanduser("~/of3_ref_out.pkl")
DATA = os.path.expanduser("~/p13_s1_trunk_msa.pt")


def pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def main():
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_trunk import OF3Trunk

    d = torch.load(DATA, map_location="cpu", weights_only=False)
    te = pickle.load(open(GOLD, "rb"))["intermediates"]["template_embedder_real"]["feat"]
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    trunk = OF3Trunk(sd, ckc, num_cycles=4)

    def ft(x):
        return ttnn.from_torch(x.float().unsqueeze(0), layout=ttnn.TILE_LAYOUT,
                               device=dev, dtype=ttnn.bfloat16)

    s_init_d = ft(d["s_init"])
    z_init_d = ft(d["z_init"])
    s_input_d = ft(d["s_input_ref"])
    msa_d = ft(d["msa_feat_tt"])
    tmpl_d = {k: ttnn.from_torch(v.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                 dtype=ttnn.bfloat16) for k, v in te.items()}

    s_dev_t, z_dev_t = trunk(s_init_d, z_init_d, tmpl_d, msa_d, s_input_d)
    s_dev = ttnn.to_torch(s_dev_t).float().reshape(d["s_init"].shape)
    z_dev = ttnn.to_torch(z_dev_t).float().reshape(d["z_init"].shape)

    z_ref, s_ref = d["z_trunk_fixed"], d["s_trunk_fixed"]
    print(f"RESULT device trunk searched-MSA: "
          f"pcc_z={pcc(z_dev, z_ref):.6f} pcc_s={pcc(s_dev, s_ref):.6f}")
    print(f"RESULT device vs upstream-draw pass: "
          f"pcc_z={pcc(z_dev, d['z_trunk_upstream']):.6f} "
          f"pcc_s={pcc(s_dev, d['s_trunk_upstream']):.6f}")
    print(f"RESULT std: z_dev={float(z_dev.std()):.3f} z_ref={float(z_ref.std()):.3f} "
          f"s_dev={float(s_dev.std()):.3f} s_ref={float(s_ref.std()):.3f}")


if __name__ == "__main__":
    main()
