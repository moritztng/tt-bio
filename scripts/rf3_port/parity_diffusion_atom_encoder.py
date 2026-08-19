#!/usr/bin/env python3
"""Score RF3's diffusion atom encoder, ceiling measured in the same run."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
PREFIX = "shadow.diffusion_module.atom_attention_encoder."


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d else float("nan")


def rel_rms(a, b):
    return float((a - b).pow(2).mean().sqrt() / b.std())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--capture", default="/home/ttuser/rf3_ref_work/fi/rna_denc.pt")
    args = ap.parse_args()

    import ttnn
    from tt_bio._vendor.rf3.loss.loss import calc_chiral_grads_flat_impl
    from tt_bio.boltz2 import get_indexing_matrix
    from tt_bio.rf3.atom_encoder import window_mask
    from tt_bio.rf3.atom_encoder_host import (ATOM_KEYS, ATOM_WINDOW,
                                              atom_to_token_mean, pair_inputs,
                                              single_features)
    from tt_bio.rf3.diffusion_atom_encoder import DiffusionAtomEncoder
    from tt_bio.rf3.weights import load_reference
    from tt_bio.tenstorrent import get_device

    cap = torch.load(args.capture, weights_only=False)
    f, r_noisy, s_trunk, z_trunk = cap["in"][0], cap["in"][1], cap["in"][2], cap["in"][3]
    want_A, want_Q, want_C, want_P = cap["out"]

    net, _ = load_reference(args.ckpt, num_steps=2)
    ref = net.diffusion_module.atom_attention_encoder

    def run(bf16):
        ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if bf16
               else torch.autocast("cpu", enabled=False))
        ff = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in f.items()}
        with torch.no_grad(), ctx:
            return [t.float() for t in ref(ff, r_noisy, s_trunk, z_trunk)]

    hi, lo = run(False), run(True)

    L = want_C.shape[0]
    I = want_A.shape[-2]
    Lp = ((L + ATOM_WINDOW - 1) // ATOM_WINDOW) * ATOM_WINDOW
    K = Lp // ATOM_WINDOW

    s_in = torch.zeros(1, Lp, 393); s_in[0, :L] = single_features(f, L)
    p_raw, v_raw = pair_inputs(f, L)
    p_in = torch.zeros(1, Lp, Lp, 32); p_in[0, :L, :L, :5] = p_raw
    v_in = torch.zeros(1, Lp, Lp, 1); v_in[0, :L, :L] = v_raw
    a2t_mean = torch.zeros(1, I, Lp); a2t_mean[0, :, :L] = atom_to_token_mean(f, L, I)
    a2t = torch.zeros(1, Lp, I)
    idx = f["atom_to_token_map"].long()[:L]
    a2t[0, torch.arange(L), idx] = 1.0
    r_in = torch.zeros(1, Lp, 3); r_in[0, :L] = r_noisy.reshape(-1, 3)[:L]

    with torch.no_grad(), torch.autocast("cpu", enabled=False):
        ch = calc_chiral_grads_flat_impl(
            r_noisy.detach().clone().float(), f["chiral_centers"].long(),
            f["chiral_center_dihedral_angles"].float(), False).nan_to_num()
    ch_in = torch.zeros(1, Lp, 3); ch_in[0, :L] = ch.reshape(-1, 3)[:L]

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    def tt(x):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)

    w = {k[len(PREFIX):]: v.float()
         for k, v in torch.load(args.ckpt, map_location="cpu",
                                weights_only=False)["model"].items()
         if k.startswith(PREFIX)}

    from tt_bio.rf3.feature_init import mlff_constant

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.process_atom_level_embedding = torch.nn.Sequential(
                torch.nn.Linear(384, 192), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                torch.nn.Linear(192, 96), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                torch.nn.Linear(96, 48), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                torch.nn.Linear(48, 16))
            self.conformers_to_atom_single_embedding = torch.nn.Sequential(
                torch.nn.Linear(128, 128, bias=False), torch.nn.LayerNorm(128))
        def forward(self, x):
            y = self.process_atom_level_embedding(x)
            y = y.permute(1, 0, 2).reshape(y.shape[1], -1)
            return self.conformers_to_atom_single_embedding(y)

    m = _M().eval()
    m.load_state_dict({k[len("process_atom_level_embedding."):]: v
                       for k, v in w.items()
                       if k.startswith("process_atom_level_embedding.")}, strict=True)

    enc = DiffusionAtomEncoder(w, cfg, mlff_constant(m))
    a_i, q_l, c_l, p_ll = enc(
        tt(s_in), tt(p_in), tt(v_in),
        tt(get_indexing_matrix(K, ATOM_WINDOW, ATOM_KEYS, torch.device("cpu"))),
        tt(a2t_mean), tt(window_mask(L, Lp)), Lp,
        tt(s_trunk.reshape(1, I, -1)), tt(z_trunk.reshape(1, I, I, -1)),
        tt(r_in), tt(ch_in), tt(a2t), tt(a2t.transpose(1, 2).contiguous()))

    def back(x):
        return torch.Tensor(ttnn.to_torch(x)).float()

    rows = []
    for name, got, want, h, l in (
        ("C_L", back(c_l)[0, :L], want_C, hi[2], lo[2]),
        ("P_LL", back(p_ll)[0, :L, :L, :16], want_P, hi[3], lo[3]),
        ("Q_L", back(q_l)[0, :L], want_Q.reshape(-1, want_Q.shape[-1]), hi[1], lo[1]),
        ("A_I", back(a_i)[0], want_A.reshape(-1, want_A.shape[-1]), hi[0], lo[0]),
    ):
        got = got.reshape(want.shape)
        ceil = rel_rms(l.reshape(want.shape), h.reshape(want.shape))
        e = rel_rms(got, want)
        rows.append({"tensor": name, "pcc": round(pcc(got, want), 7),
                     "rel_rms": round(e, 6), "ceiling": round(ceil, 6),
                     "x_ceiling": round(e / ceil, 2) if ceil else None})
    print(json.dumps({"L": L, "I": I, "scores": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
