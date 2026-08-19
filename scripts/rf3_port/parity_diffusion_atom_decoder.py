#!/usr/bin/env python3
"""Score RF3's diffusion atom decoder, ceiling measured in the same run."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
PREFIX = "shadow.diffusion_module.atom_attention_decoder."


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
    ap.add_argument("--capture", default="/home/ttuser/rf3_ref_work/fi/rna_ddec.pt")
    args = ap.parse_args()

    import ttnn
    from tt_bio.boltz2 import get_indexing_matrix
    from tt_bio.rf3.atom_encoder import window_mask
    from tt_bio.rf3.atom_encoder_host import ATOM_KEYS, ATOM_WINDOW
    from tt_bio.rf3.diffusion_atom_decoder import DiffusionAtomDecoder
    from tt_bio.rf3.weights import load_reference
    from tt_bio.tenstorrent import get_device

    cap = torch.load(args.capture, weights_only=False)
    f, a_i, q_skip, c_skip, p_skip = cap["in"][:5]
    want = cap["out"]

    net, _ = load_reference(args.ckpt, num_steps=2)
    ref = net.diffusion_module.atom_attention_decoder

    def run(bf16):
        ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if bf16
               else torch.autocast("cpu", enabled=False))
        with torch.no_grad(), ctx:
            return ref(f, a_i, q_skip, c_skip, p_skip).float()

    ceil = rel_rms(run(True).reshape(want.shape), run(False).reshape(want.shape))

    L = c_skip.shape[0]
    I = a_i.shape[-2]
    Lp = ((L + ATOM_WINDOW - 1) // ATOM_WINDOW) * ATOM_WINDOW
    K = Lp // ATOM_WINDOW

    a2t = torch.zeros(1, Lp, I)
    a2t[0, torch.arange(L), f["atom_to_token_map"].long()[:L]] = 1.0
    qs = torch.zeros(1, Lp, c_skip.shape[-1]); qs[0, :L] = q_skip.reshape(-1, c_skip.shape[-1])[:L]
    cs = torch.zeros(1, Lp, c_skip.shape[-1]); cs[0, :L] = c_skip
    ps = torch.zeros(1, Lp, Lp, p_skip.shape[-1]); ps[0, :L, :L] = p_skip

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

    dec = DiffusionAtomDecoder(w, cfg)
    got = dec(tt(a_i.reshape(1, I, -1)), tt(qs), tt(cs), tt(ps), tt(a2t),
              tt(get_indexing_matrix(K, ATOM_WINDOW, ATOM_KEYS, torch.device("cpu"))),
              tt(window_mask(L, Lp)), Lp)
    got = torch.Tensor(ttnn.to_torch(got)).float()[0, :L].reshape(want.shape)
    e = rel_rms(got, want)
    print(json.dumps({"L": L, "I": I, "tensor": "R_update",
                      "shape": list(want.shape),
                      "pcc": round(pcc(got, want), 7), "rel_rms": round(e, 6),
                      "ceiling": round(ceil, 6),
                      "x_ceiling": round(e / ceil, 2) if ceil else None}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
