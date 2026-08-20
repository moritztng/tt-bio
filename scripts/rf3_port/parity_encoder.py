#!/usr/bin/env python3
"""Score the ttnn atom attention encoder against the captured reference.

Reports every tensor the reference block returns, so a mismatch localises: C_L and
P_LL are already proven bit-exact in host torch, so a gap there is the ttnn's, and a
gap that appears only in Q_L is the atom transformer (windowing, mask, no_residual,
a_to_b_gate).

    TT_VISIBLE_DEVICES=2 python scripts/rf3_port/parity_encoder.py --ckpt ... \
        --capture /home/ttuser/rf3_ref_work/fi/rna_enc.pt
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
PREFIX = "shadow.feature_initializer.input_feature_embedder.atom_attention_encoder."


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d else float("nan")


def score(got, want, label):
    got = got.reshape(want.shape).float()
    diff = (got - want).abs()
    return {"tensor": label, "shape": list(want.shape),
            "pcc": round(pcc(got, want), 7),
            "maxabs": round(float(diff.max()), 6),
            "rel_rms": round(float(diff.pow(2).mean().sqrt() / want.std()), 6)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--capture", default="/home/ttuser/rf3_ref_work/fi/rna_enc.pt")
    args = ap.parse_args()

    import ttnn
    from tt_bio.boltz2 import get_indexing_matrix
    from tt_bio.rf3.atom_encoder import AtomAttentionEncoder, window_mask
    from tt_bio.rf3.atom_encoder_host import (ATOM_KEYS, ATOM_WINDOW, pad_pair,
                                              window_pair, window_pair_valid,
                                              atom_to_token_mean, pair_inputs,
                                              single_features)
    from tt_bio.tenstorrent import get_device

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    w = {k[len(PREFIX):]: v.float() for k, v in sd.items() if k.startswith(PREFIX)}
    cap = torch.load(args.capture, weights_only=False)
    f = cap["in"][0]
    want_A, want_Q, want_C, want_P = (cap["out"][0], cap["out"][1],
                                      cap["out"][2], cap["out"][3])

    L = want_C.shape[0]
    I = want_A.shape[-2]
    Lp = ((L + ATOM_WINDOW - 1) // ATOM_WINDOW) * ATOM_WINDOW
    K = Lp // ATOM_WINDOW

    # ---- host inputs, padded up to the window grid
    s_in = torch.zeros(1, Lp, 393); s_in[0, :L] = single_features(f, L)
    p_raw, v_raw = pair_inputs(f, L)
    p_in = torch.zeros(1, Lp, Lp, 32); p_in[0, :L, :L, :5] = p_raw
    v_in = torch.zeros(1, Lp, Lp, 1); v_in[0, :L, :L] = v_raw
    p_in = window_pair(p_in)
    v_in = window_pair(v_in)
    a2t = torch.zeros(1, I, Lp); a2t[0, :, :L] = atom_to_token_mean(f, L, I)

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    def tt(x):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16)

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

    enc = AtomAttentionEncoder(w, cfg, mlff_constant(m))
    a_i, q_l, c_l, p_ll = enc(
        tt(s_in), tt(p_in), tt(v_in),
        tt(get_indexing_matrix(K, ATOM_WINDOW, ATOM_KEYS, torch.device("cpu"))),
        tt(a2t), tt(window_mask(L, Lp)), Lp)

    def back(x):
        return torch.Tensor(ttnn.to_torch(x)).float()

    rows = [
        score(back(c_l)[0, :L], want_C, "C_L"),
        score(window_pair_valid(back(p_ll)[..., :16], L),
              window_pair_valid(window_pair(pad_pair(want_P, Lp)), L), "P_LL"),
        score(back(q_l)[0, :L], want_Q.reshape(-1, want_Q.shape[-1]), "Q_L"),
        score(back(a_i)[0], want_A.reshape(-1, want_A.shape[-1]), "A_I"),
    ]
    # Score against the CEILING, not an absolute pcc. `pcc > 0.99` called this PASS
    # while Q_L sat 23x above the pair track's bf16 error -- the same way a bare pcc
    # hid a 10x rel_rms gap earlier on this port. C_L and P_LL are proven bit-exact in
    # host torch, so their device rel_rms IS this fixture's bf16 floor; anything much
    # above it is the port, not the format.
    ceiling = max(rows[0]["rel_rms"], rows[1]["rel_rms"])
    for r in rows:
        r["x_ceiling"] = round(r["rel_rms"] / ceiling, 2)
    ok = all(r["x_ceiling"] <= 2.0 for r in rows)
    print(json.dumps({"L": L, "I": I, "Lpad": Lp, "windows": K,
                      "bf16_ceiling_rel_rms": ceiling,
                      "scores": rows,
                      "verdict": "PASS (at the bf16 ceiling)" if ok
                                 else "GAP (above the bf16 ceiling)"}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
