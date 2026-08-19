#!/usr/bin/env python3
"""Confirm the atom-encoder pair formula in host torch before porting it to ttnn.

Separates two questions that are otherwise debugged together: "is the formula
right" and "is the ttnn right". This rebuilds P_LL from the reference's own
description using checkpoint weights and scores it against the captured P_LL.

Pins the two weightless flags a key-diff cannot see:
  use_inv_dist_squared=True  -> 1/(1 + sum(D*D)), and `+=` rather than `P = P + ...`
  the V_LL multiply applies to EVERY term, including process_valid_mask's own output.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--capture", default="/home/ttuser/rf3_ref_work/fi/rna_enc.pt")
    ap.add_argument("--autocast", action="store_true",
                    help="recompute under the same cpu-bf16 autocast the capture ran in; "
                         "if the formula is exact this collapses the gap to bf16 noise")
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    w = {k[len(PREFIX):]: v.float() for k, v in sd.items() if k.startswith(PREFIX)}
    cap = torch.load(args.capture, weights_only=False)
    f = cap["in"][0]
    want_C = cap["out"][2]            # [L, c_atom]
    want_P = cap["out"][3]            # [L, L, c_atompair]

    def lin(x, key):
        return torch.nn.functional.linear(x, w[key])

    ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if args.autocast
           else torch.autocast("cpu", enabled=False))
    ref_pos, suid = f["ref_pos"], f["ref_space_uid"]
    D = ref_pos.unsqueeze(-2) - ref_pos.unsqueeze(-3)             # [L, L, 3]
    V = (suid.unsqueeze(-1) == suid.unsqueeze(-2)).unsqueeze(-1)  # [L, L, 1]

    C = want_C                        # take C_L as given; scored separately below
    with ctx:
     P = lin(D, "process_d.weight") * V
    # use_inv_dist_squared is TRUE for this checkpoint: squared, and `+=`
     P += lin(1.0 / (1.0 + (D * D).sum(-1, keepdim=True)), "process_inverse_dist.weight") * V
     P = P + lin(V.to(P.dtype), "process_valid_mask.weight") * V
     P = P + (lin(torch.relu(C), "process_single_l.1.weight").unsqueeze(-2)
              + lin(torch.relu(C), "process_single_m.1.weight").unsqueeze(-3))
     mlp = P
     for k in ("pair_mlp.1.weight", "pair_mlp.3.weight", "pair_mlp.5.weight"):
        mlp = lin(torch.relu(mlp), k)
     P = P + mlp

    # and C_L itself, from the 1d features + the MLFF constant
    from tt_bio.rf3.feature_init import mlff_constant
    names = ["ref_pos", "ref_charge", "ref_mask", "ref_element",
             "ref_atom_name_chars", "ref_pos_ground_truth", "has_atom_level_embedding"]
    L = ref_pos.shape[0]
    cols = []
    for n in names:
        t = f[n].float()
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        cols.append(t.reshape(L, -1))
    with ctx:
        C_mine = lin(torch.cat(cols, dim=-1), "process_input_features.weight")

    import types
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
    msd = {k[len("process_atom_level_embedding."):]: v
           for k, v in w.items() if k.startswith("process_atom_level_embedding.")}
    m.load_state_dict(msd, strict=True)
    # The MLFF constant must be produced the way the reference produces it: the
    # reference forward runs under autocast, so a constant precomputed in fp32 is a
    # slightly different vector -- small, but it lands on every atom.
    with ctx:
        C_mine = C_mine + mlff_constant(m)

    P, C_mine = P.float(), C_mine.float()
    rep = {
        "L": int(L),
        "autocast": bool(args.autocast),
        "P": {"pcc": round(pcc(P, want_P), 8),
              "maxabs": round(float((P - want_P).abs().max()), 6),
              "rel_rms": round(float((P - want_P).pow(2).mean().sqrt() / want_P.std()), 8)},
        "C": {"pcc": round(pcc(C_mine, want_C), 8),
              "maxabs": round(float((C_mine - want_C).abs().max()), 6),
              "rel_rms": round(float((C_mine - want_C).pow(2).mean().sqrt() / want_C.std()), 8)},
    }
    rep["P"]["bit_exact"] = bool(torch.equal(P, want_P))
    rep["C"]["bit_exact"] = bool(torch.equal(C_mine, want_C))
    rep["verdict"] = ("PASS" if rep["P"]["pcc"] > 0.9999 and rep["C"]["pcc"] > 0.9999
                      else "FORMULA MISMATCH")
    print(json.dumps(rep, indent=2))
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
