"""The same grade device_parity.py applies, run on the float64 torch fold instead of the device.

This is what separates a featurisation bug from a device-precision one: the fold and the device read
the SAME host features, so if a real captured batch grades clean here and dirty on the card, the
card is the cause, and if it grades dirty here the features are. No TT device, no ttnn import.

    python -I perf/rfd4/vae/cpu_parity.py --goldens <dir> --out <json>
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _trunk(AF, f, feats, attend):
    mask = attend[..., None].to(feats.dtype)
    x = (feats @ f["in_w"].T) * mask
    bias = AF.key_bias(attend, feats.dtype)
    for L in f["layers"]:
        x = AF.layer_forward(L, x, bias, mask)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bf16-weights", action="store_true",
                    help="round the matrices the device holds in bf16 (in_w, block linears, head_w) and keep "
                         "everything else float64: the floor a bf16-weight port cannot go below")
    ap.add_argument("--mixed", action="store_true",
                    help="bf16-mixed, upstream's production precision: --bf16-weights plus every matmul input "
                         "rounded to bf16 (fp32 accumulate, LayerNorm and softmax in full precision). A lenient "
                         "model of autocast, so a lower bound on what a GPU's bf16-mixed run deviates")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    AF = _load("ae_fold", ROOT / "tt_bio/rfd4/ae_fold.py")
    DP = _load("device_parity_grade", HERE.parent / "device_parity.py")
    CF = _load("check_fold", HERE.parent / "check_fold.py")

    sd = torch.load(CF.CKPT, map_location="cpu", weights_only=False)["state_dict"]
    folded = {p: AF.fold(sd, p) for p in ("encoder", "decoder")}
    if a.mixed:
        a.bf16_weights = True
        EM = _load("emulate", HERE.parent / "emulate.py")
        r = EM.rounder("bf16")
        AF.layer_forward = EM.mixed_layer(AF, r)
        _ln = AF._ln
        AF.trunk_forward = lambda f, feats, attend: _trunk(AF, f, r(feats), attend)  # noqa: E731
        AF.heads_forward = lambda f, x: r(_ln(x)) @ f["head_w"].T + f["head_b"]  # noqa: E731
    if a.bf16_weights:
        bf = lambda t: t.to(torch.bfloat16).double()  # noqa: E731
        mats = ("qkv_w", "g_w", "o_w", "w1", "w2")
        for f in folded.values():
            f["in_w"], f["head_w"] = bf(f["in_w"]), bf(f["head_w"])
            for L in f["layers"]:
                L.update({k: bf(L[k]) for k in mats})
    tab = AF.tables()
    rec = {"bar": DP.BAR, "dtype": "float64", "bf16_weights": a.bf16_weights, "mixed": a.mixed, "cases": {}}
    for path in sorted(Path(a.goldens).glob("*.pt")):
        g = torch.load(path, map_location="cpu", weights_only=False)
        batch = g["batch"]
        legal = tab["allowed"][batch["token_type"].long()] & g["decoder"]["emit_mask"][..., None]
        em = g["encoder"]["emit_mask"]
        e = AF.encode(folded["encoder"], batch, g["noise"])
        c = {"n_tokens": int(batch["padding_mask"].shape[1]),
             "encoder": {k: DP.rel(e[k], g["encoder"][k], em) for k in ("mean", "log_scale", "z_latent")}}
        d = AF.decode(folded["decoder"], batch, tab)
        fin = AF.finish_decode(d, batch, tab)
        c["decoder"] = DP.grade_decoder(d, g["decoder"], g["decode"], fin, legal)
        r_ref, r_tt = DP.recon(g["decode"], batch), DP.recon(fin, batch)
        if r_ref is not None:
            c["recon"] = {"upstream": r_ref, "fold": r_tt,
                          "d_rmsd_A": r_tt["all_atom_rmsd_A"] - r_ref["all_atom_rmsd_A"],
                          "d_seq_recovery": r_tt["seq_recovery"] - r_ref["seq_recovery"]}
        rec["cases"][g["case"]] = c
        print(g["case"], "max_rel", max(c["encoder"][k]["max_rel_err"] for k in c["encoder"]),
              "coords_A", c["decoder"]["coords"]["rmsd_A"], "argmax", c["decoder"]["residue"]["argmax_agree"],
              "recon", c.get("recon", {}).get("upstream", {}).get("all_atom_rmsd_A"), flush=True)
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
