"""Which rounding explains the device's decoder error? The float64 fold with bf16 weights, plus the
activations entering every matmul rounded to `--act` (bf16: 8-bit mantissa, tf32: 11-bit), graded
against the float64 goldens with device_parity.py's own decoder grade. CPU only.

    python -I perf/rfd4/vae/emulate.py --goldens <dir> --cases real_rna_pk synthetic_n500 --act bf16
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def rounder(kind):
    if kind == "none":
        return lambda t: t
    if kind == "bf16":
        return lambda t: t.to(torch.bfloat16).double()
    bits = {"tf32": 13}[kind]  # mantissa bits dropped from fp32's 23

    def tf32(t):
        i = t.float().view(torch.int32)
        i = (i + (1 << (bits - 1))) & ~((1 << bits) - 1)
        return i.view(torch.float32).double()
    return tf32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--act", default="bf16", choices=["none", "bf16", "tf32"])
    ap.add_argument("--only", default="", help="comma list of sites to round (default all): "
                    "in,h,qk,p,v,og,t,ug,head")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    AF = _load("ae_fold", ROOT / "tt_bio/rfd4/ae_fold.py")
    DP = _load("device_parity_grade", HERE.parent / "device_parity.py")
    CF = _load("check_fold", HERE.parent / "check_fold.py")
    r = rounder(a.act)
    sites = set(a.only.split(",")) if a.only else {"in", "h", "qk", "p", "v", "og", "t", "ug", "head"}
    R = {s: (r if s in sites else (lambda t: t)) for s in ("in", "h", "qk", "p", "v", "og", "t", "ug", "head")}
    bf = rounder("bf16")

    sd = torch.load(CF.CKPT, map_location="cpu", weights_only=False)["state_dict"]
    f = AF.fold(sd, "decoder")
    f["in_w"], f["head_w"] = bf(f["in_w"]), bf(f["head_w"])
    for L in f["layers"]:
        L.update({k: bf(L[k]) for k in ("qkv_w", "g_w", "o_w", "w1", "w2")})
    H, DH, D = AF.HEADS, AF.DH, AF.D

    def layer(L, x, bias, mask):
        b_, n, _ = x.shape
        h = R["h"](AF._ln(AF._ln(x) * L["ln1_w"] + L["ln1_b"], L["node_w"], L["node_b"]))
        q, k, v = (h @ L["qkv_w"].T + L["qkv_b"]).chunk(3, -1)
        q, k = R["qk"](AF._ln(q, L["q_ln_w"], L["q_ln_b"])), R["qk"](AF._ln(k, L["k_ln_w"], L["k_ln_b"]))
        q, k, v = (t.reshape(b_, n, H, DH).transpose(1, 2) for t in (q, k, R["v"](v)))
        p = R["p"](torch.softmax(q @ k.transpose(-1, -2) * DH ** -0.5 + bias, -1))
        o = (p @ v).transpose(1, 2).reshape(b_, n, D)
        og = R["og"](torch.sigmoid(h @ L["g_w"].T + L["g_b"]) * o)
        x = (x + og @ L["o_w"].T + L["o_b"]) * mask
        t = R["t"](AF._ln(x) * L["ln2_w"] + L["ln2_b"])
        u, gate = (t @ L["w1"].T).chunk(2, -1)
        return (x + R["ug"](F.silu(gate) * u) @ L["w2"].T) * mask

    tab = AF.tables()
    for case in a.cases:
        g = torch.load(Path(a.goldens) / f"{case}.pt", map_location="cpu", weights_only=False)
        batch = g["batch"]
        attend, emit = AF.ae_masks(batch)
        feats = AF.decoder_features(batch, f["emb"], tab["mask_token"])
        mask = attend[..., None].double()
        x = (R["in"](feats) @ f["in_w"].T) * mask
        bias = AF.key_bias(attend, torch.float64)
        for L in f["layers"]:
            x = layer(L, x, bias, mask)
        raw = R["head"](AF._ln(x)) @ f["head_w"].T + f["head_b"]
        d = AF.decode_heads(raw, batch, emit, tab)
        fin = AF.finish_decode(d, batch, tab)
        legal = tab["allowed"][batch["token_type"].long()] & g["decoder"]["emit_mask"][..., None]
        c = DP.grade_decoder(d, g["decoder"], g["decode"], fin, legal)
        print(case, a.act, a.only or "all", f"lg={c['seq_logits']['max_rel_err']:.4f} "
              f"pres={c['presence']['max_rel_err']:.4f} crd={c['coords']['rmsd_A']:.4f} "
              f"am={c['residue']['argmax_agree']:.4f} tv={c['residue']['tv_mean']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
