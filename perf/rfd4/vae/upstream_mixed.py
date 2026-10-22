"""The faithful bf16-mixed floor: upstream's own AE (fp32 weights) run under bf16 autocast, graded with
device_parity.py's functions against the float64 goldens. This is what upstream's production precision
(`bf16-mixed`) deviates on the same inputs, so it is the yardstick a device grade is read against.

PairBiasAttention takes an einsum fallback when CUDA is absent; production runs
`F.scaled_dot_product_attention` (bf16 q/k/v, fp32 scores), so the CUDA check is forced true for that
module. Run in the upstream env (rfd4-ref-env venv):

    python -I perf/rfd4/vae/upstream_mixed.py --goldens <dir> --out <json>
"""

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check_fold import CKPT, load_fold  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import importlib.util
    from omegaconf import OmegaConf
    import rfproteina.nn.modules.pair_bias_attn as pba
    from rfproteina.partial_autoencoder.autoencoder import AutoEncoder
    from rfproteina.partial_autoencoder.decoder import DecoderTransformer
    from rfproteina.partial_autoencoder.encoder import EncoderTransformer

    pba.torch = type("T", (), {"__getattr__": lambda s, n: getattr(torch, n)})()
    pba.torch.cuda = type("C", (), {"is_available": staticmethod(lambda: True)})()
    spec = importlib.util.spec_from_file_location("device_parity_grade", HERE / "device_parity.py")
    DP = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DP)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    nn_ae = OmegaConf.to_container(ck["hyper_parameters"]["cfg_ae"].nn_ae, resolve=True)
    F = load_fold()
    tab = F.tables()
    enc = EncoderTransformer(**nn_ae).float().eval()
    enc.load_state_dict(F._sub(sd, "encoder."), strict=True)
    dec = DecoderTransformer(**nn_ae).float().eval()
    dec.load_state_dict(F._sub(sd, "decoder."), strict=True)
    f32 = lambda b: {k: v.float() if v.is_floating_point() else v for k, v in b.items()}  # noqa: E731
    f64 = lambda d: {k: v.double() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in d.items()}  # noqa: E731

    rec = {"bar": DP.BAR, "dtype": "fp32 weights, torch.autocast(cpu, bfloat16), SDPA attention", "cases": {}}
    for path in sorted(Path(a.goldens).glob("*.pt")):
        g = torch.load(path, map_location="cpu", weights_only=False)
        batch = g["batch"]
        legal = tab["allowed"][batch["token_type"].long()] & g["decoder"]["emit_mask"][..., None]
        em = g["encoder"]["emit_mask"]
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            e = f64(enc(f32(batch)))
            d = f64(dec(f32(batch)))
            fin = f64(AutoEncoder.decode(type("A", (), {"decoder": dec})(), f32(batch)))
        e["z_latent"] = (e["mean"] + g["noise"].double() * torch.exp(e["log_scale"])) * em[..., None]
        c = {"n_tokens": int(batch["padding_mask"].shape[1]),
             "encoder": {k: DP.rel(e[k], g["encoder"][k], em) for k in ("mean", "log_scale", "z_latent")}}
        c["decoder"] = DP.grade_decoder(d, g["decoder"], g["decode"], fin, legal)
        r_ref, r_up = DP.recon(g["decode"], batch), DP.recon(fin, batch)
        if r_ref is not None:
            c["recon"] = {"upstream": r_ref, "mixed": r_up, "d_rmsd_A": r_up["all_atom_rmsd_A"] - r_ref["all_atom_rmsd_A"],
                          "d_seq_recovery": r_up["seq_recovery"] - r_ref["seq_recovery"]}
        # Common-mode check: the shift of every class pair's logit gap, production minus float64, over the
        # emitted tokens of the dominant token type (a coin-flip population flips on its sign).
        em0, tt0 = g["decoder"]["emit_mask"][0], batch["token_type"][0]
        t = int(tt0[em0].mode().values)
        m = em0 & (tt0 == t)
        cls = tab["allowed"][t].nonzero().flatten()
        dl = (d["seq_logits"][0].double() - g["decoder"]["seq_logits"][0].double())[m][:, cls]
        c["logit_shift"] = {"token_type": t, "classes": cls.tolist(), "mean": dl.mean(0).tolist(), "std": dl.std(0).tolist()}
        rec["cases"][g["case"]] = c
        r = c["decoder"]["residue"]
        print(g["case"], "enc", max(v["max_rel_err"] for v in c["encoder"].values()), "lg",
              c["decoder"]["seq_logits"]["max_rel_err"], "argmax", r["argmax_agree"], "tv", r["tv_mean"],
              "crd", c["decoder"]["coords"]["rmsd_A"], flush=True)
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
