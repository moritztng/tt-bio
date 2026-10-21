"""Upstream float64 AE goldens for the device parity run (tt_bio/rfd4/autoencoder.py).

Runs in the upstream reference env. For each case writes `<out>/<case>.pt` holding the input batch,
the encoder noise draw and upstream's float64 outputs: encoder mean / log_scale / z_latent, the
decoder's raw outputs and `AutoEncoder.decode`'s. The decoder reads upstream's own z_latent, so the
decoder grade is separable from the encoder's; `roundtrip` in the device script chains the two.

    python -I perf/rfd4/vae/gen_goldens.py --out /home/moritz/rfd4-goldens/vae [--sizes 96,250,500]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_fold import CKPT, load_fold, synthetic_batch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="96,250,500")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real-in", default="", help="dir of capture_real.py batches to add as real cases")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    from omegaconf import OmegaConf
    from rfproteina.partial_autoencoder.autoencoder import AutoEncoder
    from rfproteina.partial_autoencoder.decoder import DecoderTransformer
    from rfproteina.partial_autoencoder.encoder import EncoderTransformer

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    nn_ae = OmegaConf.to_container(ck["hyper_parameters"]["cfg_ae"].nn_ae, resolve=True)
    F = load_fold()
    tab = F.tables()
    enc = EncoderTransformer(**nn_ae).double().eval()
    enc.load_state_dict(F._sub(sd, "encoder."), strict=True)
    dec = DecoderTransformer(**nn_ae).double().eval()
    dec.load_state_dict(F._sub(sd, "decoder."), strict=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [(f"synthetic_n{n}", synthetic_batch(n, a.seed + n, tab, F.C_SEQ), 1234 + n)
             for n in (int(s) for s in a.sizes.split(",") if s)]
    for i, p in enumerate(sorted(Path(a.real_in).glob("*.pt")) if a.real_in else []):
        b = torch.load(p, map_location="cpu", weights_only=False)["batch"]
        b = {k: v.double() if v.is_floating_point() else v for k, v in b.items()}
        cases.append((f"real_{p.stem}", b, 99 + i))
    for case, batch, seed in cases:
        with torch.no_grad():
            torch.manual_seed(seed)
            e = enc(batch)
            torch.manual_seed(seed)
            noise = torch.randn_like(e["log_scale"])
            db = dict(batch, z_latent=e["z_latent"], ca_coors_nm=batch["coords_nm"][:, :, 1, :])
            d = dec(db)
            fin = AutoEncoder.decode(type("A", (), {"decoder": dec})(), db)
        torch.save({"case": case, "batch": db, "noise": noise, "encoder": e, "decoder": d, "decode": fin,
                    "dtype": "float64", "upstream_sha": "2db07bb444d91695f8c10e6acb7648e9297f9221", "ckpt": CKPT},
                   out / f"{case}.pt")
        print(case, {k: tuple(v.shape) for k, v in d.items()}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
