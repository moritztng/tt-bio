"""Float64 check of tt_bio/rfd4/ae_fold.py against upstream's EncoderTransformer / DecoderTransformer.

Runs in the upstream reference env (`/home/moritz/rfd4-ref-env/venv`, pinned sha). Both sides load the
released AE checkpoint in float64 and read one synthetic batch that exercises every mask the AE has:
protein tokens, atomized ligand tokens (no local latent), in-context conditioning tokens, padding,
and a partly conditioned C_SEQ. Writes a JSON record of max abs / rel error per output.

    python -I perf/rfd4/vae/check_fold.py --out <json> [--n 96] [--seed 0]
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
CKPT = ("/home/moritz/.cache/huggingface/hub/models--nvidia--RFD4-Proteina/snapshots/"
        "48377e45eb25a4fce8d2aff56c1a659b726d2106/ae_checkpoints/rfd4_proteina_ae-ema.ckpt")


def load_fold():
    spec = importlib.util.spec_from_file_location("ae_fold", ROOT / "tt_bio/rfd4/ae_fold.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def synthetic_batch(n: int, seed: int, tab: dict, C_SEQ: str) -> dict:
    g = torch.Generator().manual_seed(seed)
    b = 2
    n_pad = [0, n // 8]
    n_lig = 6
    token_type = torch.zeros(b, n, dtype=torch.long)
    token_type[:, -n_lig - max(n_pad):-max(n_pad)] = 3  # atomized ligand block
    res = torch.randint(0, 20, (b, n), generator=g)  # standard residues are classes 0..19 or so
    allowed = tab["allowed"]
    for i in range(b):
        for j in range(n):
            legal = allowed[token_type[i, j]].nonzero().flatten()
            res[i, j] = legal[torch.randint(0, len(legal), (1,), generator=g)]
    atom37 = tab["atom_mask"][res].clone()
    atom37[..., 1] = True
    # CA trace: a 0.38 nm random walk, atoms scattered 0.15 nm around their CA.
    step = torch.randn(b, n, 3, generator=g, dtype=torch.float64)
    ca = torch.cumsum(0.38 * step / step.norm(dim=-1, keepdim=True), 1)
    coords = ca[:, :, None, :] + 0.15 * torch.randn(b, n, 37, 3, generator=g, dtype=torch.float64)
    coords[:, :, 1, :] = ca
    padding = torch.ones(b, n, dtype=torch.bool)
    for i, p in enumerate(n_pad):
        if p:
            padding[i, -p:] = False
    in_ctx = torch.zeros(b, n, dtype=torch.bool)
    in_ctx[0, 3:7] = True
    supports = token_type != 3
    cseq = torch.full((b, n), tab["mask_token"], dtype=torch.long)
    cseq[:, 10:20] = res[:, 10:20]
    batch = {
        "padding_mask": padding,
        "is_in_context_conditioning": in_ctx,
        "supports_local_latent": supports,
        "token_type": token_type,
        "coords_nm": coords,
        "coords": coords * 10.0,
        "atom37_mask": atom37,
        "chain_iid": (torch.arange(n)[None].expand(b, n) >= n // 2).to(torch.int16),
        "incident_bond_type_counts": torch.randint(0, 3, (b, n, 37, 10), generator=g),
        "ground_truth_element": torch.randint(1, 30, (b, n, 37), generator=g),
        "charge": torch.randint(-2, 3, (b, n, 37), generator=g),
        C_SEQ: cseq,
    }
    return batch


def err(a: torch.Tensor, b: torch.Tensor, mask=None) -> dict:
    a, b = a.double(), b.double()
    if mask is not None:
        a, b = a[mask], b[mask]
    d = (a - b).abs()
    return {"max_abs": float(d.max()), "max_rel": float(d.max() / b.abs().max().clamp_min(1e-30))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)

    from rfproteina.partial_autoencoder.decoder import DecoderTransformer
    from rfproteina.partial_autoencoder.encoder import EncoderTransformer

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    nn_ae = OmegaConf.to_container(ck["hyper_parameters"]["cfg_ae"].nn_ae, resolve=True)
    F = load_fold()
    tab = F.tables()
    batch = synthetic_batch(a.n, a.seed, tab, F.C_SEQ)
    rec = {"n": a.n, "seed": a.seed, "dtype": "float64", "ckpt": CKPT}

    enc = EncoderTransformer(**nn_ae).double().eval()
    enc.load_state_dict(F._sub(sd, "encoder."), strict=True)
    dec = DecoderTransformer(**nn_ae).double().eval()
    dec.load_state_dict(F._sub(sd, "decoder."), strict=True)

    with torch.no_grad():
        torch.manual_seed(1234)
        up_e = enc(batch)
        torch.manual_seed(1234)
        noise = torch.randn_like(up_e["log_scale"])
        fe = F.fold(sd, "encoder")
        my_e = F.encode(fe, batch, noise)
        em = up_e["emit_mask"]
        rec["encoder"] = {k: err(my_e[k], up_e[k], em) for k in ("mean", "log_scale", "z_latent")}
        rec["encoder"]["emit_mask_equal"] = bool(torch.equal(my_e["emit_mask"], em))

        dbatch = dict(batch)
        dbatch["z_latent"] = up_e["z_latent"]
        dbatch["ca_coors_nm"] = batch["coords_nm"][:, :, 1, :]
        up_d = dec(dbatch)
        fd = F.fold(sd, "decoder")
        my_d = F.decode(fd, dbatch, tab)
        em = up_d["emit_mask"]
        allowed = tab["allowed"][batch["token_type"]] & em[..., None]
        rec["decoder"] = {
            "seq_logits": err(my_d["seq_logits"][allowed], up_d["seq_logits"][allowed]),
            "coors_nm": err(my_d["coors_nm"], up_d["coors_nm"], em),
            "atom_presence_logits": err(my_d["atom_presence_logits"], up_d["atom_presence_logits"], em),
            "aatype_equal": bool(torch.equal(my_d["aatype_max"], up_d["aatype_max"])),
            "disallowed_fill_equal": bool(torch.equal(my_d["seq_logits"][~allowed & em[..., None]],
                                                      up_d["seq_logits"][~allowed & em[..., None]])),
        }
        from rfproteina.partial_autoencoder.autoencoder import AutoEncoder
        up_f = AutoEncoder.decode(type("A", (), {"decoder": dec})(), dbatch)
        my_f = F.finish_decode(my_d, dbatch, tab)
        rec["decode"] = {
            "coors_nm": err(my_f["coors_nm"], up_f["coors_nm"]),
            "residue_type_equal": bool(torch.equal(my_f["residue_type"], up_f["residue_type"])),
            "atom_mask_equal": bool(torch.equal(my_f["atom_mask"], up_f["atom_mask"])),
        }
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
