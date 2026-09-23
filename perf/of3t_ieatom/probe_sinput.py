#!/usr/bin/env python3
"""of3t-ieatom: s_input from the device InputAtomEncoder and from the host leg, same batch.

    probe_sinput.py --batch B.pt --out S.pt

The training step now builds s_input with `openfold3.InputAtomEncoder`; inference keeps
`run_input_atom_encoder`. Both are dumped [n_token, 449] with the token mask, so a float64
upstream s_input on the same batch file can say how far each is from exact.
"""
import sys
from pathlib import Path

import torch


def main() -> int:
    argv = sys.argv[1:]
    batch, out = Path(argv[argv.index("--batch") + 1]), Path(argv[argv.index("--out") + 1])
    import ttnn
    from tt_bio.openfold3_fold import build_dm_device_aux
    from tt_bio.openfold3_host_prep import (derive_block_aux, ref_atom_device_inputs,
                                            ref_atom_embed, run_input_atom_encoder)
    from tt_bio.openfold3_weights import _sub
    from tt_bio.train.openfold3 import OpenFold3Dataset, OpenFold3Forward

    fwd = OpenFold3Forward(Path.home() / "of3-weights/of3-p2-155k.pt", seed=20260922)
    m, dev = fwd.model, fwd.device
    f = OpenFold3Dataset(batch).batch([0])["features"]
    aux = derive_block_aux(f)
    n_atom, n_token = aux["n_atom"], aux["n_token"]
    tok = f["token_mask"].float()
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                                   dtype=ttnn.bfloat16)
    token_feats = torch.cat([f["restype"], f["profile"], f["deletion_mean"].unsqueeze(-1)], -1)
    host = torch.cat([run_input_atom_encoder(dev, m.ckc, m.sd, f, aux), token_feats], -1)
    cl0, plm0 = ref_atom_embed(
        _sub(m.sd, "diffusion_module.atom_attn_enc.ref_atom_feature_embedder"), f)
    a = build_dm_device_aux(
        dev, ft, cl0=cl0, plm0=plm0, atom_mask=aux["atom_mask"],
        atom_to_token_index=aux["atom_to_token_index"],
        npe_q_indices=aux["npe_q_indices"], npe_k_indices=aux["npe_k_indices"],
        zij_mask=aux["zij_mask"], key_block_idxs=aux["key_block_idxs"],
        invalid_mask=aux["invalid_mask"], mask_trunked=aux["mask_trunked"],
        atom_to_token_mean=aux["atom_to_token_mean"], token_mask=tok, n_atom=n_atom,
        n_token=n_token, nb=aux["nb"], NP=aux["NP"], n_tok_pad=n_token)
    ref_in = ref_atom_device_inputs(dev, f, aux["atom_mask"], aux["NP"],
                                    dtype=m.input_atom_enc._act_dtype)
    mean = ttnn.from_torch(aux["atom_to_token_mean"].unsqueeze(0).float(), layout=ttnn.TILE_LAYOUT,
                           device=dev, dtype=m.input_atom_enc._act_dtype)
    d = m.input_atom_enc(ref_in, a["amc_d"], a["kidx_tt"], a["valid_d"], a["mb_d"], a["pm_d"],
                         a["amc_na_d"], mean, ft(token_feats.unsqueeze(0)),
                         n_atom, aux["NP"], aux["nb"])
    device = ttnn.to_torch(d).float().reshape(n_token, -1)[:, :449]
    torch.save({"host": host, "device": device, "token_mask": tok, "batch": str(batch)}, out)
    real = tok > 0
    rel = lambda x, y: float((x[real] - y[real]).norm() / y[real].norm())  # noqa: E731
    print(f"device vs host s_input (real tokens): rel {rel(device, host):.4e}; ai part "
          f"{rel(device[:, :384], host[:, :384]):.4e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
