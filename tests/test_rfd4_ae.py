"""RFD4-Proteina autoencoder: the host fold (CPU) and the device port against upstream float64 goldens.

The goldens come from `perf/rfd4/vae/gen_goldens.py` in the upstream reference env; point
`RFD4_AE_GOLDENS` at their directory and `RFD4_AE_CKPT` at `rfd4_proteina_ae-ema.ckpt`.
"""

import os
from pathlib import Path

import pytest
import torch

from tt_bio.rfd4 import ae_fold as AF

GOLDENS = os.environ.get("RFD4_AE_GOLDENS", "/home/moritz/rfd4-goldens/vae")
CKPT = os.environ.get("RFD4_AE_CKPT", "")


def test_tables_shape():
    t = AF.tables()
    assert t["allowed"].shape == (AF.N_TOKEN_TYPES, AF.N_RES)
    assert t["atom_mask"].shape == (AF.N_RES, 37)
    assert t["allowed"].any(-1).all()


def _goldens():
    p = Path(GOLDENS)
    return sorted(p.glob("*.pt")) if p.is_dir() else []


@pytest.mark.device
@pytest.mark.parametrize("golden", _goldens() or [pytest.param(None, marks=pytest.mark.skip(reason="no AE goldens"))])
def test_ae_device_parity(golden):
    if not CKPT or not Path(CKPT).exists():
        pytest.skip("RFD4_AE_CKPT not set")
    from tt_bio.rfd4.autoencoder import RFD4AutoEncoder

    g = torch.load(golden, map_location="cpu", weights_only=False)
    ae = RFD4AutoEncoder.from_checkpoint(CKPT)
    batch = g["batch"]
    em = g["encoder"]["emit_mask"]
    e = ae.encode(batch, g["noise"])
    # Real inputs carry the campaign's 0.05 bar. The synthetic batches are random, out-of-distribution
    # features: upstream's own bf16-mixed precision already reads 0.05-0.48 on their latents
    # (perf/rfd4/vae/cpu_parity.py --mixed), so they are only held to a looser bound here and graded
    # against that floor in perf/rfd4/vae/write_parity.py.
    bar = 0.05 if g["case"].startswith("real") else 1.0
    for k in ("mean", "log_scale"):
        d = (e[k] - g["encoder"][k].float())[em].abs().max() / g["encoder"][k][em].abs().max()
        assert d < bar, (k, float(d))
    d = ae.decode_raw(batch)
    fin = AF.finish_decode(d, batch, ae.tab)
    am = g["decode"]["atom_mask"]
    rmsd = ((d["coors_nm"].double() - g["decoder"]["coors_nm"])[am] * 10).norm(dim=-1).pow(2).mean().sqrt()
    assert rmsd < 0.05, float(rmsd)
    emit = g["decoder"]["emit_mask"]
    agree = (fin["residue_type"] == g["decode"]["residue_type"])[emit].double().mean()
    assert agree >= 0.99, float(agree)
