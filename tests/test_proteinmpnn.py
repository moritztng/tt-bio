"""Parity test for tt_bio.proteinmpnn against the official ProteinMPNN checkpoint.

Feeds the captured golden tensors (X, S, mask, chain_M, residue_idx,
chain_encoding_all, randn) into our clean reimplementation and asserts:
  * per-step log-prob PCC >= 0.999 vs the reference forward, and
  * greedy (argmax, teacher-forced) sequence recovery matches the reference to
    3 decimals.

Skips when the checkpoint or golden fixtures are absent (set via env or defaults
to the qb1 scratch paths) so this is a no-op on machines without the fixtures.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

torch.manual_seed(0)

CKPT = os.environ.get(
    "PROTEINMPNN_CKPT",
    str(Path.home() / "scratch/ProteinMPNN/vanilla_model_weights/v_48_020.pt"),
)
GOLDEN_DIR = Path(os.environ.get(
    "PROTEINMPNN_GOLDEN_DIR", str(Path.home() / "scratch/mpnn_golden")))


def _pcc(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _golden_cases():
    if not CKPT or not Path(CKPT).exists() or not GOLDEN_DIR.exists():
        return []
    return sorted(GOLDEN_DIR.glob("*.npz"))


@pytest.mark.parametrize("case", _golden_cases(), ids=lambda p: p.stem)
def test_forward_parity_vs_reference(case):
    g = np.load(case)
    from tt_bio.proteinmpnn import load_checkpoint

    model, k = load_checkpoint(CKPT, device="cpu", augment_eps=0.0)

    X = torch.from_numpy(g["X"]).float()
    S = torch.from_numpy(g["S"]).long()
    mask = torch.from_numpy(g["mask"]).float()
    chain_M = torch.from_numpy(g["chain_M"]).float()
    chain_M_pos = torch.from_numpy(g["chain_M_pos"]).float()
    residue_idx = torch.from_numpy(g["residue_idx"]).long()
    chain_enc = torch.from_numpy(g["chain_encoding_all"]).long()
    randn = torch.from_numpy(g["randn"]).float()

    with torch.no_grad():
        log_probs = model(X, S, mask, chain_M * chain_M_pos, residue_idx, chain_enc, randn)

    pcc = _pcc(log_probs.numpy(), g["log_probs"])
    pred = torch.argmax(log_probs, -1)
    m = (mask * chain_M * chain_M_pos)
    rec = (((pred == S).float() * m).sum() / m.sum()).item()
    ref_rec = float(g["greedy_recovery"])
    assert pcc >= 0.999, f"{case.stem}: log-prob PCC {pcc:.6f} < 0.999"
    assert abs(rec - ref_rec) < 5e-4, f"{case.stem}: recovery {rec:.4f} != ref {ref_rec:.4f}"


def test_param_count():
    from tt_bio.proteinmpnn import load_checkpoint
    if not CKPT or not Path(CKPT).exists():
        pytest.skip("checkpoint absent")
    model, _ = load_checkpoint(CKPT, device="cpu")
    n = sum(p.numel() for p in model.parameters())
    assert n == 1_660_485, f"param count {n} != published 1.66M"

def test_design_backbone_sanity():
    """End-to-end design: produces a full-length sequence with sane recovery."""
    import os
    from pathlib import Path
    from tt_bio.proteinmpnn import load_checkpoint, design_backbone
    ckpt = os.environ.get(
        "PROTEINMPNN_CKPT",
        str(Path.home() / "scratch/ProteinMPNN/vanilla_model_weights/v_48_020.pt"))
    pdb = Path.home() / "scratch/ProteinMPNN/inputs/PDB_monomers/pdbs/6MRR.pdb"
    if not Path(ckpt).exists() or not pdb.exists():
        pytest.skip("checkpoint/backbone fixtures absent")
    model, _ = load_checkpoint(ckpt, device="cpu")
    out = design_backbone(model, str(pdb), num_sequences=2, temperature=0.1, seed=0)
    assert len(out) == 2
    seq, rec = out[0]
    assert len(seq) == 68
    assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWYX")
    assert rec is not None and rec > 0.3, f"recovery {rec} implausibly low"
