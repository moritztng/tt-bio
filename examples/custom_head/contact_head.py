"""A contact head for Boltz-2 and the loss it is trained with.

Two heads for `tt-bio predict --head`:

    pair_features   returns the pair representation z, to build a training set
    ContactHead     P(Cb-Cb < 8 A) for every residue pair, from z, with trained weights

`contact_loss` is the custom loss. Nothing here imports anything Tenstorrent-specific: the
trunk runs wherever `tt-bio predict` runs it, and these run in plain PyTorch on the host.
"""

from pathlib import Path

import torch
from torch import nn

WEIGHTS = Path(__file__).with_name("contact_head.pt")


def pair_features(fold):
    return {"z": fold.z}


class ContactHead(nn.Module):
    def __init__(self, c_z=128, hidden=64, pretrained=True):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(c_z), nn.Linear(c_z, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        if pretrained:
            self.load_state_dict(torch.load(WEIGHTS))

    def logits(self, z):
        x = self.mlp(z.float()).squeeze(-1)
        return 0.5 * (x + x.T)          # a contact map is symmetric, so the head is too

    def forward(self, fold):
        return {"contact_probs": torch.sigmoid(self.logits(fold.z))}


def contact_loss(logits, contacts, min_sep=6, long_range=24, long_weight=4.0):
    """Class-balanced BCE over residue pairs at least `min_sep` apart in sequence.

    Long-range pairs (|i-j| >= `long_range`) count `long_weight` times: they are the contacts a
    structure is hardest to get right and the ones a contact map is usually judged on.
    """
    n = len(contacts)
    sep = (torch.arange(n)[:, None] - torch.arange(n)[None, :]).abs()
    w = (sep >= min_sep).float() * torch.where(sep >= long_range, long_weight, 1.0)
    pos = (contacts * w).sum() / w.sum()
    balance = torch.where(contacts > 0, 0.5 / pos, 0.5 / (1 - pos))
    bce = nn.functional.binary_cross_entropy_with_logits(logits, contacts, reduction="none")
    return (bce * w * balance).sum() / w.sum()
