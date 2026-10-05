"""Train ContactHead with contact_loss on the pair representations a fold exported.

    python train_head.py RUNS TRUTH

RUNS is a `tt-bio predict --head contact_head.py:pair_features` output directory, TRUTH holds
<id>.npy contact maps from fetch.py. Writes contact_head.pt next to contact_head.py.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from contact_head import WEIGHTS, ContactHead, contact_loss


def load(runs, truth):
    data = []
    for f in sorted(Path(runs).glob("*/structures/*_pair_features.npz")):
        rec = f.name.removesuffix("_pair_features.npz")
        z = torch.from_numpy(np.load(f)["z"])
        y = torch.from_numpy(np.load(Path(truth) / f"{rec}.npy"))
        assert z.shape[:2] == y.shape, f"{rec}: z {tuple(z.shape)} vs truth {tuple(y.shape)}"
        data.append((rec, z, y))
    return data


def main(runs, truth, epochs=300, lr=1e-3):
    torch.manual_seed(0)
    data = load(runs, truth)
    print("training on", ", ".join(f"{r} ({len(y)} aa)" for r, _, y in data))
    head = ContactHead(pretrained=False)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for epoch in range(epochs + 1):
        loss = sum(contact_loss(head.logits(z), y) for _, z, y in data) / len(data)
        if epoch % 50 == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}")
        if epoch < epochs:
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save(head.state_dict(), WEIGHTS)
    print(f"wrote {WEIGHTS}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
