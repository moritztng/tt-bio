"""Grade tt-bio's host ESM-2 path against upstream DiffDock's own fair-esm embeddings.

    TT_VISIBLE_DEVICES= PYTHONPATH=$PWD env/bin/python3.10 \
        scripts/diffdock_port/esm2_fair_grade.py --ids 7BJ6_TVK,7MMH_ZJY

Upstream reads `representations[33]` from fair-esm's `esm2_t33_650M_UR50D`; tt-bio reads the
last hidden state of HF's `facebook/esm2_t33_650M_UR50D`. Same weights, same post-norm layer,
two loaders -- which is exactly the claim `esm2_embedding_tt` makes and the card test cannot
make, because it grades the card against the host, not the host against upstream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

LM_CACHE = Path("/home/moritz/scratch/dfd/data/lm_cache")
SEL = Path("/home/moritz/scratch/dfd/data/posebusters_dl/sel")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="7BJ6_TVK,7MMH_ZJY")
    ap.add_argument("--out", default=".lmgrade/fair.json")
    args = ap.parse_args()

    from tt_bio.diffdock import esm, input as dfd_input

    rows = []
    for cid in args.ids.split(","):
        up = torch.load(LM_CACHE / f"{cid}.pt", map_location="cpu", weights_only=False)[0]
        chains = dfd_input.chain_sequences_for_embedding(SEL / cid / f"{cid}_protein.pdb")
        got = esm.receptor_embeddings(chains, on_device=False)
        for j, (u, g) in enumerate(zip(up, got)):
            u = u.float()
            assert u.shape == g.shape, (cid, j, u.shape, g.shape)
            pcc = torch.stack([torch.corrcoef(torch.stack([u[i], g[i]]))[0, 1]
                               for i in range(u.shape[0])])
            rel = (torch.linalg.norm(g - u) / torch.linalg.norm(u)).item()
            rows.append({"id": cid, "chain": j, "residues": int(u.shape[0]),
                         "worst_residue_pcc": float(pcc.min()),
                         "rel_frobenius": rel,
                         "max_abs": float((g - u).abs().max())})
            print(rows[-1], flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
