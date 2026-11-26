"""Grade the assembled port against upstream's score network on UPSTREAM'S OWN GRAPH.

Consumes the npz written by ``score_net_parity.py`` (reference env): node features --
receptor features already carry upstream's 1280-wide ESM-2 block, so this grade contains no
language model of ours -- positions, both static edge lists, the rotatable-bond mask, the
step's t values, and upstream's ``(tr, rot, tor)`` for pose 0.  Feeding the port the same
tensors isolates the network from the graph builder: a gap here is arithmetic or wiring, a
gap only in ``score_model_grade.py`` is graph construction.

    TT_VISIBLE_DEVICES= PYTHONPATH=$PWD /home/moritz/tt-bio/env/bin/python3.10 \
        scripts/diffdock_port/score_net_grade.py --npz /tmp/dfd_net_7A9E.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

CKPT = Path("/home/moritz/scratch/dfd/ckpt/extracted/score_model")


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if a.numel() < 2:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def rel(a, b):
    s = b.abs().max()
    return float((a - b).abs().max() / s) if s > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from tt_bio.diffdock import config as C
    from tt_bio.diffdock import sampling as S
    from tt_bio.diffdock.score_model import load_released

    z = np.load(a.npz, allow_pickle=True)
    cfg = C.load(str(CKPT), old_model=False)
    sd = torch.load(CKPT / "best_ema_inference_epoch_model.pt", map_location="cpu",
                    weights_only=False)
    model = load_released(sd, cfg)
    bounds = S.SigmaBounds(tr_sigma_min=cfg.tr_sigma_min, tr_sigma_max=cfg.tr_sigma_max,
                           rot_sigma_min=cfg.rot_sigma_min, rot_sigma_max=cfg.rot_sigma_max,
                           tor_sigma_min=cfg.tor_sigma_min, tor_sigma_max=cfg.tor_sigma_max)

    out = {"complex": str(z["complex"]), "npz": a.npz, "steps": {}}
    T = lambda k, d=torch.float32: torch.as_tensor(z[k]).to(d)
    for k in [int(s) for s in z["steps"]]:
        p = f"s{k}_"
        t_tr, t_rot, t_tor = (float(z[p + "t_tr"]), float(z[p + "t_rot"]),
                              float(z[p + "t_tor"]))
        tr_s, rot_s, tor_s = bounds.sigmas(t_tr, t_rot, t_tor)
        with torch.no_grad():
            tr, rot, tor = model(
                T(p + "lig_x"), T(p + "lig_pos"), T(p + "bond_index", torch.long),
                T(p + "bond_attr"), T(p + "edge_mask", torch.bool), T(p + "rec_x"),
                T(p + "rec_pos"), T(p + "rec_edge_index", torch.long),
                t_tr, torch.as_tensor([tr_s]), torch.as_tensor([rot_s]),
                torch.as_tensor([tor_s]))
        got = {"tr": tr.reshape(-1), "rot": rot.reshape(-1), "tor": tor.reshape(-1)}
        want = {n: torch.as_tensor(z[p + n]).double().reshape(-1)
                for n in ("tr", "rot", "tor")}
        row = {n: {"pcc": pcc(got[n], want[n]), "max_rel_err": rel(got[n].double(), want[n])}
               for n in ("tr", "rot", "tor")}
        out["steps"][k] = row
        print(f"step {k:2d} tr_sigma {tr_s:7.3f}  " + "  ".join(
            f"{n} pcc {row[n]['pcc']:.6f} rel {row[n]['max_rel_err']:.3e}"
            for n in ("tr", "rot", "tor")), flush=True)

    print(json.dumps(out))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
