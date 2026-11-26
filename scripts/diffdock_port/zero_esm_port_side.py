"""The port's answer on upstream's own graph, with the receptor embedding as upstream had it.

The other half of ``zero_esm_upstream.py``. That script dumps upstream's step-0 graph and its
``(tr, rot, tor)`` for each fill of the receptor's ESM-2 block; this one feeds the port the
identical tensors and prints both answers side by side. Same graph, same released weights, so
a gap here is the implementation and nothing else.

    TT_VISIBLE_DEVICES= PYTHONPATH=$PWD ~/.venvs/jfp/bin/python \\
        scripts/diffdock_port/zero_esm_port_side.py ~/scratch/dfd/parity-nan/up7A9E_0.npz \\
                                                    ~/scratch/dfd/parity-nan/up7A9E_real.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

CKPT = Path.home() / "scratch/dfd/ckpt/extracted/score_model"


def pcc(a, b):
    a, b = torch.as_tensor(a).flatten().double(), torch.as_tensor(b).flatten().double()
    if a.numel() < 2 or not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    from tt_bio.diffdock import checkpoints, config as dfd_config, score_model
    from tt_bio.diffdock import sampling as S

    ckpt, params = checkpoints.resolve("score", str(CKPT), None)
    cfg = dfd_config.load(params.parent, old_model=False, name="diffdock")
    model = score_model.load_released(
        torch.load(ckpt, map_location="cpu", weights_only=False), cfg).eval()

    bounds = S.SigmaBounds(tr_sigma_min=cfg.tr_sigma_min, tr_sigma_max=cfg.tr_sigma_max,
                           rot_sigma_min=cfg.rot_sigma_min, rot_sigma_max=cfg.rot_sigma_max,
                           tor_sigma_min=cfg.tor_sigma_min, tor_sigma_max=cfg.tor_sigma_max)
    out = {}
    for path in a.npz:
        d = np.load(path)
        t = (float(d["t_tr"]), float(d["t_rot"]), float(d["t_tor"]))
        tr_s, rot_s, tor_s = bounds.sigmas(*t)
        args = (torch.from_numpy(d["lig_x"]).long(), torch.from_numpy(d["lig_pos"]),
                torch.from_numpy(d["bond_index"]).long(), torch.from_numpy(d["bond_attr"]),
                torch.from_numpy(d["edge_mask"]), torch.from_numpy(d["rec_x"]),
                torch.from_numpy(d["rec_pos"]), torch.from_numpy(d["rec_edge_index"]).long(),
                t[0], torch.as_tensor(tr_s), torch.as_tensor(rot_s), torch.as_tensor(tor_s))
        with torch.no_grad():
            tr, rot, tor = model(*args)[:3]

        rec_esm = torch.from_numpy(d["rec_x"])[:, 1:]
        row = {
            "esm_std": float(rec_esm.std()),
            "port": {"tr_absmax": float(tr.abs().max()), "rot_absmax": float(rot.abs().max()),
                     "tor_absmax": float(tor.abs().max()),
                     "tr_nan": int(torch.isnan(tr).sum()), "rot_nan": int(torch.isnan(rot).sum())},
            "upstream": {"tr_absmax": float(np.abs(d["tr"]).max()),
                         "rot_absmax": float(np.abs(d["rot"]).max()),
                         "tor_absmax": float(np.abs(d["tor"]).max())},
            "pcc": {"tr": pcc(tr[0], d["tr"]), "rot": pcc(rot[0], d["rot"]),
                    "tor": pcc(tor[:len(d["tor"])], d["tor"])},
        }
        out[Path(path).name] = row
        print(Path(path).name, json.dumps(row), flush=True)

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
