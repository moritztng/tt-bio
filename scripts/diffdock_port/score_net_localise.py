"""Name the FIRST layer where the port diverges from upstream's score network.

Consumes the npz from ``score_net_internals.py`` (upstream's per-layer node features for
pose 0) and replays the port on the same pose-0 inputs with the same hooks, then reports
every layer in forward order with its PCC and max relative error.  The first row that falls
off 1e-6 is the layer to read; everything after it is downstream contamination.

Upstream runs a batch of 10 poses, so each recorded tensor is sliced to pose 0:
ligand nodes are the first ``n_lig`` rows, receptor nodes inside the fused conv stack start
at ``10 * n_lig``, and the per-graph head outputs are row 0.

    TT_VISIBLE_DEVICES= PYTHONPATH=$PWD /home/moritz/tt-bio/env/bin/python3.10 \
        scripts/diffdock_port/score_net_localise.py --npz /tmp/dfd_int_7A9E.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

CKPT = Path("/home/moritz/scratch/dfd/ckpt/extracted/score_model")
N_POSES = 10


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if a.numel() < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def rel(a, b):
    s = b.abs().max()
    return float((a - b).abs().max() / s) if s > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from tt_bio.diffdock import config as C
    from tt_bio.diffdock import sampling as S
    from tt_bio.diffdock.score_model import load_released

    z = np.load(a.npz, allow_pickle=True)
    p = f"s{a.step}_"
    n_lig, n_rec = int(z[p + "n_lig"]), int(z[p + "n_rec"])

    cfg = C.load(str(CKPT), old_model=False)
    sd = torch.load(CKPT / "best_ema_inference_epoch_model.pt", map_location="cpu",
                    weights_only=False)
    model = load_released(sd, cfg)
    bounds = S.SigmaBounds(tr_sigma_min=cfg.tr_sigma_min, tr_sigma_max=cfg.tr_sigma_max,
                           rot_sigma_min=cfg.rot_sigma_min, rot_sigma_max=cfg.rot_sigma_max,
                           tor_sigma_min=cfg.tor_sigma_min, tor_sigma_max=cfg.tor_sigma_max)

    got = {}

    def hook(name):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            got[name] = o.detach()
        return f

    for i, m in enumerate(model.rec_emb_layers):
        m.register_forward_hook(hook(f"rec_emb_{i}"))
    for i, m in enumerate(model.lig_emb_layers):
        m.register_forward_hook(hook(f"lig_emb_{i}"))
    for i, m in enumerate(model.conv_layers):
        m.register_forward_hook(hook(f"conv_{i}"))
    model.rec_node_embedding.register_forward_hook(hook("rec_node_emb"))
    model.rec_edge_embedding.register_forward_hook(hook("rec_edge_emb"))
    model.lig_node_embedding.register_forward_hook(hook("lig_node_emb"))
    model.lig_edge_embedding.register_forward_hook(hook("lig_edge_emb"))
    model.cross_edge_embedding.register_forward_hook(hook("cross_edge_emb"))
    model.rec_sigma_embedding.register_forward_hook(hook("rec_sigma_emb"))
    model.center_head.center_edge_embedding.register_forward_hook(hook("center_edge_emb"))
    model.center_head.final_conv.register_forward_hook(hook("final_conv"))
    model.center_head.tr_final_layer.register_forward_hook(hook("tr_final"))
    model.center_head.rot_final_layer.register_forward_hook(hook("rot_final"))
    model.torsion_head.final_edge_embedding.register_forward_hook(hook("final_edge_emb"))
    model.torsion_head.tor_bond_conv.register_forward_hook(hook("tor_bond_conv"))
    model.torsion_head.tor_final_layer.register_forward_hook(hook("tor_final"))

    T = lambda k, d=torch.float32: torch.as_tensor(z[p + k]).to(d)
    t_tr, t_rot, t_tor = (float(z[p + "t_tr"]), float(z[p + "t_rot"]), float(z[p + "t_tor"]))
    tr_s, rot_s, tor_s = bounds.sigmas(t_tr, t_rot, t_tor)
    with torch.no_grad():
        tr, rot, tor = model(
            T("lig_x"), T("lig_pos"), T("bond_index", torch.long), T("bond_attr"),
            T("edge_mask", torch.bool), T("rec_x"), T("rec_pos"),
            T("rec_edge_index", torch.long), t_tr, torch.as_tensor([tr_s]),
            torch.as_tensor([rot_s]), torch.as_tensor([tor_s]))

    n_bonds = int(T("edge_mask", torch.bool).sum())

    # how to cut pose 0 out of upstream's batched recording, per layer
    LIG = ("lig_node_emb", "lig_emb_0", "lig_emb_1", "lig_emb_2", "lig_emb_3",
           "lig_emb_4", "lig_emb_5")
    REC = ("rec_node_emb", "rec_emb_0", "rec_emb_1", "rec_emb_2", "rec_emb_3",
           "rec_emb_4", "rec_emb_5", "rec_sigma_emb")
    rows = []

    def cut(name, want):
        """Pose 0's rows of upstream's recording of *name*."""
        if name in LIG:
            return want[:n_lig]
        if name in REC:
            return want[:1] if name == "rec_sigma_emb" else want[:n_rec]
        if name.startswith("conv_"):
            # the fused stack: ligand block for all poses, then the receptor block
            return torch.cat([want[:n_lig], want[N_POSES * n_lig:N_POSES * n_lig + n_rec]])
        if name in ("final_conv", "tr_final", "rot_final"):
            return want[:1]
        if name in ("tor_bond_conv", "tor_final"):
            return want[:n_bonds]
        return want

    def cut_got(name, g):
        if name.startswith("conv_"):
            return torch.cat([g[:n_lig], g[n_lig:n_lig + n_rec]])
        return g

    order = (["lig_node_emb"] + [f"lig_emb_{i}" for i in range(len(model.lig_emb_layers))]
             + ["rec_node_emb", "rec_sigma_emb"]
             + [f"rec_emb_{i}" for i in range(len(model.rec_emb_layers))]
             + [f"conv_{i}" for i in range(len(model.conv_layers))]
             + ["final_conv", "tr_final", "rot_final", "tor_bond_conv", "tor_final"])

    print(f"{'layer':<16} {'shape(ours)':>16} {'shape(up)':>16} {'pcc':>12} {'rel':>11}")
    for name in order:
        if p + name not in z or name not in got:
            rows.append({"layer": name, "note": "not recorded on one side"})
            print(f"{name:<16} {'-':>16} {'-':>16} {'-':>12} {'-':>11}")
            continue
        want = cut(name, torch.as_tensor(z[p + name]).float())
        ours = cut_got(name, got[name].float())
        if ours.shape != want.shape:
            rows.append({"layer": name, "ours": list(ours.shape), "up": list(want.shape),
                         "note": "SHAPE MISMATCH"})
            print(f"{name:<16} {str(tuple(ours.shape)):>16} {str(tuple(want.shape)):>16} "
                  f"{'SHAPE':>12} {'MISMATCH':>11}")
            continue
        r = {"layer": name, "pcc": pcc(ours, want), "max_rel_err": rel(ours, want),
             "shape": list(ours.shape)}
        rows.append(r)
        print(f"{name:<16} {str(tuple(ours.shape)):>16} {str(tuple(want.shape)):>16} "
              f"{r['pcc']:12.9f} {r['max_rel_err']:11.3e}")

    out = {"complex": str(z["complex"]), "step": a.step, "n_lig": n_lig, "n_rec": n_rec,
           "n_bonds": n_bonds, "layers": rows,
           "outputs": {n: {"pcc": pcc(g, w), "max_rel_err": rel(g.double(), w)}
                       for n, g, w in (
                           ("tr", tr.reshape(-1), torch.as_tensor(z[p + "tr"]).reshape(-1)),
                           ("rot", rot.reshape(-1), torch.as_tensor(z[p + "rot"]).reshape(-1)),
                           ("tor", tor.reshape(-1), torch.as_tensor(z[p + "tor"]).reshape(-1)))}}
    print(json.dumps(out["outputs"], indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
