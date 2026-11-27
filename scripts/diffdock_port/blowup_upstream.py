"""Upstream's answer on the exact pose where the port's score head reaches 1e15.

The other half of ``blowup_dump.py``. That script records the score network's twelve input
tensors at the call whose output blows up; this one rebuilds the same complex in upstream's
own ``InferenceDataset``, checks the graph tensors are identical to the dumped ones, puts the
dumped pose in, and runs upstream's unmodified ``TensorProductScoreModel`` on it.

One input, two implementations. If upstream also returns ~1e15 the pose is pathological and
the port's rollout merely walked into it; if upstream returns O(10) the port's network is
wrong on a real input and the gap is localisable from here.

    cd ~/scratch/dfd/upstream && PYTHONPATH=. ~/scratch/dfd/venv-ref/bin/python \\
      /home/moritz/.coworker/wt/dfd-parity/scripts/diffdock_port/blowup_upstream.py \\
      --id 7ZCC_OGA --npz ~/scratch/dfd/parity-blowup/7ZCC_OGA/blowup.npz \\
      --out /tmp/dfd_blowup_7ZCC.json
"""
import argparse
import copy
import json
import sys
from argparse import Namespace
from pathlib import Path
from functools import partial

import numpy as np
import torch
import yaml

D = Path.home() / "scratch/dfd"
sys.path.insert(0, str(D / "scripts"))
import dfd_ref_lib as L  # noqa: E402
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "_dfd_dump_graph_golden", Path(__file__).resolve().parent / "dump_graph_golden.py")
_dg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_dg)
PB, lm_embeddings_for = _dg.PB, _dg.lm_embeddings_for


def smiles_of(cid):
    from rdkit import Chem
    m = next(Chem.SDMolSupplier(f"{PB}/{cid}/{cid}_ligand.sdf", sanitize=True, removeHs=True),
             None)
    return None if m is None else Chem.MolToSmiles(Chem.RemoveHs(m))


def census(t):
    t = torch.as_tensor(t)
    f = t[torch.isfinite(t)]
    return {"n": int(t.numel()), "nan": int(torch.isnan(t).sum()),
            "inf": int(torch.isinf(t).sum()),
            "absmax": float(f.abs().max()) if f.numel() else float("nan")}


def pcc(a, b):
    a = torch.as_tensor(np.asarray(a)).flatten().double()
    b = torch.as_tensor(np.asarray(b)).flatten().double()
    if a.numel() < 2 or not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--npz", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="allowed maxdiff on rec_x: the two sides ran ESM-2 on different "
                         "hosts, so the embedding carries float noise. Every other tensor "
                         "must match exactly.")
    ap.add_argument("--pb", default=None, help="PoseBusters root, if not dump_graph_golden's")
    ap.add_argument("--sdf-ligand", action="store_true",
                    help="build the ligand from the crystal SDF instead of its SMILES")
    a = ap.parse_args()
    global PB
    if a.pb:
        PB = Path(a.pb).expanduser()

    L.setup(a.seed)
    from torch_geometric.loader import DataLoader
    from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, set_time
    from utils.utils import crop_beyond
    from utils.inference_utils import InferenceDataset
    from utils.utils import get_model

    cid = a.id
    prot = PB / cid / f"{cid}_protein.pdb"
    lig = str(PB / cid / f"{cid}_ligand.sdf") if a.sdf_ligand else smiles_of(cid)
    run_dir = D / "runs/blowup" / cid
    run_dir.mkdir(parents=True, exist_ok=True)
    inf = yaml.safe_load((D / "upstream/default_inference_args.yaml").read_text())
    with open(D / "ckpt/extracted/score_model/model_parameters.yml") as f:
        sma = Namespace(**yaml.full_load(f))

    ds = InferenceDataset(
        out_dir=str(run_dir), complex_names=[cid], protein_files=[str(prot)],
        ligand_descriptions=[lig], protein_sequences=[None], lm_embeddings=True,
        receptor_radius=sma.receptor_radius, remove_hs=sma.remove_hs,
        c_alpha_max_neighbors=sma.c_alpha_max_neighbors, all_atoms=sma.all_atoms,
        atom_radius=sma.atom_radius, atom_max_neighbors=sma.atom_max_neighbors,
        precomputed_lm_embeddings=lm_embeddings_for(cid, str(prot)),
        knn_only_graph=(not sma.not_knn_only_graph) if hasattr(sma, "not_knn_only_graph")
        else False)

    device = torch.device("cpu")
    t_to_sigma = partial(t_to_sigma_compl, args=sma)
    model = get_model(sma, device, t_to_sigma=t_to_sigma, no_parallel=True,
                      old=inf.get("old_score_model", False))
    model.load_state_dict(torch.load(D / "ckpt/extracted/score_model" / inf["ckpt"],
                                     map_location="cpu", weights_only=False), strict=True)
    model.eval()

    graph = next(iter(DataLoader(dataset=ds, batch_size=1, shuffle=False)))
    assert bool(graph.success[0])

    results = {}
    for path in a.npz:
        d = np.load(path)
        g = copy.deepcopy(graph)
        t_tr, t_rot, t_tor = float(d["t_tr"]), float(d["t_rot"]), float(d["t_tor"])

        # The pose has to go in BEFORE the crop: upstream crops the receptor against the
        # ligand it is about to score, every step, at 3 * tr_sigma + model_args.crop_beyond
        # (utils/sampling.py:104-108, crop_beyond: 20 in the v1.1 model_parameters.yml). The
        # port does the same thing as a node mask (`graph.rec_keep`), so a dump taken mid
        # rollout carries the CROPPED receptor and comparing it against the full graph is a
        # mismatch in the check, not in the port.
        g["ligand"].pos = torch.from_numpy(d["lig_pos"]).float()
        tr_sigma, _, _ = t_to_sigma(t_tr, t_rot, t_tor)
        cropped_at = None
        if getattr(sma, "crop_beyond", None) is not None:
            cropped_at = float(tr_sigma) * 3 + sma.crop_beyond
            crop_beyond(g, cropped_at, getattr(sma, "all_atoms", False))
            # crop_beyond rewrites x/pos/side_chain_vecs and the rec-rec edges but not the
            # batch vector, which upstream's sampler never needs because it rebuilds the
            # Batch from the cropped data list. One complex, so it is all zeros.
            g["receptor"].batch = torch.zeros(g["receptor"].x.shape[0], dtype=torch.long)
            g["ligand"].batch = torch.zeros(g["ligand"].x.shape[0], dtype=torch.long)

        # The port's receptor x carries the ESM block already concatenated; upstream's carries
        # the residue-type column plus the same block, so the two must agree column for column.
        checks = {
            "lig_x": (g["ligand"].x.numpy(), d["lig_x"]),
            "bond_index": (g["ligand", "ligand"].edge_index.numpy(), d["bond_index"]),
            "bond_attr": (g["ligand", "ligand"].edge_attr.numpy(), d["bond_attr"]),
            "edge_mask": (g["ligand"].edge_mask.numpy(), d["edge_mask"]),
            "rec_x": (g["receptor"].x.numpy(), d["rec_x"]),
            "rec_pos": (g["receptor"].pos.numpy(), d["rec_pos"]),
            "rec_edge_index": (g["receptor", "receptor"].edge_index.numpy(),
                               d["rec_edge_index"]),
        }
        diffs = {}
        for k, (u, p) in checks.items():
            u, p = np.asarray(u), np.asarray(p)
            if u.shape != p.shape:
                diffs[k] = f"shape {u.shape} vs {p.shape}"
            else:
                diffs[k] = float(np.abs(u.astype(np.float64) - p.astype(np.float64)).max())
        results[Path(path).stem] = {"graph_maxdiff": diffs, "cropped_at": cropped_at,
                                    "n_rec_after_crop": int(g["receptor"].x.shape[0])}
        bad = {k: v for k, v in diffs.items()
               if isinstance(v, str) or v > (a.tol if k == "rec_x" else 0.0)}
        if bad:
            print(f"{path}: GRAPH MISMATCH {bad}", flush=True)
            continue

        set_time(g, None, t_tr, t_rot, t_tor, 1, getattr(sma, "all_atoms", False), device)
        with torch.no_grad():
            tr, rot, tor = model(g)[:3]
        ours = {k: d[k] for k in ("tr", "rot", "tor")}
        row = results[Path(path).stem]
        row.update({
            "t_tr": t_tr, "tr_sigma": float(tr_sigma),
            "upstream": {k: census(v) for k, v in
                         (("tr", tr), ("rot", rot), ("tor", tor))},
            "port": {k: census(v) for k, v in ours.items()},
            "pcc": {k: pcc(v.numpy(), ours[k]) for k, v in
                    (("tr", tr), ("rot", rot), ("tor", tor))},
        })
        print(Path(path).stem, json.dumps(row, indent=1), flush=True)

    Path(a.out).write_text(json.dumps({"complex": cid, "seed": a.seed, "rows": results},
                                      indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
