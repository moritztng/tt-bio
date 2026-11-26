"""Dump upstream DiffDock-L's dynamic graphs at three diffusion steps, for one complex.

Runs in the reference environment (``~/scratch/dfd/venv-ref``), which has torch_cluster and
e3nn; tt-bio has neither, which is the whole reason this file exists. It writes a small npz
that ``tests/test_diffdock_graph.py`` grades the static-graph port against:

* the ligand and receptor coordinates the sampler actually reached at the step,
* the sigma-dependent cross cutoff,
* every edge torch_cluster returned for the three dynamic graphs (ligand radius, ligand ->
  receptor cross, torsion bond conv), for pose 0, plus the per-pose edge counts for all ten,
* a sample of upstream's own edge features and spherical harmonics, so the port's
  featurisation is pinned against e3nn and not only against itself.

Only pose 0's edge lists are stored: the ten poses share the ligand's bond graph and differ
only in coordinates, so one pose carries the geometry question and the other nine are
covered by their edge counts. That keeps the fixture at tens of kilobytes.

    ~/scratch/dfd/venv-ref/bin/python scripts/diffdock_port/dump_graph_golden.py \
        --id 8G43_ZU6 --out tests/data/diffdock_graph/8G43_ZU6_steps.npz
"""
import argparse
import copy
import json
import os
import sys
from argparse import Namespace
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

D = Path.home() / "scratch/dfd"
sys.path.insert(0, str(D / "scripts"))
import dfd_ref_lib as L  # noqa: E402

PB = D / "data/posebusters_dl/sel"
LM_CACHE = D / "data/lm_cache"
SAMPLE_EDGES = 64       # edges per graph whose upstream features are kept


def lm_embeddings_for(complex_id, protein_path):
    cache = LM_CACHE / f"{complex_id}.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=False)
    from esm import pretrained
    from utils.inference_utils import compute_ESM_embeddings, get_sequences
    model, alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    model.eval()
    seqs = get_sequences([protein_path], [None])
    labels, sequences = [], []
    for j, s in enumerate(seqs[0].split(":")):
        sequences.append(s)
        labels.append(f"{complex_id}_chain_{j}")
    emb = compute_ESM_embeddings(model, alphabet, labels, sequences)
    out = [[emb[f"{complex_id}_chain_{j}"] for j in range(len(sequences))]]
    LM_CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    del model
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="8G43_ZU6")
    ap.add_argument("--steps", default="0,8,18")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    want = [int(s) for s in args.steps.split(",")]
    out_path = Path(args.out).resolve()

    L.setup(args.seed)
    from datasets.process_mols import write_mol_with_coords  # noqa: F401  (import check)
    from torch_geometric.loader import DataLoader
    from utils.diffusion_utils import get_t_schedule, t_to_sigma as t_to_sigma_compl
    from utils.inference_utils import InferenceDataset
    from utils.sampling import randomize_position, sampling
    from utils.utils import get_model

    cid = args.id
    prot, lig = PB / cid / f"{cid}_protein.pdb", PB / cid / f"{cid}_ligand.sdf"
    assert prot.exists() and lig.exists(), f"missing inputs for {cid}"
    run_dir = D / "runs/graph-golden" / cid
    run_dir.mkdir(parents=True, exist_ok=True)

    inf = yaml.safe_load((D / "upstream/default_inference_args.yaml").read_text())
    with open(D / "ckpt/extracted/score_model/model_parameters.yml") as f:
        sma = Namespace(**yaml.full_load(f))

    ds = InferenceDataset(
        out_dir=str(run_dir), complex_names=[cid], protein_files=[str(prot)],
        ligand_descriptions=[str(lig)], protein_sequences=[None], lm_embeddings=True,
        receptor_radius=sma.receptor_radius, remove_hs=sma.remove_hs,
        c_alpha_max_neighbors=sma.c_alpha_max_neighbors, all_atoms=sma.all_atoms,
        atom_radius=sma.atom_radius, atom_max_neighbors=sma.atom_max_neighbors,
        precomputed_lm_embeddings=lm_embeddings_for(cid, str(prot)),
        knn_only_graph=(not sma.not_knn_only_graph) if hasattr(sma, "not_knn_only_graph") else False)

    device = torch.device("cpu")
    t_to_sigma = partial(t_to_sigma_compl, args=sma)
    model = get_model(sma, device, t_to_sigma=t_to_sigma, no_parallel=True,
                      old=inf.get("old_score_model", False))
    model.load_state_dict(torch.load(D / "ckpt/extracted/score_model" / inf["ckpt"],
                                     map_location="cpu", weights_only=False), strict=True)
    model.eval()

    n = inf["samples_per_complex"]
    graph = next(iter(DataLoader(dataset=ds, batch_size=1, shuffle=False)))
    assert bool(graph.success[0]), f"dataset failed to build {cid}"
    data_list = [copy.deepcopy(graph) for _ in range(n)]

    sched = get_t_schedule(sigma_schedule=inf["sigma_schedule"],
                           inference_steps=inf["inference_steps"],
                           inf_sched_alpha=inf["inf_sched_alpha"],
                           inf_sched_beta=inf["inf_sched_beta"])

    # --- capture ------------------------------------------------------------------
    # One score-model forward per diffusion step. Patch the three builders that call
    # torch_cluster and keep what they saw, for the steps asked for.
    store, state = {}, {"step": -1}
    build_cross, build_lig, build_bond = (model.build_cross_conv_graph,
                                          model.build_lig_conv_graph,
                                          model.build_bond_conv_graph)
    fwd = model.forward

    def forward(data):
        state["step"] += 1
        state["data"] = data
        return fwd(data)

    def keep(step, **kw):
        store.setdefault(step, {}).update(kw)

    def cross(data, cutoff):
        out = build_cross(data, cutoff)
        if state["step"] in want:
            keep(state["step"],
                 lig_pos=data["ligand"].pos.detach().numpy(),
                 lig_batch=data["ligand"].batch.numpy(),
                 rec_pos=data["receptor"].pos.detach().numpy(),
                 rec_batch=data["receptor"].batch.numpy(),
                 cross_cutoff=np.asarray(cutoff.detach().numpy() if torch.is_tensor(cutoff)
                                         else cutoff, dtype=np.float64),
                 cross_edge_index=out[0].numpy().astype(np.int32),
                 cross_edge_attr=out[1].detach().numpy(),
                 cross_edge_sh=out[2].detach().numpy(),
                 rev_cross_edge_sh=out[3].detach().numpy())
        return out

    def ligg(data):
        out = build_lig(data)
        if state["step"] in want:
            keep(state["step"],
                 lig_edge_index=out[1].numpy().astype(np.int32),
                 lig_edge_attr=out[2].detach().numpy(),
                 lig_edge_sh=out[3].detach().numpy(),
                 lig_bond_edge_index=data["ligand", "ligand"].edge_index.numpy().astype(np.int32),
                 lig_bond_edge_attr=data["ligand", "ligand"].edge_attr.detach().numpy(),
                 lig_node_sigma_emb=data["ligand"].node_sigma_emb.detach().numpy(),
                 lig_edge_mask=data["ligand"].edge_mask.numpy())
        return out

    def bondg(data):
        out = build_bond(data)
        if state["step"] in want:
            keep(state["step"],
                 tor_bonds=out[0].numpy().astype(np.int32),
                 tor_edge_index=out[1].numpy().astype(np.int32),
                 tor_edge_sh=out[3].detach().numpy())
        return out

    model.forward, model.build_cross_conv_graph = forward, cross
    model.build_lig_conv_graph, model.build_bond_conv_graph = ligg, bondg

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    randomize_position(data_list, sma.no_torsion, False, sma.tr_sigma_max,
                       initial_noise_std_proportion=inf["initial_noise_std_proportion"])
    sampling(data_list=data_list, model=model, inference_steps=inf["actual_steps"],
             tr_schedule=sched, rot_schedule=sched, tor_schedule=sched, device=device,
             t_to_sigma=t_to_sigma, model_args=sma, confidence_model=None,
             confidence_data_list=None, confidence_model_args=None, batch_size=n,
             no_final_step_noise=inf["no_final_step_noise"],
             temp_sampling=[inf["temp_sampling_tr"], inf["temp_sampling_rot"], inf["temp_sampling_tor"]],
             temp_psi=[inf["temp_psi_tr"], inf["temp_psi_rot"], inf["temp_psi_tor"]],
             temp_sigma_data=[inf["temp_sigma_data_tr"], inf["temp_sigma_data_rot"],
                              inf["temp_sigma_data_tor"]])

    # --- reduce to pose 0 and write ------------------------------------------------
    flat, meta = {}, {"complex": cid, "seed": args.seed, "steps": want,
                      "upstream": str(D / "upstream"), "samples_per_complex": n}
    rng = np.random.default_rng(0)
    for step in want:
        s = store[step]
        nl = int((s["lig_batch"] == 0).sum())
        nr = int((s["rec_batch"] == 0).sum())
        p = f"s{step}_"
        flat[p + "lig_pos"] = s["lig_pos"][:nl].astype(np.float32)
        flat[p + "rec_pos"] = s["rec_pos"][:nr].astype(np.float32)
        flat[p + "cross_cutoff"] = s["cross_cutoff"].reshape(-1)[:1].astype(np.float64)
        flat[p + "lig_node_sigma_emb"] = s["lig_node_sigma_emb"][:nl].astype(np.float32)

        def pose0(key, srcs, limits):
            """Columns of an edge index whose endpoints all belong to pose 0."""
            ei = s[key]
            keep_cols = np.ones(ei.shape[1], dtype=bool)
            for row, lim in zip(srcs, limits):
                keep_cols &= ei[row] < lim
            return keep_cols

        # cross: row 0 indexes ligand nodes, row 1 receptor nodes (both batched)
        m = pose0("cross_edge_index", (0, 1), (nl, nr))
        flat[p + "cross_edge_index"] = s["cross_edge_index"][:, m]
        flat[p + "cross_edge_counts"] = np.bincount(
            s["lig_batch"][s["cross_edge_index"][0]], minlength=n).astype(np.int32)
        # ligand: bond edges first, then the radius edges
        nbond = s["lig_bond_edge_index"].shape[1] // n
        lig_ei = s["lig_edge_index"]
        mb = pose0("lig_edge_index", (0, 1), (nl, nl))
        flat[p + "lig_edge_index"] = lig_ei[:, mb]
        flat[p + "lig_n_bond_edges_per_pose"] = np.asarray([nbond], dtype=np.int32)
        flat[p + "lig_bond_edge_index"] = s["lig_bond_edge_index"][:, :nbond]
        flat[p + "lig_bond_edge_attr"] = s["lig_bond_edge_attr"][:nbond].astype(np.float32)
        flat[p + "lig_edge_mask"] = s["lig_edge_mask"][:nbond]
        # torsion bond conv: row 0 indexes bond centres, row 1 ligand nodes
        nbc = int(s["lig_edge_mask"][:nbond].sum())
        mt = pose0("tor_edge_index", (0, 1), (nbc, nl))
        flat[p + "tor_edge_index"] = s["tor_edge_index"][:, mt]
        flat[p + "tor_bonds"] = s["tor_bonds"][:, :nbc]

        # feature samples, graded against e3nn / upstream's own expansions
        for name, idx_key, feats in (("cross", "cross_edge_index",
                                      ("cross_edge_attr", "cross_edge_sh", "rev_cross_edge_sh")),
                                     ("lig", "lig_edge_index", ("lig_edge_attr", "lig_edge_sh")),
                                     ("tor", "tor_edge_index", ("tor_edge_sh",))):
            cols = np.flatnonzero({"cross": m, "lig": mb, "tor": mt}[name])
            pick = cols if cols.size <= SAMPLE_EDGES else rng.choice(cols, SAMPLE_EDGES, replace=False)
            pick = np.sort(pick)
            # index within pose 0's own edge list, which is what the test rebuilds
            flat[f"{p}{name}_sample_rows"] = np.searchsorted(cols, pick).astype(np.int32)
            for f in feats:
                flat[f"{p}{f}_sample"] = s[f][pick].astype(np.float32)
        meta.setdefault("shapes", {})[str(step)] = {
            "n_lig": nl, "n_rec": nr, "n_bond_edges": nbond, "n_rot_bonds": nbc,
            "cross_edges_pose0": int(m.sum()), "lig_edges_pose0": int(mb.sum()),
            "tor_edges_pose0": int(mt.sum()),
            "cross_cutoff": float(flat[p + "cross_cutoff"][0]),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, meta=json.dumps(meta), **flat)
    print(json.dumps({"out": str(out_path), "bytes": out_path.stat().st_size, **meta["shapes"]},
                     indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TORCH_HOME", str(D / "torchhub"))
    main()
