"""Grade `device_loop.DeviceStep`'s towers against `static_net.static_forward`'s own values.

    TT_VISIBLE_DEVICES=<chip> PYTHONPATH=. python3 \\
        scripts/diffdock_port/device_towers_grade.py --out t.json

`device_step_grade.py` says the step's tr/rot/tor are wrong; it cannot say which tower did it.
This runs the host block of `static_forward` (static_net.py:140-183) and `DeviceStep` on the
same complex, positions and sigmas, and compares at every boundary between them: the node
encodings, each tower's output, the main stack and the centre head's edge input. The host side
is the shipped reference function's own expressions, kept in the same order, so a mismatch here
names a device block rather than a difference of formulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

DATA = Path(__file__).resolve().parents[2] / "tests/fixtures/diffdock"


def stats(got, ref) -> dict:
    g, r = got.detach().double().flatten(), ref.detach().double().flatten()
    n = min(g.numel(), r.numel())
    g, r = g[:n], r[:n]
    return {"n": int(n),
            "max_abs": float((g - r).abs().max()),
            "rel_frobenius": float((g - r).norm() / r.norm().clamp_min(1e-30)),
            "pcc": float(torch.corrcoef(torch.stack([g, r]))[0, 1]) if n > 1 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=int, default=4)
    ap.add_argument("--step", type=float, default=0.73)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio.diffdock import device_loop as dl, esm, input as dfd_input, network, sampling
    from tt_bio.diffdock.conv import pad_nodes
    from tt_bio.diffdock.job import DockOptions
    from tt_bio.diffdock.features import timestep_embedding
    from tt_bio.diffdock.harmonics import spherical_harmonics
    from tt_bio.diffdock.static_net import (StaticComplex, main_stack_plan, pad_width,
                                            run_stack)

    model = network.load(DockOptions(device="tt"))
    spec = dfd_input.ComplexSpec(name="6Z5Z_BDF", protein_path=str(DATA / "6Z5Z_BDF_protein.pdb"),
                                 ligand_description=str(DATA / "6Z5Z_BDF_ligand.sdf"))
    embeddings = esm.receptor_embeddings(
        dfd_input.chain_sequences_for_embedding(Path(spec.protein_path)), on_device=False)
    graph = dfd_input.build_complex_graph(spec, seed=0, receptor_path=Path(spec.protein_path),
                                          lm_embeddings=embeddings,
                                          all_atoms=model.config.all_atoms)
    lig, rec = graph.ligand, graph.receptor
    P, n_lig = args.poses, lig["pos"].shape[0]
    c = StaticComplex.build(lig, model.host._receptor_x(rec), rec["pos"].float(),
                            rec["edge_index"], P)
    sg, m, cfg, ns = c.sg, model.model, model.config, model.model.ns

    g = torch.Generator().manual_seed(0)
    poses = lig["pos"].float().expand(P, n_lig, 3) + torch.randn(P, 1, 3, generator=g) * 2
    lig_pos = pad_nodes(poses.reshape(P * n_lig, 3), sg.n_lig_padded)

    t = args.step
    bounds = sampling.SigmaBounds.from_config(cfg)
    tr_sigma, rot_sigma, tor_sigma = (float(x) for x in bounds.sigmas(t, t, t))
    out = {"poses": P, "t_tr": t, "crop_cutoff": None}
    flush = lambda: Path(args.out).write_text(json.dumps(out, indent=1))

    # ---- the host side: static_forward's own block, crop off (keep = all ones) -------------
    dt = lig_pos.dtype
    with torch.no_grad():
        sigma_emb = timestep_embedding(torch.as_tensor([t], dtype=dt), cfg.sigma_embed_dim,
                                       cfg.embedding_scale)
        lig_sig = sigma_emb.expand(sg.n_lig_padded, -1)
        rei = c.rec_edge_index
        keep = pad_nodes(torch.ones(sg.n_rec_total, 1, dtype=dt), sg.n_rec_padded)[:, 0]
        rec_mask = sg.rec_edge_mask(keep, rei)
        rvec = c.rec_pos[rei[1]] - c.rec_pos[rei[0]]
        rec_eattr = m.rec_edge_embedding(m.rec_distance_expansion(rvec.norm(dim=-1)))
        rec_sh = spherical_harmonics(rvec, m.sh_ls)
        rec_enc = m.rec_node_embedding(c.rec_x)
        rec_h = run_stack(m.rec_emb_layers, rec_enc, rei, rec_eattr, rec_sh, rec_mask, ns)
        rec_sig = m.rec_sigma_embedding(sigma_emb)
        rec_h = rec_h + 0
        rec_h[:, :ns] = rec_h[:, :ns] + rec_sig
        rec_eattr_s = rec_eattr + rec_sig

        lei = sg.lig_edge_index
        lig_mask = sg.lig_mask(lig_pos, cfg.lig_max_radius)
        lvec = lig_pos[lei[1]] - lig_pos[lei[0]]
        bond = torch.zeros(lei.shape[1], c.bond_attr.shape[1], dtype=dt)
        bond[:sg.lig_n_bond_edges] = c.bond_attr
        lig_eattr = m.lig_edge_embedding(torch.cat(
            [bond, lig_sig[lei[0]], m.lig_distance_expansion(lvec.norm(dim=-1))], 1))
        lig_sh = spherical_harmonics(lvec, m.sh_ls)
        lig_enc = m.lig_node_embedding(torch.cat([c.lig_x, lig_sig], 1))
        lig_h_tower = run_stack(m.lig_emb_layers, lig_enc, lei, lig_eattr, lig_sh, lig_mask, ns)

        cutoff = ((torch.as_tensor(tr_sigma, dtype=dt) * 3 + 20).expand(P)
                  if cfg.dynamic_max_cross else cfg.cross_max_distance)
        cei = sg.cross_edge_index
        cross_m = sg.cross_mask(lig_pos, c.rec_pos, cutoff) * keep[cei[1]]
        cvec = c.rec_pos[cei[1]] - lig_pos[cei[0]]
        cross_eattr = m.cross_edge_embedding(torch.cat(
            [lig_sig[cei[0]], m.cross_distance_expansion(cvec.norm(dim=-1))], 1))
        cross_sh = spherical_harmonics(cvec, m.sh_ls)
        rev_sh = spherical_harmonics(-cvec, m.sh_ls)
        node = torch.cat([pad_width(lig_h_tower, rec_h.shape[1]), rec_h], 0)
        ei, plan, _ = main_stack_plan(m, sg, rei)
        ea = torch.cat([lig_eattr, cross_eattr, rec_eattr_s, cross_eattr], 0)
        esh = torch.cat([lig_sh, cross_sh, rec_sh, rev_sh], 0)
        em = torch.cat([lig_mask, cross_m, rec_mask, cross_m], 0)
        node_main = run_stack(m.conv_layers, node, ei, ea, esh, em, ns, plan)
        lig_h_main = node_main[:sg.rec_offset]
    out["host_ok"] = True
    flush()

    # ---- the device side -------------------------------------------------------------------
    step = dl.DeviceStep(model.device, m, c, model.convs)
    step.capture = {}
    sigma_row = dl.upload(model.device,
                          timestep_embedding(torch.tensor([t]), cfg.sigma_embed_dim,
                                             cfg.embedding_scale).float())
    step(dl.upload(model.device, lig_pos), sigma_row, tr_sigma, rot_sigma, tor_sigma,
         crop_cutoff=None)
    cap = {k: ttnn.to_torch(v) for k, v in step.capture.items()}

    # the device's own encodings, outside the step
    out["rec_enc"] = stats(ttnn.to_torch(step.rec_enc()), rec_enc)
    out["lig_enc"] = stats(ttnn.to_torch(step.lig_enc(sigma_row)), lig_enc)
    out["rec_eattr"] = stats(cap["rec_eattr"], rec_eattr_s)
    out["lig_eattr"] = stats(cap["lig_eattr"], lig_eattr)
    out["rec_h"] = stats(cap["rec_h"][:rec_h.shape[0], :rec_h.shape[1]], rec_h)
    out["lig_h_tower"] = stats(cap["lig_h_tower"][:lig_h_tower.shape[0], :lig_h_tower.shape[1]],
                               lig_h_tower)
    out["node_main"] = stats(cap["node_main"][:node_main.shape[0], :node_main.shape[1]],
                             node_main)
    out["lig_h_main"] = stats(cap["lig_h_main"][:lig_h_main.shape[0], :lig_h_main.shape[1]],
                              lig_h_main)
    print(json.dumps(out, indent=1), flush=True)
    flush()


if __name__ == "__main__":
    main()
