"""Grade `device_loop.DeviceStep` against `static_net.static_forward` on a card.

    TT_VISIBLE_DEVICES=1 PYTHONPATH=. python3 scripts/diffdock_port/device_step_grade.py --out s.json

The whole step with nothing on the host, against the same step as the shipped device network
runs it -- same weights, same StaticComplex, same positions, same sigmas. The host side is the
float64 network, so the comparison is against the reference and not against another
approximation; the device convolutions inside `static_forward` are the ones the shipped path
uses, so what this isolates is everything DeviceStep moved off the host.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

DATA = Path(__file__).resolve().parents[2] / "tests/fixtures/diffdock"


def stats(got: torch.Tensor, ref: torch.Tensor) -> dict:
    g, r = got.detach().double().flatten(), ref.detach().double().flatten()
    return {"max_abs": float((g - r).abs().max()),
            "rel_frobenius": float((g - r).norm() / r.norm().clamp_min(1e-30)),
            "pcc": float(torch.corrcoef(torch.stack([g, r]))[0, 1]) if r.numel() > 1 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=int, default=4)
    ap.add_argument("--step", type=float, default=0.73, help="t_tr of the graded step")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio.diffdock import device_loop as dl, esm, input as dfd_input, network, sampling
    from tt_bio.diffdock.conv import pad_nodes
    from tt_bio.diffdock.job import DockOptions
    from tt_bio.diffdock.features import timestep_embedding
    from tt_bio.diffdock.static_net import StaticComplex, static_forward

    model = network.load(DockOptions(device="tt"))        # the shipped seam, gemm form
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

    g = torch.Generator().manual_seed(0)
    poses = lig["pos"].float().expand(P, n_lig, 3) + torch.randn(P, 1, 3, generator=g) * 2
    lp = pad_nodes(poses.reshape(P * n_lig, 3), c.sg.n_lig_padded)

    cfg = model.config
    bounds = sampling.SigmaBounds.from_config(cfg)
    t = args.step
    tr_sigma, rot_sigma, tor_sigma = (float(x) for x in bounds.sigmas(t, t, t))
    crop = None if cfg.crop_beyond is None else 3 * tr_sigma + cfg.crop_beyond

    out = {"poses": P, "t_tr": t, "tr_sigma": tr_sigma, "crop_cutoff": crop}
    flush = lambda: Path(args.out).write_text(json.dumps(out, indent=1))

    with torch.no_grad():
        ref = static_forward(model.model, c, lp, t, tr_sigma, rot_sigma, tor_sigma,
                             crop_cutoff=crop)
    out["host_ok"] = True
    flush()

    step = dl.DeviceStep(model.device, model.model, c, model.convs)
    sigma_row = dl.upload(model.device,
                          timestep_embedding(torch.tensor([t]), cfg.sigma_embed_dim,
                                             cfg.embedding_scale).float())
    tr, rot, tor = step(dl.upload(model.device, lp), sigma_row, tr_sigma, rot_sigma, tor_sigma,
                        crop_cutoff=crop)
    out["tr"] = stats(ttnn.to_torch(tr)[:P, :3], ref[0])
    out["rot"] = stats(ttnn.to_torch(rot)[:P, :3], ref[1])
    if tor is not None and ref[2].numel():
        n = ref[2].shape[0]
        out["tor"] = stats(ttnn.to_torch(tor)[:n, 0], ref[2])
    print(json.dumps(out, indent=1), flush=True)
    flush()


if __name__ == "__main__":
    main()
