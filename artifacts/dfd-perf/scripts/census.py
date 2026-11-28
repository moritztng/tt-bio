"""Card-free: static edge counts and live fractions for WORKLOAD-S ligands on 5SAK_ZRY."""
import sys, json, torch
torch.set_num_threads(4)
from pathlib import Path
from tt_bio.diffdock import network, esm, input as dfd_input, sampling
from tt_bio.diffdock.job import DockOptions, read_csv_jobs
from tt_bio.diffdock import run as dock_run
from tt_bio.diffdock.static_net import StaticComplex, pad_nodes, main_stack_plan
WL = Path("/home/ttuser/scratch/dfd/wl/workload"); OUT = Path("/home/ttuser/scratch/dfd/perf/census")
o = DockOptions(device="cpu")
model = network.load(o); host = getattr(model, "host", model)
cfg = dock_run._sampling_config(o); bounds = sampling.SigmaBounds.from_config(model.config)
jobs = read_csv_jobs(WL / "screen72.csv")
for job in [jobs[i] for i in (8, 9, 12)]:
    rp = dock_run._receptor_structure(job, OUT / job.name)
    spec = dfd_input.ComplexSpec(name=job.name, protein_path=str(rp), ligand_description=job.ligand)
    emb = esm.receptor_embeddings(dfd_input.chain_sequences_for_embedding(rp), on_device=False)
    g = dfd_input.build_complex_graph(spec, seed=o.seed, receptor_path=rp, lm_embeddings=emb, all_atoms=model.config.all_atoms)
    lig, rec = g.ligand, g.receptor
    P = 10; n_lig = lig["pos"].shape[0]
    c = StaticComplex.build(lig, host._receptor_x(rec), rec["pos"].float(), rec["edge_index"], P)
    sg = c.sg
    ei, plan, (s1, s2, s3) = main_stack_plan(host.model if hasattr(host, "model") else model.model, sg, c.rec_edge_index)
    row = {"name": job.name, "n_lig": n_lig, "n_rec": rec["pos"].shape[0], "E_lig": s1, "E_cross": s2 - s1, "E_rec": s3 - s2, "E_main": ei.shape[1]}
    noise = sampling.Noise(seed=0)
    start = sampling.randomize_position(lig["pos"], receptor_pos=rec["pos"], n_samples=P, cfg=cfg, bounds=bounds, noise=noise,
        edge_index=lig["edge_index"].T[lig["edge_mask"]], mask_rotate=lig["mask_rotate"], no_torsion=model.config.no_torsion, centre=None)
    final = lig["pos"].float().unsqueeze(0).expand(P, -1, -1)
    for tag, pos, trs in (("step0", start, bounds.tr_max if hasattr(bounds, "tr_max") else 19.0), ("last", final, 0.1)):
        lp = pad_nodes(pos.reshape(P * n_lig, 3).float(), sg.n_lig_padded)
        cut = torch.full((P,), trs * 3 + 20.0)
        keep = sg.rec_keep(lp, c.rec_pos, trs * 3 + model.config.crop_beyond)
        cm = sg.cross_mask(lp, c.rec_pos, cut) * keep[sg.cross_edge_index[1]]
        rm = sg.rec_edge_mask(keep, c.rec_edge_index); lm = sg.lig_mask(lp, model.config.lig_max_radius)
        row[tag] = {"cross_live": int((cm > 0).sum()), "rec_live": int((rm > 0).sum()), "lig_live": int((lm > 0).sum()), "rec_keep": int((keep > 0).sum())}
    print(json.dumps(row), flush=True)
