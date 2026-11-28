"""Where the card confidence pass spends its seconds, per stage, on real dock poses.

    TT_VISIBLE_DEVICES=<chip> python scripts/diffdock_port/conf_stage_split.py \
        --pb <posebusters dir> --work <dir> --out X.json [--ids 5S8I_2LY 6M2B_EZO 5SB2_1K2]

Per complex: one seeded card rollout gives the pose set (kept in ``<work>/<id>.npz`` and reused
on the next run, so a before/after pair scores the same poses), then
``network.confidence(graph, poses)`` once cold (``first_call_s``), *n* times timed, and once
more on the same poses moved 0.5 A (``new_shapes_s``: new edge counts in a warm process). Every stage of the pass is
wrapped with an exclusive timer, the card is drained before the read-back so device seconds
are not booked to it, and AICLK plus load are sampled during the timed block
(``perf/clocksample.py``). Host scores on the same poses give top-1, order and Spearman.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


class Stages:
    """Exclusive wall seconds per named stage: a nested stage's time leaves its parent's."""

    def __init__(self):
        self.s, self.n, self.stack = defaultdict(float), defaultdict(int), []

    def wrap(self, owner, attr, name):
        fn = getattr(owner, attr)

        def timed(*a, **k):
            t0 = time.perf_counter()
            self.stack.append(0.0)
            try:
                return fn(*a, **k)
            finally:
                dt = time.perf_counter() - t0
                self.s[name] += dt - self.stack.pop()
                self.n[name] += 1
                if self.stack:
                    self.stack[-1] += dt

        setattr(owner, attr, timed)

    def reset(self):
        self.s.clear()
        self.n.clear()

    def table(self, runs: int) -> dict:
        return {k: dict(s=round(v / runs, 4), calls=self.n[k] // runs)
                for k, v in sorted(self.s.items(), key=lambda kv: -kv[1])}


def _spearman(a, b) -> float:
    ra, rb = (np.argsort(np.argsort(-np.asarray(x, np.float64))) for x in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _files(pb: Path, cid: str):
    d = pb / cid if (pb / cid).is_dir() else pb
    return d / f"{cid}_protein.pdb", d / f"{cid}_ligand.sdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", nargs="+", default=["5S8I_2LY", "6M2B_EZO", "5SB2_1K2"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--edge-tp", default="gemm")
    ap.add_argument("--ckpt", help="directory holding score_model/ and confidence_model/")
    ap.add_argument("--replay", action="store_true",
                    help="also capture one pose's 45 convolutions as a trace and time its replay "
                         "against the eager body: the ceiling of a capture/replay lever")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import ttnn
    from perf.clocksample import during

    from tt_bio.diffdock import chunking, confidence_model, device as dev_mod, esm, network
    from tt_bio.diffdock import input as dfd_input, neighbour_cap, receptor as rec_mod, sampling
    from tt_bio.diffdock import run as dock_run, structure
    from tt_bio.diffdock.job import DockJob, DockOptions

    ck = ({} if args.ckpt is None else
          dict(checkpoint=f"{args.ckpt}/score_model",
               confidence_checkpoint=f"{args.ckpt}/confidence_model"))
    options = DockOptions(device="tt", edge_tp=args.edge_tp, seed=0, **ck)
    if args.replay:                       # the trace region is fixed when the device opens
        from tt_bio.tenstorrent import get_device
        get_device(trace="bc2")
    host = network.load(DockOptions(device="cpu", **ck))
    dev = network.load(options)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    st = Stages()
    cm = dev.host.confidence_model
    st.wrap(structure, "parse_receptor", "host: parse receptor")
    st.wrap(rec_mod, "build_receptor_graph", "host: all-atom receptor graph")
    st.wrap(cm, "receptor_side", "host: receptor encoders + harmonics")
    st.wrap(cm, "ligand_side", "host: ligand side + cross edges")
    st.wrap(cm, "head", "host: head")
    st.wrap(dev_mod.DeviceConfidence, "_edges", "upload: edge attrs, harmonics, mask")
    st.wrap(dev_mod.DeviceConvLayer, "graph", "upload: gather index + dense scatter")
    st.wrap(dev_mod.DeviceConvLayer, "_group_index", "upload: group indices")
    st.wrap(dev_mod.DeviceConvLayer, "fc1_split", "upload: fc1 blocks (first call)")
    st.wrap(dev_mod, "to_device", "upload: node tables")
    st.wrap(cm, "convolve", "card: 45 conv dispatch")
    real_to_torch = ttnn.to_torch

    def drained(t, *a, **k):
        t0 = time.perf_counter()
        ttnn.synchronize_device(dev.device)
        st.s["card: drain queue"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        out = real_to_torch(t, *a, **k)
        st.s["read-back"] += time.perf_counter() - t0
        return out

    res = dict(chip=os.environ.get("TT_VISIBLE_DEVICES"), edge_tp=args.edge_tp, n=args.n,
               torch_threads=torch.get_num_threads(), loadavg_start=os.getloadavg(),
               complexes={})
    for cid in args.ids:
        pdb, sdf = _files(Path(args.pb), cid)
        job = DockJob(name=cid, receptor_path=pdb, ligand=str(sdf))
        receptor = dock_run._receptor_structure(job, work / cid)
        spec = dfd_input.ComplexSpec(name=cid, protein_path=str(receptor),
                                     ligand_description=job.ligand)
        emb = esm.receptor_embeddings(dfd_input.chain_sequences_for_embedding(receptor),
                                      on_device=True)
        graph = dfd_input.build_complex_graph(spec, seed=options.seed, receptor_path=receptor,
                                              lm_embeddings=emb, all_atoms=dev.config.all_atoms)
        keep = work / f"{cid}.npz"
        if keep.exists():
            poses = torch.from_numpy(np.load(keep)["poses"])
        else:
            neighbour_cap.check_ligand_graph(graph.ligand["pos"], dev.config.lig_max_radius,
                                             label=cid, on_input=True)
            cfg = dock_run._sampling_config(options)
            bounds = sampling.SigmaBounds.from_config(dev.config)
            noise = sampling.Noise(seed=options.seed)
            ei = graph.ligand["edge_index"].T[graph.ligand["edge_mask"]]
            rot = graph.ligand["mask_rotate"]
            start = sampling.randomize_position(
                graph.ligand["pos"], receptor_pos=graph.receptor["pos"],
                n_samples=options.diffusion_samples, cfg=cfg, bounds=bounds, noise=noise,
                edge_index=ei, mask_rotate=rot, no_torsion=dev.config.no_torsion, centre=None)
            score_fn = chunking.chunked_score_fn(
                dev.score_fn(graph), options.max_parallel_samples,
                n_torsions=0 if dev.config.no_torsion else len(rot),
                reset=getattr(dev, "drop_trace", None))
            t0 = time.perf_counter()
            poses = sampling.sample(score_fn, start, cfg=cfg, bounds=bounds, noise=noise,
                                    edge_index=ei, mask_rotate=rot,
                                    no_torsion=dev.config.no_torsion,
                                    crop_beyond_a=dev.config.crop_beyond, trajectory=False).poses
            print(cid, "rollout", round(time.perf_counter() - t0, 2), "s", flush=True)
            np.savez(keep, poses=torch.as_tensor(poses).numpy())

        ref = torch.as_tensor(host.confidence(graph, poses)).double().numpy()
        t0 = time.perf_counter()
        dev.confidence(graph, poses)                          # warm: kernels, programs
        first = time.perf_counter() - t0
        ttnn.to_torch = drained
        st.reset()
        runs = []
        with during(period=0.5) as clk:
            for _ in range(args.n):
                t0 = time.perf_counter()
                got = dev.confidence(graph, poses)
                runs.append(time.perf_counter() - t0)
        ttnn.to_torch = real_to_torch
        got = torch.as_tensor(got).double().numpy()
        # What the next ligand pays in a warm process: the same complex with every pose moved
        # 0.5 A rms, so the cross and radius edge counts, and with them the shapes, are new.
        moved = torch.as_tensor(poses) + 0.5 * torch.randn(
            torch.as_tensor(poses).shape, generator=torch.Generator().manual_seed(1))
        t0 = time.perf_counter()
        dev.confidence(graph, moved)
        new_shapes = time.perf_counter() - t0
        table = st.table(args.n)
        table["card: 45 conv dispatch"]["s"] = round(
            table["card: 45 conv dispatch"]["s"]
            - sum(v["s"] for k, v in table.items() if k.startswith("upload: gather")
                  or k.startswith("upload: group") or k.startswith("upload: fc1")), 4)
        g = cm.ligand_side(cm.receptor_side(*_receptor_args(graph, dev)), *_ligand_args(graph, poses[0]))
        res["complexes"][cid] = dict(
            n_poses=len(poses), lig_atoms=int(graph.ligand["pos"].shape[0]),
            nodes={k: int(v.shape[0]) for k, v in g["nodes"].items()},
            edges={k: int(v[0].shape[1]) for k, v in g["edges"].items()},
            first_call_s=round(first, 3), new_shapes_s=round(new_shapes, 3), runs_s=[round(r, 3) for r in runs],
            median_s=round(float(np.median(runs)), 3), stages=table,
            clock=clk.summary().get(0), clock_line=clk.line(), load_per_core=clk.load,
            scores_card=got.tolist(), scores_host=ref.tolist(),
            top1_same=int(np.argmax(got)) == int(np.argmax(ref)),
            order_same=np.argsort(-got, kind="stable").tolist()
            == np.argsort(-ref, kind="stable").tolist(),
            spearman=_spearman(got, ref), max_abs=float(np.max(np.abs(got - ref))))
        c = res["complexes"][cid]
        if args.replay:
            c["replay"] = _replay(dev._confidence, g, args.n)
            print(cid, "replay", c["replay"], flush=True)
        print(cid, "first", c["first_call_s"], "new shapes", c["new_shapes_s"], c["runs_s"],
              c["clock_line"], "top1", c["top1_same"], "rho", c["spearman"],
              flush=True)
        for k, v in table.items():
            print(f"   {v['s']:8.4f} s  x{v['calls']:<5d} {k}", flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
    res["loadavg_end"] = os.getloadavg()
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out, flush=True)


def _replay(dc, g, n: int) -> dict:
    """Pose *g*'s convolutions issued from Python against the same work replayed from a trace.

    Both are timed to an empty card queue with every input already on the card, so the
    difference is what Python dispatch costs and what a capture could give back per pose.
    """
    import ttnn
    from tt_bio.tenstorrent import capture_trace
    dc(g)                                 # uploads this pose's tables, edges and graphs
    nodes = {s: dc._held[("node", s)][1] for s in g["nodes"]}
    edges = {es: dc._held[("edge", es)][1] for es in g["edges"]}
    device = dc.device

    def body():
        return dc.model.convolve(nodes, edges, conv=dc._conv, combine=dc._combine)

    def timed(run):
        secs = []
        for _ in range(n):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = run()
            ttnn.synchronize_device(device)
            secs.append(round(time.perf_counter() - t0, 4))
        return secs, out

    eager_s, eager = timed(body)
    want = ttnn.to_torch(eager)
    tid, out = capture_trace(device, body)
    try:
        replay_s, _ = timed(lambda: ttnn.execute_trace(device, tid, cq_id=0, blocking=True))
        gap = float((ttnn.to_torch(out) - want).abs().max())
    finally:
        ttnn.release_trace(device, tid)
    return dict(eager_s=eager_s, replay_s=replay_s, replay_vs_eager_max_abs=gap)


def _receptor_args(graph, dev):
    """The confidence receptor graph of *graph*, as :meth:`ConfidenceModel.receptor_side` takes it."""
    from tt_bio.diffdock import input as dfd_input
    from tt_bio.diffdock.receptor import build_receptor_graph
    from tt_bio.diffdock.structure import parse_receptor
    rec = build_receptor_graph(parse_receptor(graph.receptor_path),
                               lm_embeddings=graph.lm_embeddings, **dfd_input.CONFIDENCE_GRAPH)
    a = rec["atom"]
    return (rec["x"].float(), rec["pos"].float(), rec["edge_index"].long(), a["x"].float(),
            a["pos"].float(), a["edge_index"].long(), a["atom_rec_edge_index"].long())


def _ligand_args(graph, pose):
    lig = graph.ligand
    return (lig["x"].float(), torch.as_tensor(pose).float() + graph.original_center,
            lig["edge_index"].long(), lig["edge_attr"].float())


if __name__ == "__main__":
    main()
