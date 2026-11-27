"""The confidence pass on the card: parity, rank agreement and time, in one run.

    TT_VISIBLE_DEVICES=<chip> python scripts/diffdock_port/conf_card_bench.py --chip <chip> --out X.json

Three parts, all against the host port in the same process:

1. Per convolution: every one of the confidence model's conv layers run on the card and, in the
   same call, in float64 on the host (``convs_on_device(grade=True)``), on one golden pose set.
2. The assembled head on the three golden pose sets ``dfd-confidence`` captured from upstream
   (``confidence_inputs.npz``, the identical graph on both sides): PCC, max relative error, top-1
   and the full ranking against upstream's rebuild and its rollout capture.
3. Time on dfd-perf's ``dock_stages`` recipe (5S8I_2LY, 10 poses, ``DockOptions(device="tt")``):
   one card rollout for the poses, then ``network.confidence(graph, poses)`` for the host network
   and for the device network, *n* times each after one warm call. AICLK is sampled every ~2 s
   during the whole run and attributed to each timed window.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

GOLDEN_IDS = ("8G43_ZU6", "8AEU_M0L", "7A9E_R4W")
TT_SMI = Path.home() / ".local/bin/tt-smi"


class Clock(threading.Thread):
    """``(epoch, aiclk MHz)`` for *chip*, every ~2 s, until :meth:`stop`."""

    def __init__(self, chip: int):
        super().__init__(daemon=True)
        self.chip, self.samples, self._halt = chip, [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                out = subprocess.run([str(TT_SMI), "-s"], capture_output=True, text=True,
                                     timeout=30).stdout
                clk = json.loads(out)["device_info"][self.chip]["telemetry"].get("aiclk")
                self.samples.append((time.time(), int(str(clk).split()[0])))
            except Exception:                    # a missed sample is a gap, not a crash
                pass
            self._halt.wait(2.0)

    def stop(self):
        self._halt.set()

    def during(self, t0: float, t1: float) -> list[int]:
        return [c for t, c in self.samples if t0 <= t <= t1]


def _order(a) -> list[int]:
    return np.argsort(-np.asarray(a, np.float64).reshape(len(a), -1)[:, 0], kind="stable").tolist()


def _spearman(a, b) -> float:
    ra = np.argsort(np.argsort(-np.asarray(a, np.float64)))
    rb = np.argsort(np.argsort(-np.asarray(b, np.float64)))
    return float(np.corrcoef(ra, rb)[0, 1])


def _max_rel(got, ref) -> float:
    """Max |got - ref| / max(|ref|, rms(ref)): confidence_parity.py's definition."""
    g, r = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    scale = max(float(np.sqrt((r ** 2).mean())), 1e-12)
    return float(np.max(np.abs(g - r) / np.maximum(np.abs(r), scale)))


def _agreement(got, ref) -> dict:
    return dict(pcc=float(np.corrcoef(np.ravel(got), np.ravel(ref))[0, 1]),
                max_rel_err=_max_rel(got, ref),
                max_abs=float(np.max(np.abs(np.asarray(got) - np.asarray(ref)))),
                top1_same=_order(got)[0] == _order(ref)[0],
                order_same=_order(got) == _order(ref),
                spearman=_spearman(np.asarray(got)[:, 0], np.asarray(ref)[:, 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--goldens", default=str(Path.home() / "scratch/dfd/goldens"))
    ap.add_argument("--pb", default=str(Path.home() / "scratch/dfd/pb"))
    args = ap.parse_args()

    import copy
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    from test_diffdock_confidence_device import score_golden

    from tt_bio.diffdock import chunking, esm, network, neighbour_cap, sampling
    from tt_bio.diffdock import input as dfd_input
    from tt_bio.diffdock import run as dock_run
    from tt_bio.diffdock.device import convs_on_device
    from tt_bio.diffdock.job import DockJob, DockOptions

    clock = Clock(args.chip)
    clock.start()
    res = dict(chip=args.chip, torch_threads=torch.get_num_threads(), cpu_count=os.cpu_count(),
               loadavg_start=os.getloadavg())

    host = network.load(DockOptions(device="cpu"))
    dev = network.load(DockOptions(device="tt"))
    device = dev.device

    # 1. per convolution, float64 golden, on the first golden set
    gold = Path(args.goldens)
    graded = copy.deepcopy(host.confidence_model)
    logs = convs_on_device(graded, device, grade=True)
    score_golden(graded, gold / f"{GOLDEN_IDS[0]}__s0")
    per = {k: m.log for k, m in logs.items() if m.log}
    res["per_conv"] = dict(
        golden=GOLDEN_IDS[0], convs_graded=len(per),
        pcc_min=min(e["pcc"] for v in per.values() for e in v if e["pcc"] is not None),
        max_rel_err_max=max(e["max_rel_err"] for v in per.values() for e in v),
        rel_frobenius_max=max(e["rel_frobenius"] for v in per.values() for e in v),
        worst=max(((k, e["max_rel_err"]) for k, v in per.items() for e in v),
                  key=lambda x: x[1]))
    del graded
    print("per_conv", json.dumps(res["per_conv"]), flush=True)

    # 2. the assembled head on the three golden pose sets
    res["goldens"] = {}
    for cid in GOLDEN_IDS:
        d = gold / f"{cid}__s0"
        got = score_golden(dev.host.confidence_model, d)
        ref_host = score_golden(host.confidence_model, d)
        rebuild = np.load(d / "confidence.npz")["confidence_raw"]
        rollout = np.load(d / "poses.npz")["confidence_raw"]
        res["goldens"][cid] = dict(
            n_poses=len(got), card_vs_upstream_rebuild=_agreement(got, rebuild),
            card_vs_upstream_rollout=_agreement(got, rollout),
            card_vs_host_port=_agreement(got, ref_host),
            host_port_vs_upstream_rebuild=_agreement(ref_host, rebuild))
        print(cid, json.dumps(res["goldens"][cid]["card_vs_upstream_rebuild"]), flush=True)

    # 3. time on dock_stages' recipe
    pb = Path(args.pb)
    job = DockJob(name="5S8I_2LY", receptor_path=pb / "5S8I_2LY_protein.pdb",
                  ligand=str(pb / "5S8I_2LY_ligand.sdf"))
    options = DockOptions(device="tt")
    out = Path(args.out).with_suffix(".work")
    out.mkdir(parents=True, exist_ok=True)
    receptor = dock_run._receptor_structure(job, out)
    spec = dfd_input.ComplexSpec(name=job.name, protein_path=str(receptor),
                                 ligand_description=job.ligand)
    emb = esm.receptor_embeddings(dfd_input.chain_sequences_for_embedding(receptor),
                                  on_device=True)
    graph = dfd_input.build_complex_graph(spec, seed=options.seed, receptor_path=receptor,
                                          lm_embeddings=emb, all_atoms=dev.config.all_atoms)
    neighbour_cap.check_ligand_graph(graph.ligand["pos"], dev.config.lig_max_radius,
                                     label=job.name, on_input=True)
    cfg = dock_run._sampling_config(options)
    bounds = sampling.SigmaBounds.from_config(dev.config)
    noise = sampling.Noise(seed=options.seed)
    ei = graph.ligand["edge_index"].T[graph.ligand["edge_mask"]]
    start = sampling.randomize_position(
        graph.ligand["pos"], receptor_pos=graph.receptor["pos"],
        n_samples=options.diffusion_samples, cfg=cfg, bounds=bounds, noise=noise,
        edge_index=ei, mask_rotate=graph.ligand["mask_rotate"],
        no_torsion=dev.config.no_torsion, centre=None)
    n_tor = 0 if dev.config.no_torsion else len(graph.ligand["mask_rotate"])
    score_fn = chunking.chunked_score_fn(dev.score_fn(graph), options.max_parallel_samples,
                                         n_torsions=n_tor,
                                         reset=getattr(dev, "drop_trace", None))
    poses = sampling.sample(score_fn, start, cfg=cfg, bounds=bounds, noise=noise,
                            edge_index=ei, mask_rotate=graph.ligand["mask_rotate"],
                            no_torsion=dev.config.no_torsion,
                            crop_beyond_a=dev.config.crop_beyond, trajectory=False).poses

    timing = {}
    for name, net in (("host", host), ("device", dev)):
        net.confidence(graph, poses)                        # warm: kernels, graph uploads
        runs, clk, scores = [], [], None
        for _ in range(args.n):
            t0 = time.time()
            scores = net.confidence(graph, poses)
            t1 = time.time()
            runs.append(t1 - t0)
            clk += clock.during(t0, t1)
        timing[name] = dict(s=runs, median_s=float(np.median(runs)), aiclk_samples=clk,
                            scores=torch.as_tensor(scores).tolist())
        print(name, runs, clk, flush=True)
    h, d = np.array(timing["host"]["scores"]), np.array(timing["device"]["scores"])
    res["timing"] = timing
    res["dock_recipe_rank"] = dict(
        complex="5S8I_2LY", n_poses=len(h), top1_same=int(np.argmax(h)) == int(np.argmax(d)),
        order_same=_order(h[:, None]) == _order(d[:, None]), spearman=_spearman(h, d),
        max_abs=float(np.max(np.abs(h - d))))
    res["loadavg_end"] = os.getloadavg()
    clock.stop()
    res["aiclk_all"] = sorted({c for _, c in clock.samples})
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
