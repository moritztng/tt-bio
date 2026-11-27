"""Pose-level and SE(3) re-grade of the score network as a perf lever leaves it.

    PYTHONPATH=<worktree> python3 scripts/diffdock_port/device_pose_grade.py rollout \
        --fixture fx_8G43.pt --net tile --seed 0 --poses 10 --out tile_s0.json
    ... device_pose_grade.py equiv --fixture fx_8G43.pt --net tile --out tile_equiv.json
    ... device_pose_grade.py compare --run tile_s0.json --ref host_s0.json --floor host_s1.json

`rollout` runs the gated recipe end to end (19 steps, 10 poses, one seeded `Noise` for the start
and every step) with `static_net.static_forward` as the score function, so `--net host` and
`--net gemm|tile` differ only in where the convolutions execute and any pose difference is the
network's arithmetic fed back 19 times. Poses are saved, and `compare` quotes the run's deviation
from a reference in Angstrom beside the floor a seed change costs on the same network, which is
what QUALITY.md asks a lever to be graded against.

`equiv` applies a rigid (R, t) to the receptor and every pose on the fixture's steps and checks
that tr and rot come back rotated and tor does not move. The card's rounding sits in both legs,
so the residual is reported beside the card-vs-host gap on the untransformed input: a broken
symmetry reads far above that gap, rounding reads at its size. Seed 0 is the identity control.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import torch

CKPT = Path(os.environ.get("DFD_CKPT", str(Path.home() / "scratch/dfd/ckpt/extracted")))


def rigid_transform(seed: int):
    """`tests/test_diffdock_equivariance.py`'s transform: Haar SO(3), ~10 A translation."""
    if seed == 0:
        return torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    g = torch.Generator().manual_seed(1000 + seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q, torch.randn(3, generator=g, dtype=torch.float64) * 10.0


def rel_residual(got, want) -> float:
    got, want = got.double(), want.double()
    scale = want.abs().max().item()
    return (got - want).abs().max().item() / scale if scale > 0 else 0.0


def build(net: str, fixture: str):
    """(fixture dict, model, convs) with the convolutions where *net* says."""
    from tt_bio.diffdock import config as C
    from tt_bio.diffdock.score_model import load_released

    fx = torch.load(fixture, weights_only=False)
    cfg = C.load(str(CKPT / "score_model"), old_model=False)
    sd = torch.load(CKPT / "score_model/best_ema_inference_epoch_model.pt", map_location="cpu",
                    weights_only=False)
    model = load_released(sd, cfg)
    if net == "host":
        return fx, model, {}
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    from tt_bio.diffdock import device as ddev
    from tt_bio.tenstorrent import get_device
    return fx, model, ddev.convs_on_device(model, get_device(), edge_tp=net)


def receptor_x(model, fx):
    from tt_bio.diffdock.score_model import HostScoreNetwork
    return HostScoreNetwork(model, lm_embeddings=fx["emb"])._receptor_x(fx["graph"].receptor)


def static_complex(model, fx, n_poses: int):
    from tt_bio.diffdock.static_net import StaticComplex
    g = fx["graph"]
    return StaticComplex.build(g.ligand, receptor_x(model, fx), g.receptor["pos"].float(),
                               g.receptor["edge_index"], n_poses)


def score_fn(model, c, n_lig: int):
    """What `sampling.sample` calls, over the fixed-shape forward."""
    from tt_bio.diffdock.conv import pad_nodes
    from tt_bio.diffdock.static_net import static_forward

    def score(step):
        P = step.lig_pos.shape[0]
        lp = pad_nodes(step.lig_pos.reshape(P * n_lig, 3).float(), c.sg.n_lig_padded)
        with torch.no_grad():
            return static_forward(model, c, lp, step.t_tr, step.tr_sigma, step.rot_sigma,
                                  step.tor_sigma, crop_cutoff=step.crop_cutoff)
    return score


class Clock:
    """AICLK sampled during the run; a timing without one is not a measurement."""

    def __init__(self, on: bool):
        self.samples, self.stop, self.on = [], threading.Event(), on
        if on:
            from tt_bio.diffdock.capture import sample_aiclk
            threading.Thread(target=sample_aiclk, daemon=True, args=(
                self.stop, self.samples, os.environ.get("TT_VISIBLE_DEVICES", "0"))).start()

    def summary(self) -> dict:
        from tt_bio.diffdock.capture import clock_summary
        self.stop.set()
        return clock_summary(self.samples) if self.on else {"aiclk_mhz": None, "host": True}


def cmd_rollout(args):
    from tt_bio.diffdock import sampling

    fx, model, convs = build(args.net, args.fixture)
    g = fx["graph"]
    cfg = sampling.SamplingConfig.from_recipe()
    bounds = sampling.SigmaBounds.from_config(model.config)
    noise = sampling.Noise(seed=args.seed)
    start = sampling.randomize_position(
        g.ligand["pos"], receptor_pos=g.receptor["pos"], n_samples=args.poses, cfg=cfg,
        bounds=bounds, noise=noise, edge_index=g.ligand["edge_index"],
        mask_rotate=g.ligand["mask_rotate"], no_torsion=model.config.no_torsion)
    c = static_complex(model, fx, args.poses)
    clock = Clock(args.net != "host")
    t0 = time.perf_counter()
    out = sampling.sample(score_fn(model, c, g.ligand["pos"].shape[0]), start.clone(), cfg=cfg,
                          bounds=bounds, noise=noise, edge_index=g.ligand["edge_index"],
                          mask_rotate=g.ligand["mask_rotate"],
                          no_torsion=model.config.no_torsion,
                          crop_beyond_a=model.config.crop_beyond)
    secs = time.perf_counter() - t0
    rec = {"complex": fx["complex"], "net": args.net, "seed": args.seed, "poses": args.poses,
           "steps": cfg.steps, "seconds_per_complex": secs,
           "seconds_in_device_layers": sum(m.secs for m in convs.values()) or None,
           "poses_out": out.poses.tolist(), **clock.summary()}
    print(json.dumps({k: v for k, v in rec.items() if k != "poses_out"}), flush=True)
    Path(args.out).write_text(json.dumps(rec, indent=1))


def cmd_equiv(args):
    fx, model, convs = build(args.net, args.fixture)
    g = fx["graph"]
    clock = Clock(args.net != "host")
    rows = {}
    for k, step in sorted(fx["steps"].items()):
        P, n_lig = step.lig_pos.shape[0], g.ligand["pos"].shape[0]
        c = static_complex(model, fx, P)
        base = score_fn(model, c, n_lig)(step)
        row = {}
        for seed in (0, 1, 2, 3):
            R, t = rigid_transform(seed)
            moved = fx["graph"].receptor["pos"].double() @ R.T + t
            fx2 = dict(fx)
            g2 = type(g)(**{**g.__dict__, "receptor": {**g.receptor, "pos": moved.float()}})
            fx2["graph"] = g2
            c2 = static_complex(model, fx2, P)
            step2 = type(step)(**{**step.__dict__,
                                  "lig_pos": (step.lig_pos.double() @ R.T + t).float()})
            tr, rot, tor = score_fn(model, c2, n_lig)(step2)
            row[seed] = {"tr": rel_residual(tr, base[0].double() @ R.T),
                         "rot": rel_residual(rot, base[1].double() @ R.T),
                         "tor": rel_residual(tor, base[2])}
            print(f"step {k} seed {seed}: " + " ".join(
                f"{n} {row[seed][n]:.3g}" for n in ("tr", "rot", "tor")), flush=True)
        rows[k] = row
    out = {"net": args.net, "complex": fx["complex"], "bar": 1e-5,
           "upstream_pinned_floor": {"tr": 1.57e-7, "rot": 1.25e-6, "tor": 9.29e-7},
           "steps": rows, **clock.summary()}
    print(json.dumps({k: v for k, v in out.items() if k != "steps"}), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


def cmd_compare(args):
    def poses(p):
        d = json.loads(Path(p).read_text())
        return d, torch.tensor(d["poses_out"], dtype=torch.float64)

    rd, ref = poses(args.ref)
    gd, got = poses(args.run)
    dev = (got - ref).norm(dim=-1)                      # per atom, per pose
    out = {"run": gd["net"], "ref": rd["net"], "seed": gd["seed"],
           "max_atom_deviation_a": dev.max().item(),
           "rmsd_per_pose_a": (dev.pow(2).mean(dim=-1).sqrt()).tolist()}
    if args.floor:
        fd, flo = poses(args.floor)
        fdev = (flo - ref).norm(dim=-1)
        out["seed_floor"] = {"seed": fd["seed"], "net": fd["net"],
                             "max_atom_deviation_a": fdev.max().item(),
                             "rmsd_per_pose_a": fdev.pow(2).mean(dim=-1).sqrt().tolist()}
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rollout")
    r.add_argument("--fixture", required=True)
    r.add_argument("--net", default="tile", choices=["host", "gemm", "tile"])
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--poses", type=int, default=10)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_rollout)
    e = sub.add_parser("equiv")
    e.add_argument("--fixture", required=True)
    e.add_argument("--net", default="tile", choices=["host", "gemm", "tile"])
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_equiv)
    c = sub.add_parser("compare")
    c.add_argument("--run", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--floor")
    c.add_argument("--out")
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
