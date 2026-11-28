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


def build(net: str, fixture: str, gemm_modules=()):
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
    return fx, model, ddev.convs_on_device(model, get_device(), edge_tp=net,
                                           gemm_modules=gemm_modules)


def receptor_x(model, fx):
    from tt_bio.diffdock.score_model import HostScoreNetwork
    return HostScoreNetwork(model, lm_embeddings=fx["emb"])._receptor_x(fx["graph"].receptor)


def static_complex(model, fx, n_poses: int, dt=torch.float32, rec_pos=None):
    """The fixed-shape layout in *dt*; *rec_pos* overrides the fixture's (a moved receptor)."""
    from tt_bio.diffdock.static_net import StaticComplex
    g = fx["graph"]
    lig = {**g.ligand, "edge_attr": g.ligand["edge_attr"].to(dt)}
    pos = g.receptor["pos"] if rec_pos is None else rec_pos
    c = StaticComplex.build(lig, receptor_x(model, fx).to(dt), pos.to(dt),
                            g.receptor["edge_index"], n_poses)
    c.lig_valid = c.lig_valid.to(dt)
    return c


def score_fn(model, c, n_lig: int, dt=torch.float32):
    """What `sampling.sample` calls, over the fixed-shape forward, positions in *dt*."""
    from tt_bio.diffdock.conv import pad_nodes
    from tt_bio.diffdock.static_net import static_forward

    def score(step):
        P = step.lig_pos.shape[0]
        lp = pad_nodes(step.lig_pos.reshape(P * n_lig, 3).to(dt), c.sg.n_lig_padded)
        with torch.no_grad():
            return static_forward(model, c, lp, step.t_tr, step.tr_sigma, step.rot_sigma,
                                  step.tor_sigma, crop_cutoff=step.crop_cutoff)
    return score


def resident_score_fn(model, c, n_lig: int, convs):
    """The same call through `device_loop.DeviceStep`, which is what `tt-bio dock` runs on a
    card: edge vectors, harmonics, embeddings and the heads on the card, not only the convs."""
    import ttnn
    from tt_bio.diffdock import device_loop as dl
    from tt_bio.diffdock.conv import pad_nodes
    from tt_bio.diffdock.features import timestep_embedding
    from tt_bio.tenstorrent import get_device

    dev, mc = get_device(), model.config
    net = dl.DeviceStep(dev, model, c, convs)

    def score(step):
        P = step.lig_pos.shape[0]
        lp = pad_nodes(step.lig_pos.reshape(P * n_lig, 3).float(), c.sg.n_lig_padded)
        row = dl.upload(dev, timestep_embedding(torch.tensor([step.t_tr]), mc.sigma_embed_dim,
                                                mc.embedding_scale).float())
        tr, rot, tor = net(dl.upload(dev, lp), row, step.tr_sigma, step.rot_sigma,
                           step.tor_sigma, crop_cutoff=step.crop_cutoff)
        n_tor = P * c.sg.n_rot_bonds
        return (ttnn.to_torch(tr)[:P, :3].float(), ttnn.to_torch(rot)[:P, :3].float(),
                torch.zeros(0) if tor is None else ttnn.to_torch(tor)[:n_tor, 0].float())
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

    fx, model, convs = build(args.net, args.fixture, args.gemm_modules)
    g = fx["graph"]
    cfg = sampling.SamplingConfig.from_recipe()
    bounds = sampling.SigmaBounds.from_config(model.config)
    noise = sampling.Noise(seed=args.seed)
    # The rotatable bonds, as run.py hands them to the sampler: [T, 2], one row per mask_rotate.
    rot_edges = g.ligand["edge_index"].T[g.ligand["edge_mask"]]
    start = sampling.randomize_position(
        g.ligand["pos"], receptor_pos=g.receptor["pos"], n_samples=args.poses, cfg=cfg,
        bounds=bounds, noise=noise, edge_index=rot_edges,
        mask_rotate=g.ligand["mask_rotate"], no_torsion=model.config.no_torsion)
    c = static_complex(model, fx, args.poses)
    clock = Clock(args.net != "host")
    t0 = time.perf_counter()
    out = sampling.sample(score_fn(model, c, g.ligand["pos"].shape[0]), start.clone(), cfg=cfg,
                          bounds=bounds, noise=noise, edge_index=rot_edges,
                          mask_rotate=g.ligand["mask_rotate"],
                          no_torsion=model.config.no_torsion,
                          crop_beyond_a=model.config.crop_beyond)
    secs = time.perf_counter() - t0
    rec = {"complex": fx["complex"], "net": args.net, "gemm_modules": args.gemm_modules,
           "seed": args.seed, "poses": args.poses,
           "steps": cfg.steps, "seconds_per_complex": secs,
           "seconds_in_device_layers": sum(m.secs for m in convs.values()) or None,
           "poses_out": out.poses.tolist(), **clock.summary()}
    print(json.dumps({k: v for k, v in rec.items() if k != "poses_out"}), flush=True)
    Path(args.out).write_text(json.dumps(rec, indent=1))


def frob(got, want, scale) -> float:
    """||got - want||_F / scale: rotation-invariant, so both legs share one yardstick."""
    return (got.double() - want.double()).norm().item() / scale if scale > 0 else 0.0


EQUIV_RULE = ("per step and score: max gap_b over transforms <= 2*median(gap_b over transforms) "
              "+ 1e-6; a form other than stepped also needs median <= 1.25*the stepped form's "
              "median (--stepped-ref); host_residual <= 1e-5")


def equiv_verdict(steps, ref_steps=None, device=True, needs_ref=False):
    """The SE(3) verdict on the per-step, per-transform gaps of one equivariance run.

    The card's gap to the float64 host is rounding, and rounding is direction-blind, so over
    the transforms (the identity among them) the gaps are one population. A broken symmetry
    puts a transformed leg far outside it. The rule therefore reads the worst gap against the
    population's median, not against the identity's single draw: two forms with the same
    17 gaps passed and failed on where that one draw happened to land (D-dfd-orch-121,
    `state/dfd/PERF.md` section 17). A median cannot see a form that is uniformly worse, so
    a form other than the stepped one is also held to 1.25x the stepped form's median, per
    step and score, and fails closed without that reference.

    *steps* and *ref_steps* are the ``steps`` member of an equiv json, keys as stored.
    Returns ``(ok, table)``; ``table[step][score]`` carries the numbers the verdict used.
    """
    ok, table = True, {}
    for k, row in steps.items():
        table[str(k)] = {}
        for n in ("tr", "rot", "tor"):
            gaps = sorted(v[n]["gap_b"] for v in row.values())
            med = float(torch.tensor(gaps, dtype=torch.float64).median())
            cell = {"n": len(gaps), "median_gap": med, "worst_gap": gaps[-1],
                    "worst_over_median": gaps[-1] / med if med > 0 else 0.0,
                    "worst_host_residual": max(v[n]["host_residual"] for v in row.values())}
            good = cell["worst_host_residual"] <= 1e-5
            if device:
                good = good and gaps[-1] <= 2 * med + 1e-6
                if needs_ref:
                    ref = (ref_steps or {}).get(str(k)) or (ref_steps or {}).get(k)
                    if ref is None:
                        cell["stepped_median_gap"], good = None, False
                    else:
                        rg = torch.tensor([v[n]["gap_b"] for v in ref.values()],
                                          dtype=torch.float64)
                        cell["stepped_median_gap"] = float(rg.median())
                        cell["median_over_stepped"] = med / cell["stepped_median_gap"]
                        good = good and med <= 1.25 * cell["stepped_median_gap"]
            cell["pass"] = bool(good)
            ok = ok and good
            table[str(k)][n] = cell
    return bool(ok), table


def cmd_regrade(args):
    """Re-read stored equiv runs under :func:`equiv_verdict`. No card."""
    run = json.loads(Path(args.run).read_text())
    ref = json.loads(Path(args.stepped_ref).read_text())["steps"] if args.stepped_ref else None
    ok, table = equiv_verdict(run["steps"], ref, device=run["net"] != "host",
                              needs_ref=run.get("form", "stepped") != "stepped")
    out = {"run": args.run, "stepped_ref": args.stepped_ref, "net": run["net"],
           "form": run.get("form", "stepped"), "rule": EQUIV_RULE, "pass": ok, "verdict": table,
           "aiclk_min": run.get("aiclk_min"), "aiclk_max": run.get("aiclk_max")}
    print(json.dumps(out, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


def cmd_equiv(args):
    """Rotate the complex, check tr/rot rotate and tor does not, on the card and on the host.

    A bf16 lever cannot meet the host's 1e-5 bar, so the card's residual alone does not say
    whether a symmetry broke. Beside it, each leg's card-vs-host gap (float64 host network and
    a float64 transform, same static forward; D-dfd-orch-70 -- the fp32 host's own 1.4e-5 was
    round-off, not a symmetry break, and a device net graded against it is graded against another
    approximation): rounding is direction-blind, so a sound kernel is as accurate on the
    rotated input as on the original (gap_b ~ gap_a), while a broken symmetry makes the rotated
    leg wrong by O(1). All gaps are Frobenius-relative to the host's untransformed score.
    The verdict is :func:`equiv_verdict`'s.
    """
    fx, model, convs = build(args.net, args.fixture, args.gemm_modules)
    host = (build("host", args.fixture)[1] if args.net != "host" else model).double()
    f64 = torch.float64
    # `--net host` grades the host against itself, and `host` is `model` doubled in place.
    mdt = f64 if args.net == "host" else torch.float32
    g = fx["graph"]
    clock = Clock(args.net != "host")
    rows = {}
    for k, step in sorted(fx["steps"].items()):
        if args.steps and str(k) not in args.steps:
            continue
        P, n_lig = step.lig_pos.shape[0], g.ligand["pos"].shape[0]
        mstep = type(step)(**{**step.__dict__, "lig_pos": step.lig_pos.to(mdt)})
        def card(c):
            return (resident_score_fn(model, c, n_lig, convs) if args.form == "resident"
                    else score_fn(model, c, n_lig, mdt))
        base = card(static_complex(model, fx, P, mdt))(mstep)
        hstep = type(step)(**{**step.__dict__, "lig_pos": step.lig_pos.double()})
        hbase = score_fn(host, static_complex(host, fx, P, f64), n_lig, f64)(hstep)
        row = {}
        for seed in range(args.seeds):
            R, t = rigid_transform(seed)
            moved = fx["graph"].receptor["pos"].double() @ R.T + t
            mstep2 = type(step)(**{**step.__dict__,
                                   "lig_pos": (step.lig_pos.double() @ R.T + t).to(mdt)})
            got = card(static_complex(model, fx, P, mdt, moved))(mstep2)
            hstep2 = type(step)(**{**step.__dict__,
                                   "lig_pos": step.lig_pos.double() @ R.T + t})
            hgot = score_fn(host, static_complex(host, fx, P, f64, moved), n_lig, f64)(hstep2)
            r = {}
            for i, n in enumerate(("tr", "rot", "tor")):
                def moved_ref(x, i=i):
                    return x.double() @ R.T if i < 2 else x.double()
                scale = hbase[i].double().norm().item()
                r[n] = {"max_rel": rel_residual(got[i], moved_ref(base[i])),
                        "residual": frob(got[i], moved_ref(base[i]), scale),
                        "gap_a": frob(base[i], hbase[i], scale),
                        "gap_b": frob(got[i], hgot[i], scale),
                        "host_residual": frob(hgot[i], moved_ref(hbase[i]), scale)}
            row[seed] = r
            print(f"step {k} seed {seed}: " + "  ".join(
                f"{n} res {v['residual']:.2g} gap_a {v['gap_a']:.2g} gap_b {v['gap_b']:.2g} "
                f"host {v['host_residual']:.2g}" for n, v in r.items()), flush=True)
        rows[k] = row
    ref = json.loads(Path(args.stepped_ref).read_text())["steps"] if args.stepped_ref else None
    ok, table = equiv_verdict({str(k): {str(a): b for a, b in row.items()}
                               for k, row in rows.items()}, ref,
                              device=args.net != "host", needs_ref=args.form != "stepped")
    out = {"net": args.net, "form": args.form, "gemm_modules": args.gemm_modules,
           "complex": fx["complex"], "host_bar": 1e-5, "host_reference": "float64",
           "rule": EQUIV_RULE, "pass": ok, "verdict": table, "stepped_ref": args.stepped_ref,
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
    r.add_argument("--gemm-modules", nargs="*", default=[],
                    help="conv paths (suffix) kept on the fp32 GEMM form under --net tile")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--poses", type=int, default=10)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_rollout)
    e = sub.add_parser("equiv")
    e.add_argument("--fixture", required=True)
    e.add_argument("--net", default="tile", choices=["host", "gemm", "tile"])
    e.add_argument("--gemm-modules", nargs="*", default=[],
                    help="conv paths (suffix) kept on the fp32 GEMM form under --net tile")
    e.add_argument("--seeds", type=int, default=4, help="transform seeds; 0 is the identity")
    e.add_argument("--form", default="stepped", choices=["stepped", "resident"],
                   help="stepped: static_forward with the convs on the card. resident: "
                        "device_loop.DeviceStep, the whole step on the card")
    e.add_argument("--steps", nargs="*", help="fixture steps to grade (default: all)")
    e.add_argument("--stepped-ref", help="the stepped form's equiv json, same fixture and "
                   "transforms: the median guard a resident form is held to")
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_equiv)
    c = sub.add_parser("compare")
    c.add_argument("--run", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--floor")
    c.add_argument("--out")
    c.set_defaults(fn=cmd_compare)
    g = sub.add_parser("regrade")
    g.add_argument("--run", required=True, help="a stored equiv json")
    g.add_argument("--stepped-ref", help="the stepped form's equiv json, for a resident run")
    g.add_argument("--out")
    g.set_defaults(fn=cmd_regrade)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
