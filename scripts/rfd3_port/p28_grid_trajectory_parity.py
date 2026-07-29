"""Accuracy cost of a numerics-changing lever, measured over a FULL diffusion trajectory.

`RFD3_FAST_GRID=1` pins a core grid on the linears that `BATCH_INVARIANT_GRID = None`
deliberately leaves on ttnn's default program config. That regroups the fp32 accumulation, so
it is not bit-exact (p28 §1 measured non-exactness on all 22 shipped shapes) and a single-step
diff says nothing useful -- a diffusion trajectory compounds per-step differences over 199
steps (`rfd3-batch-default-parity-fix`: batch 8 once measured 0.94 PCC at 200 steps from
per-step differences too small to see at 20).

So this dumps the FINAL coordinates of a full run and diffs two dumps. Two processes are
needed (one device context per process, and the flag is read at import), which is sound here
because the sampler takes an explicit `torch.Generator`: the same seed draws the same noise in
either process, so the two runs share their random draws by construction and the diff is pure
numerics (`diffusion-port-parity-shared-draws` -- no re-seeding needed, nothing to correct for).

The number only means something against a scale, so `--compare` also accepts a
DIFFERENT-SEED dump of the baseline. That fixes the two ends of the ruler:

  * 0.0        -- bit-exact, what every lever this lineage has merged measured.
  * seed-to-seed RMSD -- two independent samples of the same design, i.e. the scale on which
    "a completely different structure" lives.

A drift near the first end is a rounding perturbation of the same sample; a drift near the
second means the lever changed which structure comes out, which is a different claim entirely.

Usage:
  # arm A (shipped), arm B (fast grid), and a different-seed baseline for the scale
  RFD3_FAST_GRID=0 ... p28_grid_trajectory_parity.py --dump /tmp/p28/base_s2000.pt --seed 2000
  RFD3_FAST_GRID=1 ... p28_grid_trajectory_parity.py --dump /tmp/p28/fast_s2000.pt --seed 2000
  RFD3_FAST_GRID=0 ... p28_grid_trajectory_parity.py --dump /tmp/p28/base_s2001.pt --seed 2001
  p28_grid_trajectory_parity.py --compare /tmp/p28/base_s2000.pt /tmp/p28/fast_s2000.pt \
      --scale /tmp/p28/base_s2001.pt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, help="run a trajectory and save the result here")
    ap.add_argument("--compare", type=Path, nargs=2, metavar=("A", "B"))
    ap.add_argument("--scale", type=Path,
                    help="a different-seed dump of arm A, to calibrate the drift")
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--warmup-timesteps", type=int, default=4)
    ap.add_argument("--pdb", type=Path, default=PDB)
    ap.add_argument("--contig", default="A1-10,230,A31-40")
    ap.add_argument("--spec", type=Path, help="JSON InputSpecification; overrides --contig")
    args = ap.parse_args()
    if not args.dump and not args.compare:
        ap.error("need --dump or --compare")
    return args


def pcc(a, b):
    a = a.double().flatten() - a.double().flatten().mean()
    b = b.double().flatten() - b.double().flatten().mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d > 0 else float("nan")


def rmsd(a, b):
    """Plain per-design RMSD over atoms, no alignment.

    The right primary metric here: both runs start from the SAME noise and the motif atoms
    are held at fixed coordinates, so there is no global rotation freedom for an alignment
    to absorb. Any displacement is real drift.
    """
    d = (a.double() - b.double()).pow(2).sum(-1)          # [D, L]
    return d.mean(-1).sqrt()                              # [D]


def kabsch_rmsd(a, b):
    """Weighted-rigid-aligned RMSD, per design, host fp32/fp64.

    Reported alongside the plain RMSD because it is the shape-only number a structural
    biologist would quote. SVD stays on host in double precision (`ttnn-host-kabsch`).
    """
    out = []
    for x, y in zip(a.double(), b.double()):
        xc, yc = x - x.mean(0), y - y.mean(0)
        u, _, vt = torch.linalg.svd(xc.T @ yc)
        sign = torch.sign(torch.det(vt.T @ u.T))
        dd = torch.diag(torch.tensor([1.0, 1.0, sign], dtype=torch.float64))
        rot = vt.T @ dd @ u.T
        out.append((xc @ rot.T - yc).pow(2).sum(-1).mean().sqrt())
    return torch.stack(out)


def run(args):
    from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    from tt_bio.rfd3_sampler import RFD3Sampler
    import tt_bio.rfd3 as rfd3

    if args.spec:
        spec_data = json.loads(args.spec.read_text())
        p = Path(spec_data["input"])
        spec_data["input"] = str((p if p.is_absolute() else args.spec.parent / p).resolve())
        fixture = f"spec={args.spec.name}"
    else:
        spec_data = {"input": str(args.pdb), "contig": args.contig}
        fixture = f"contig={args.contig!r}"
    spec = InputSpecification.from_dict(spec_data)
    spec.validate()
    features = featurize(spec_data["input"], spec)
    features = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in features.items()}
    token_initializer = build_token_initializer(torch.load(
        GOLDEN_DIR / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True))
    diffusion_module = build_diffusion_module(torch.load(
        GOLDEN_DIR / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True))
    with torch.no_grad():
        initial = token_initializer({k: (v.clone() if torch.is_tensor(v) else v)
                                     for k, v in features.items()})
    length = features["ref_pos"].shape[0]
    fixed = features["is_motif_atom_with_fixed_coord"]
    coord = features["motif_pos"].float().unsqueeze(0)
    grid = rfd3.BATCH_INVARIANT_GRID
    meta = {
        "fixture": fixture, "L": length, "D": args.batch, "seed": args.seed,
        "timesteps": args.timesteps,
        "fast_grid": bool(rfd3.FAST_GRID),
        "batch_invariant_grid": None if grid is None else [grid.x, grid.y],
        "core_grid_main": [rfd3.CORE_GRID_MAIN.x, rfd3.CORE_GRID_MAIN.y],
    }
    print("META " + json.dumps(meta), flush=True)
    with torch.no_grad():
        # Warm the kernel cache so the timed/dumped run is steady state, exactly as
        # p27_real_design_timing.py does. Uses a separate sampler and a separate seed so it
        # cannot touch the measured run's draws.
        RFD3Sampler(num_timesteps=args.warmup_timesteps).sample(
            diffusion_module, args.batch, length, coord, features, initial, fixed,
            generator=torch.Generator().manual_seed(7 + args.batch))
        t0 = time.perf_counter()
        output, extra = RFD3Sampler(num_timesteps=args.timesteps).sample(
            diffusion_module, args.batch, length, coord, features, initial, fixed,
            generator=torch.Generator().manual_seed(args.seed))
        wall = time.perf_counter() - t0
    meta["run_s"] = round(wall, 4)
    meta["finite"] = bool(torch.isfinite(output).all().item())
    args.dump.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": output.detach().float().cpu(), "meta": meta}, args.dump)
    print("DUMP " + json.dumps({**meta, "path": str(args.dump)}), flush=True)


def compare(args):
    a = torch.load(args.compare[0], map_location="cpu", weights_only=False)
    b = torch.load(args.compare[1], map_location="cpu", weights_only=False)
    for key in ("L", "D", "seed", "timesteps"):
        if a["meta"][key] != b["meta"][key]:
            raise SystemExit(f"dumps disagree on {key}: "
                             f"{a['meta'][key]} vs {b['meta'][key]} -- not comparable")
    xa, xb = a["X"], b["X"]
    row = {
        "arm_a": {k: a["meta"][k] for k in ("fast_grid", "batch_invariant_grid")},
        "arm_b": {k: b["meta"][k] for k in ("fast_grid", "batch_invariant_grid")},
        "L": a["meta"]["L"], "D": a["meta"]["D"], "seed": a["meta"]["seed"],
        "timesteps": a["meta"]["timesteps"],
        "maxabs": float((xa.double() - xb.double()).abs().max()),
        "pcc": pcc(xa, xb),
        "rmsd_A": [round(v, 6) for v in rmsd(xa, xb).tolist()],
        "kabsch_rmsd_A": [round(v, 6) for v in kabsch_rmsd(xa, xb).tolist()],
        "run_s": [a["meta"].get("run_s"), b["meta"].get("run_s")],
    }
    if args.scale:
        s = torch.load(args.scale, map_location="cpu", weights_only=False)
        if s["meta"]["seed"] == a["meta"]["seed"]:
            raise SystemExit("--scale must be a DIFFERENT seed than --compare A")
        xs = s["X"]
        row["scale_seed"] = s["meta"]["seed"]
        row["scale_rmsd_A"] = [round(v, 6) for v in rmsd(xa, xs).tolist()]
        row["scale_kabsch_rmsd_A"] = [round(v, 6) for v in kabsch_rmsd(xa, xs).tolist()]
        row["scale_pcc"] = pcc(xa, xs)
        drift = max(row["rmsd_A"])
        scale = max(row["scale_rmsd_A"])
        row["drift_as_frac_of_seed_change"] = round(drift / scale, 6) if scale else None
    print("COMPARE " + json.dumps(row, indent=2), flush=True)


def main():
    args = parse_args()
    if args.dump:
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
