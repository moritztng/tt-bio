"""Generic RFD3 release-video/QC sampler: run any existing parity fixture
(scripts/rfd3_port/parity_artifacts/<fixture>/<spec>.json) at production step
count and write the real per-step trajectory + final structure, for
downstream geometry/designability checks or rendering. Generalizes
render_1j79_symmetric_ligand.py (kept as-is, it's the C2+ligand hero
candidate's own dedicated script) to any fixture dir, so other candidate
subjects (e.g. non-symmetric motif scaffolding) can be tried the same way.

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/sample_fixture.py \
      --fixture unindex_dictform --spec spec.json --pdb <path-if-not-input-in-spec> \
      --seed 1 --num_timesteps 200 --out_dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
from tt_bio.rfd3_design import _write_cif
from tt_bio.rfd3_featurize import featurize
from tt_bio.rfd3_input import InputSpecification
from tt_bio.rfd3_sampler import RFD3Sampler

DIR = os.path.dirname(__file__)
GOLDEN_DIR = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")


def build_f(fixture_dir, spec_name, pdb_override=None):
    spec_json = os.path.join(fixture_dir, spec_name)
    with open(spec_json) as fh:
        spec_dict = json.load(fh)
    if pdb_override:
        pdb = pdb_override
    else:
        inp = spec_dict.get("input")
        pdb = inp if (inp and os.path.isabs(inp)) else os.path.join(fixture_dir, os.path.basename(inp)) if inp else None
        if pdb is None or not os.path.exists(pdb):
            # fall back: fixtures with no "input" key use a companion .pdb next to the spec
            cands = list(Path(fixture_dir).glob("*.pdb"))
            if cands:
                pdb = str(cands[0])
    spec_dict["input"] = pdb
    spec = InputSpecification.from_dict(spec_dict)
    spec.validate()
    f = featurize(pdb, spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}
    return f, pdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, help="dir name under scripts/rfd3_port/parity_artifacts/")
    ap.add_argument("--spec", default="spec.json")
    ap.add_argument("--pdb", default=None, help="override input pdb path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_timesteps", type=int, default=200)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_recycle", type=int, default=None)
    args = ap.parse_args()

    fixture_dir = os.path.join(DIR, "parity_artifacts", args.fixture)
    out_dir = Path(os.path.expanduser(args.out_dir))
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    struct_dir = out_dir / "struct"
    struct_dir.mkdir(parents=True, exist_ok=True)

    t_feat0 = time.time()
    f, pdb = build_f(fixture_dir, args.spec, args.pdb)
    n_atoms = f["ref_pos"].shape[0]
    print(f"[setup] fixture={args.fixture} spec={args.spec} pdb={pdb}: {n_atoms} atoms "
          f"(featurize took {time.time()-t_feat0:.1f}s)", flush=True)

    ti_weights = torch.load(os.path.join(GOLDEN_DIR, "token_initializer.real_weights.pt"),
                             map_location="cpu", weights_only=True)
    dm_weights = torch.load(os.path.join(GOLDEN_DIR, "diffusion_module.real_weights.pt"),
                             map_location="cpu", weights_only=True)

    t_build0 = time.time()
    dev_ti = build_token_initializer(ti_weights)
    dev_dm = build_diffusion_module(dm_weights)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    init = {k: v.float() for k, v in init.items()}
    print(f"[setup] device modules built + TokenInitializer run in {time.time()-t_build0:.1f}s", flush=True)

    L = f["ref_pos"].shape[0]
    is_motif = f["is_motif_atom_with_fixed_coord"]
    coord0 = f["motif_pos"].float().unsqueeze(0)

    sampler = RFD3Sampler(num_timesteps=args.num_timesteps)
    g = torch.Generator().manual_seed(args.seed)

    t_sample0 = time.time()
    with torch.no_grad():
        X, traj = sampler.sample(dev_dm, 1, L, coord0, f, init, is_motif,
                                  generator=g, n_recycle=args.n_recycle)
    dt_sample = time.time() - t_sample0
    print(f"[sample] {len(traj)} steps in {dt_sample:.1f}s ({dt_sample/max(1,len(traj))*1000:.0f} ms/step)",
          flush=True)

    t_write0 = time.time()
    for k, step in enumerate(traj):
        _write_cif(step["X_L"][0], f, frames_dir / f"f{k:04d}.cif")
    final_path = struct_dir / "design.cif"
    _write_cif(X[0], f, final_path)
    print(f"[write] {len(traj)} frame CIFs + final structure in {time.time()-t_write0:.1f}s", flush=True)

    meta = {
        "fixture": args.fixture, "spec": args.spec, "pdb": pdb,
        "seed": args.seed, "num_timesteps": args.num_timesteps, "n_recycle": args.n_recycle,
        "n_atoms": n_atoms, "sample_wall_clock_s": dt_sample,
        "ms_per_step": dt_sample / max(1, len(traj)) * 1000,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
