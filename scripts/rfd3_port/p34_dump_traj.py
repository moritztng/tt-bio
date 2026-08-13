"""Dump the REAL per-step noisy atom coordinates of one RFD3 design trajectory.

The tile-density screen for L1 (do not compute the attention tiles that hold no neighbour)
is worth exactly the tile density of `f["attn_indices"]`, and that density is a property of
a folding protein: it is ~1.0 at step 0, where the coordinates are Gaussian noise at
sigma=2560 A, and it only falls as the structure condenses. Modelling the trajectory as
`clean + sigma*eps` needs a clean structure this fixture does not have on host (the contig
designs 230 of its 250 residues de novo), so this dumps the actual X_noisy_L the sampler
feeds the atom block on every step. One 200-timestep design at 3359 atoms, ~55 s.

Device-only. The analysis it feeds (p34_tile_density.py) is pure host, so this runs once.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tt_bio.rfd3.design import build_diffusion_module, build_token_initializer  # noqa: E402
from tt_bio.rfd3.featurize import featurize  # noqa: E402
from tt_bio.rfd3.input import InputSpecification  # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler  # noqa: E402

# The f fields _create_attention_indices reads, plus what a caller needs to rebuild it.
F_KEYS = ("atom_to_token_map", "asym_id", "unindexing_pair_mask", "entity_id",
          "residue_index", "token_index", "is_motif_atom_with_fixed_coord")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--contig", required=True)
    ap.add_argument("--ckpt", default="/home/ttuser/.boltz/rfd3/weights")
    ap.add_argument("--num_timesteps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = InputSpecification.from_dict({"input": a.pdb, "contig": a.contig})
    spec.validate()
    f = featurize(a.pdb, spec)

    cap = Path(a.ckpt)
    ti_w = torch.load(cap / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True)
    dm_w = torch.load(cap / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti_w)
    dev_dm = build_diffusion_module(dm_w)

    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    L = init["Q_L_init"].shape[0]
    is_motif = f["is_motif_atom_with_fixed_coord"]
    coord0 = f["motif_pos"].float().unsqueeze(0) if "motif_pos" in f else torch.zeros(1, L, 3)
    print(f"[dump] L={L} atoms, I={int(f['atom_to_token_map'].max()) + 1} tokens, "
          f"timesteps={a.num_timesteps}", flush=True)

    sampler = RFD3Sampler(num_timesteps=a.num_timesteps)
    t0 = time.perf_counter()
    with torch.no_grad():
        X, traj = sampler.sample(dev_dm, 1, L, coord0, f, init, is_motif,
                                 generator=torch.Generator().manual_seed(a.seed))
    wall = time.perf_counter() - t0
    print(f"[dump] {len(traj)} steps in {wall:.1f} s = {wall / len(traj) * 1e3:.1f} ms/step", flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "X_noisy": torch.stack([s["X_noisy_L"][0].float().cpu() for s in traj]),   # [steps, L, 3]
        "t_hat": torch.tensor([float(s["t_hat"]) for s in traj]),
        "X_final": X[0].float().cpu(),
        "f": {k: f[k].cpu() for k in F_KEYS if k in f},
        "L": L, "num_timesteps": a.num_timesteps, "seed": a.seed,
        "wall_s": wall, "ms_per_step": wall / len(traj) * 1e3,
    }, out)
    print(f"[dump] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
