"""Verify exact dense-vs-sparse-QK RFD3 forward and trajectory parity."""
import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contig", default="A1-10,20,A31-40")
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    from tt_bio.rfd3_sampler import RFD3Sampler

    spec = InputSpecification.from_dict({"input": str(PDB), "contig": args.contig})
    spec.validate()
    f = featurize(str(PDB), spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
         for k, v in f.items()}
    length = f["ref_pos"].shape[0]
    ti = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True)
    dm = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti)
    dev_dm = build_diffusion_module(dm)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})

    generator = torch.Generator().manual_seed(1234)
    noisy = torch.randn(1, length, 3, generator=generator) * 16.0
    times = torch.tensor([8.0])
    os.environ["RFD3_SPARSE_QK"] = "0"
    with torch.no_grad():
        dense_forward = dev_dm(X_noisy_L=noisy, t=times, f=f, **init)
    os.environ["RFD3_SPARSE_QK"] = "1"
    with torch.no_grad():
        sparse_forward = dev_dm(X_noisy_L=noisy, t=times, f=f, **init)
    for key in ("X_L", "sequence_logits_I"):
        maxabs = (dense_forward[key] - sparse_forward[key]).abs().max().item()
        print(f"single-forward {key}: pcc={pcc(dense_forward[key], sparse_forward[key]):.9f} maxabs={maxabs:.9g}")
        assert maxabs == 0.0, f"{key} is not bit-exact"

    sampler = RFD3Sampler(num_timesteps=args.timesteps)
    coord = f["motif_pos"].float().unsqueeze(0)
    fixed = f["is_motif_atom_with_fixed_coord"]
    os.environ["RFD3_SPARSE_QK"] = "0"
    with torch.no_grad():
        dense, _ = sampler.sample(
            dev_dm, 1, length, coord, f, init, fixed,
            generator=torch.Generator().manual_seed(args.seed),
        )
    os.environ["RFD3_SPARSE_QK"] = "1"
    with torch.no_grad():
        sparse, _ = sampler.sample(
            dev_dm, 1, length, coord, f, init, fixed,
            generator=torch.Generator().manual_seed(args.seed),
        )
    maxabs = (dense - sparse).abs().max().item()
    corr = pcc(dense, sparse)
    print(f"{args.timesteps}-step trajectory: pcc={corr:.9f} maxabs={maxabs:.9g}")
    assert maxabs == 0.0, "trajectory is not bit-exact"
    print("SPARSE_QK_PARITY PASS")


if __name__ == "__main__":
    main()
