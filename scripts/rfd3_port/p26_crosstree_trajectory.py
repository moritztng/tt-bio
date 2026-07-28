"""Dump a full seeded RFD3 trajectory so two trees can be diffed step by step.

`dump_forward_for_crosstree_parity.py` diffs one forward. A host-boundary change can
also be bit-exact per step and still diverge over a trajectory (p4: RFD3 batch 8 once
measured 0.94 PCC at 200 steps from per-step bf16 differences too small to see in a
20-step check), so the claim this makes is about the whole run: identical seed,
identical schedule, `maxabs` on the final coordinates and a per-step digest that says
which step first differed if any did.

Run in the change tree and in a `git archive` of its parent, then `--compare` the two:

  PYTHONPATH=<tree> python3 <tree>/scripts/rfd3_port/p26_crosstree_trajectory.py \
      --out /tmp/traj_p26.pt --contig "A1-10,230,A31-40" --batch 8 --timesteps 200
  python3 ... --compare /tmp/traj_p26.pt /tmp/traj_parent.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()


def compare(a_path: Path, b_path: Path) -> int:
    a = torch.load(a_path, map_location="cpu", weights_only=True)
    b = torch.load(b_path, map_location="cpu", weights_only=True)
    print(f"A {a_path}  L={a['length']} D={a['batch']} steps={a['timesteps']}")
    print(f"B {b_path}  L={b['length']} D={b['batch']} steps={b['timesteps']}")
    if (a["length"], a["batch"], a["timesteps"]) != (b["length"], b["batch"], b["timesteps"]):
        print("MISMATCHED CONFIGURATION -- not comparable")
        return 1
    final = (a["X_L"] - b["X_L"]).abs().max().item()
    digest = (a["digest"] - b["digest"]).abs()
    first = int(digest.nonzero()[0].item()) + 1 if digest.any() else None
    print(f"final coordinates maxabs = {final:.6e}")
    print(f"per-step digest maxabs   = {digest.max().item():.6e}"
          + (f"  first differing step = {first}" if first is not None else ""))
    ok = final == 0.0 and not digest.any()
    print("RESULT:", "PARITY PASS (maxabs 0.0)" if ok else "PARITY FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", type=Path, nargs=2)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--pdb", type=Path, default=PDB)
    ap.add_argument("--contig", default="A1-10,230,A31-40")
    ap.add_argument("--spec", type=Path, help="JSON InputSpecification; overrides --contig")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    if args.out is None:
        ap.error("--out is required unless --compare is given")

    from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    from tt_bio.rfd3_sampler import RFD3Sampler

    if args.spec:
        data = json.loads(args.spec.read_text())
        src = Path(data["input"])
        if not src.is_absolute():
            src = args.spec.parent / src
        data["input"] = str(src.resolve())
    else:
        data = {"input": str(args.pdb), "contig": args.contig}
    spec = InputSpecification.from_dict(data)
    spec.validate()
    features = featurize(data["input"], spec)
    features = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v
                for k, v in features.items()}

    ti = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt",
                    map_location="cpu", weights_only=True)
    dmw = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt",
                     map_location="cpu", weights_only=True)
    with torch.no_grad():
        initial = build_token_initializer(ti)(
            {k: v.clone() if torch.is_tensor(v) else v for k, v in features.items()})
        length = features["ref_pos"].shape[0]
        sampler = RFD3Sampler(num_timesteps=args.timesteps)
        X_L, traj = sampler.sample(
            build_diffusion_module(dmw), args.batch, length,
            features["motif_pos"].float().unsqueeze(0), features, initial,
            features["is_motif_atom_with_fixed_coord"],
            generator=torch.Generator().manual_seed(args.seed))

    # One float64 sum per step of the denoised coordinates: enough to name the first
    # differing step without carrying 200 full coordinate sets.
    digest = torch.tensor([s["X_denoised_L"].double().sum().item() for s in traj],
                          dtype=torch.float64)
    torch.save({"X_L": X_L.cpu(), "digest": digest, "length": length,
                "batch": args.batch, "timesteps": args.timesteps,
                "fixture": data.get("contig", str(args.spec))}, args.out)
    print(f"L={length} D={args.batch} steps={args.timesteps} -> {args.out}  "
          f"finite={torch.isfinite(X_L).all().item()}  "
          f"digest[0]={digest[0].item():.10e} digest[-1]={digest[-1].item():.10e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
