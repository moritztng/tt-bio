"""Bit-exactness gate for the cached -1e4 attention-mask template.

A single forward cannot catch what this change could break. The template is now
allocated once and reused, while the neighbour graph -- and therefore the scatter
index written into it -- moves on every sampling step. A stale value surviving from
step N-1 would only show up in a real multi-step trajectory, so this gate runs the
actual RFD3Sampler loop and compares the final coordinates elementwise.

Two independent trees, not two processes on one tree: point PYTHONPATH at a
`git archive` of the pre-change commit for the reference, then at the worktree for
the comparison. Nothing in the runtime is toggled, so there is no chance of the
"fast path" and the "reference path" sharing a bug.

  git archive <pre-change-sha> | tar -x -C /tmp/ref && cp <this file> /tmp/ref/scripts/rfd3_port/
  PYTHONPATH=/tmp/ref python3 /tmp/ref/scripts/rfd3_port/verify_mask_template_parity.py \
      --contig ... --batches 1 8 --dump-to /tmp/ref.pt
  PYTHONPATH=$WT python3 scripts/rfd3_port/verify_mask_template_parity.py \
      --contig ... --batches 1 8 --compare-to /tmp/ref.pt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contig", default="A1-10,20,A31-40")
    parser.add_argument("--pdb", type=Path, default=PDB)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1])
    parser.add_argument("--timesteps", type=int, default=8,
                        help="Sampler steps. Must be >1 for the reuse to be exercised.")
    parser.add_argument("--trace-decoder", action="store_true")
    parser.add_argument("--dump-to", type=Path)
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args()
    if args.trace_decoder:
        os.environ["RFD3_TRACE_DECODER"] = "1"
        os.environ.setdefault("TT_BIO_TRACE_REGION_SIZE", str(1 << 28))

    from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    from tt_bio.rfd3_sampler import RFD3Sampler

    if args.spec:
        spec_data = json.loads(args.spec.read_text())
        path = Path(spec_data["input"])
        spec_data["input"] = str(
            (path if path.is_absolute() else args.spec.parent / path).resolve())
    else:
        spec_data = {"input": str(args.pdb), "contig": args.contig}
    spec = InputSpecification.from_dict(spec_data)
    spec.validate()
    features = featurize(spec_data["input"], spec)
    features = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in features.items()}

    ti = build_token_initializer(torch.load(
        GOLDEN_DIR / "token_initializer.real_weights.pt",
        map_location="cpu", weights_only=True))
    dm = build_diffusion_module(torch.load(
        GOLDEN_DIR / "diffusion_module.real_weights.pt",
        map_location="cpu", weights_only=True))
    with torch.no_grad():
        initial = ti({k: (v.clone() if torch.is_tensor(v) else v)
                      for k, v in features.items()})

    length = features["ref_pos"].shape[0]
    fixed = features["is_motif_atom_with_fixed_coord"]
    coord = features["motif_pos"].float().unsqueeze(0)

    outputs = {}
    for batch in args.batches:
        sampler = RFD3Sampler(num_timesteps=args.timesteps)
        with torch.no_grad():
            out, _ = sampler.sample(
                dm, batch, length, coord, features, initial, fixed,
                generator=torch.Generator().manual_seed(2000 + batch),
            )
        outputs[batch] = out.float().cpu()

    if args.dump_to:
        args.dump_to.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"atoms": length, "timesteps": args.timesteps,
                    "outputs": outputs}, args.dump_to)
        print(f"DUMPED atoms={length} steps={args.timesteps} batches={list(outputs)}")
        return

    if not args.compare_to:
        parser.error("pass --dump-to or --compare-to")
    reference = torch.load(args.compare_to, weights_only=False)
    assert reference["atoms"] == length, "reference is from a different fixture"
    assert reference["timesteps"] == args.timesteps, "reference used a different step count"
    ok = True
    for batch, got in outputs.items():
        ref = reference["outputs"][batch]
        maxabs = (ref - got).abs().max().item()
        finite = bool(torch.isfinite(got).all())
        ok &= maxabs == 0.0 and finite
        print(f"MASK_TEMPLATE_PARITY batch={batch} atoms={length} "
              f"steps={args.timesteps} X maxabs={maxabs:.6e} finite={finite}",
              flush=True)
    print("MASK TEMPLATE PARITY " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
