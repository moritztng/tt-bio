"""Verify the device-side tilization upload path is bit-identical to the host one.

Runs one full RFD3 denoiser forward twice in the same process -- once with
_DEVICE_TILIZE_MIN_ELEMENTS at its shipped value (device tilize) and once with it
disabled (the previous host-tilize path) -- and compares every output element.

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_device_tilize_parity.py \
      --contig "A1-10,230,A31-40" --batch 1
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
    parser.add_argument("--trace-decoder", action="store_true")
    args = parser.parse_args()
    if args.trace_decoder:
        os.environ["RFD3_TRACE_DECODER"] = "1"
        os.environ.setdefault("TT_BIO_TRACE_REGION_SIZE", str(1 << 28))

    import ttnn
    from tt_bio import rfd3
    from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification

    if args.spec:
        spec_data = json.loads(args.spec.read_text())
        path = Path(spec_data["input"])
        spec_data["input"] = str((path if path.is_absolute() else args.spec.parent / path).resolve())
    else:
        spec_data = {"input": str(args.pdb), "contig": args.contig}
    spec = InputSpecification.from_dict(spec_data)
    spec.validate()
    features = featurize(spec_data["input"], spec)
    features = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in features.items()}

    ti = build_token_initializer(torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt",
                                            map_location="cpu", weights_only=True))
    dm = build_diffusion_module(torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt",
                                          map_location="cpu", weights_only=True))
    with torch.no_grad():
        initial = ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in features.items()})

    length = features["ref_pos"].shape[0]
    shipped = rfd3._DEVICE_TILIZE_MIN_ELEMENTS
    ok = True
    for batch in args.batches:
        noisy = torch.randn(batch, length, 3, generator=torch.Generator().manual_seed(42)) * 16.0
        times = torch.full((batch,), 8.0)
        outs = {}
        for label, threshold in (("device_tilize", shipped), ("host_tilize", float("inf"))):
            rfd3._DEVICE_TILIZE_MIN_ELEMENTS = threshold
            with torch.no_grad():
                outs[label] = dm(X_noisy_L=noisy, t=times, f=features, **initial)
        rfd3._DEVICE_TILIZE_MIN_ELEMENTS = shipped
        for key in ("X_L", "sequence_logits_I"):
            a = outs["device_tilize"][key].float()
            b = outs["host_tilize"][key].float()
            maxabs = (a - b).abs().max().item()
            ok &= maxabs == 0.0 and bool(torch.isfinite(a).all())
            print(f"TILIZE_PARITY batch={batch} atoms={length} {key} "
                  f"maxabs={maxabs:.6e} finite={torch.isfinite(a).all().item()}", flush=True)
    print("DEVICE TILIZE PARITY " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
