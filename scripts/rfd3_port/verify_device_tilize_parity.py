"""Verify the fast upload path is bit-identical to the original host-tilize one.

Two upload changes are covered: _tt (row-major upload + device tilize) and _tt_host /
_tt_refresh (pre-cast on host before tilizing into a persistent trace buffer). Both are
disabled together by emptying rfd3._TORCH_DTYPE, which drops every dtype pre-cast and so
restores the exact pre-p9 behaviour -- no test-only scaffolding in the runtime.

Two processes, not one: the traced decoder skips re-uploading its per-step buffers when the
host tensor identities are unchanged, so comparing two forwards inside one process could
silently compare a buffer against itself.

  # reference, original upload path
  python3 scripts/rfd3_port/verify_device_tilize_parity.py --contig ... --dump-to /tmp/ref.pt --legacy-upload
  # fast path, compares against it
  python3 scripts/rfd3_port/verify_device_tilize_parity.py --contig ... --compare-to /tmp/ref.pt
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
    parser.add_argument("--legacy-upload", action="store_true",
                        help="Disable every host-side dtype pre-cast (the pre-p9 upload path).")
    parser.add_argument("--dump-to", type=Path, help="Write the outputs for a later comparison.")
    parser.add_argument("--compare-to", type=Path, help="Compare against a previous --dump-to.")
    args = parser.parse_args()
    if args.trace_decoder:
        os.environ["RFD3_TRACE_DECODER"] = "1"
        os.environ.setdefault("TT_BIO_TRACE_REGION_SIZE", str(1 << 28))

    from tt_bio import rfd3
    from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification

    if args.legacy_upload:
        rfd3._TORCH_DTYPE = {}

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
    outputs = {}
    for batch in args.batches:
        noisy = torch.randn(batch, length, 3, generator=torch.Generator().manual_seed(42)) * 16.0
        with torch.no_grad():
            out = dm(X_noisy_L=noisy, t=torch.full((batch,), 8.0), f=features, **initial)
        outputs[batch] = {k: out[k].float().cpu() for k in ("X_L", "sequence_logits_I")}

    if args.dump_to:
        args.dump_to.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"atoms": length, "outputs": outputs}, args.dump_to)
        print(f"DUMPED atoms={length} batches={list(outputs)} legacy_upload={args.legacy_upload}")
        return

    if not args.compare_to:
        parser.error("pass --dump-to or --compare-to")
    reference = torch.load(args.compare_to, weights_only=False)
    assert reference["atoms"] == length, "reference was captured on a different fixture"
    ok = True
    for batch, tensors in outputs.items():
        for key, got in tensors.items():
            ref = reference["outputs"][batch][key]
            maxabs = (ref - got).abs().max().item()
            finite = bool(torch.isfinite(got).all())
            ok &= maxabs == 0.0 and finite
            print(f"TILIZE_PARITY batch={batch} atoms={length} {key} "
                  f"maxabs={maxabs:.6e} finite={finite}", flush=True)
    print("DEVICE TILIZE PARITY " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
