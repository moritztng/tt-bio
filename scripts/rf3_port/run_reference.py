#!/usr/bin/env python3
"""Run the RF3 torch reference end to end on one input.

Featurize with ``tt_bio.rf3.featurize``, load the reference with
``tt_bio.rf3.weights``, and run the trunk + diffusion rollout. This is the harness
the ttnn port is scored against: dump the intermediates you care about here, run
the ttnn equivalent on the same `f`, compare.

Everything is CPU and slow by design. Keep n_recycles / num_steps small unless you
mean it: the trunk is 48 Pairformer blocks and the sampler defaults to 200 steps.

    python scripts/rf3_port/run_reference.py \\
        --input scripts/rf3_port/parity_artifacts/glke/input.json \\
        --ckpt /path/to/rf3_latest.ckpt --n_recycles 1 --num_steps 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_recycles", type=int, default=1)
    ap.add_argument("--num_steps", type=int, default=8)
    ap.add_argument("--diffusion_batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--fp32", action="store_true",
                    help="disable bf16 autocast (upstream inference runs with it on)")
    ap.add_argument("--dump", help="write outputs to this .pt")
    args = ap.parse_args()

    from tt_bio.rf3.featurize import featurize, n_cycle, network_input
    from tt_bio.rf3.weights import load_reference

    t0 = time.time()
    inp = Path(args.input).resolve()
    prev = os.getcwd()
    os.chdir(inp.parent)  # fixture inputs use paths relative to their own directory
    try:
        out = featurize(
            inp.name,
            n_recycles=args.n_recycles,
            diffusion_batch_size=args.diffusion_batch_size,
            seed=args.seed,
        )[0]
    finally:
        os.chdir(prev)
    t_feat = time.time() - t0
    f = out["feats"]
    print(f"featurized in {t_feat:.1f}s: {f['restype'].shape[0]} tokens, "
          f"{f['ref_pos'].shape[0]} atoms")

    t0 = time.time()
    net, _ = load_reference(args.ckpt, use_ema=not args.no_ema, num_steps=args.num_steps)
    print(f"loaded reference in {time.time() - t0:.1f}s "
          f"({'EMA' if not args.no_ema else 'raw'} weights)")

    # Upstream runs inference under bfloat16 AMP (Lightning sets the precision, and
    # RF3.forward casts msa_stack / profile / template_* when autocast is enabled).
    # Without it the first Linear hits BFloat16 weights against float activations.
    t0 = time.time()
    autocast = torch.autocast("cpu", dtype=torch.bfloat16, enabled=not args.fp32)
    with torch.no_grad(), autocast:
        res = net(
            input=network_input(out),
            n_cycle=min(args.n_recycles, n_cycle(out)),
            coord_atom_lvl_to_be_noised=out["coord_atom_lvl_to_be_noised"],
        )
    t_fwd = time.time() - t0

    print(f"forward in {t_fwd:.1f}s; outputs:")
    for k in sorted(res):
        v = res[k]
        if isinstance(v, torch.Tensor):
            extra = ""
            if v.is_floating_point() and v.numel():
                extra = f"  span {float(v.min()):.3f} .. {float(v.max()):.3f}"
            print(f"  {k:<24} {str(list(v.shape)):<22} {str(v.dtype):<16}{extra}")
        elif isinstance(v, (list, tuple)):
            print(f"  {k:<24} {type(v).__name__} of {len(v)}")
        else:
            print(f"  {k:<24} {type(v).__name__}")

    coords = next((res[k] for k in ("X_L", "X_pred_rollout_L", "X_pred", "coords") if
                   isinstance(res.get(k), torch.Tensor)), None)
    if coords is None:
        raise SystemExit("no coordinate output found; keys: %s" % sorted(res))

    if args.dump:
        torch.save({k: v for k, v in res.items() if isinstance(v, torch.Tensor)},
                   args.dump)
        print(f"dumped -> {args.dump}")
    print(json.dumps({"tokens": int(f["restype"].shape[0]),
                      "coords_shape": list(coords.shape),
                      "atoms": int(f["ref_pos"].shape[0]),
                      "featurize_s": round(t_feat, 2),
                      "forward_s": round(t_fwd, 2)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
