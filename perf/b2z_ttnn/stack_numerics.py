#!/usr/bin/env python3
"""Do the two stacks compute the same thing? One fixed input set, one output file per stack.

The 512 aa fold folds the same target at plDDT 0.846 on 0.68.0 and 0.412 on 0.78.0. That is either
a real arithmetic difference or a changed default somewhere above the ops. This isolates the first
possibility: identical input bytes (seeded once on the host and read from .npy by both stacks),
the same ttnn calls, outputs written to .npy, compared off-device.

The ops are the ones a Boltz-2 pairformer block actually spends its time in, at tile-aligned
production-ish shapes, at the fidelities tt-bio asks for rather than the library default -- a
changed DEFAULT fidelity is one of the hypotheses, so the probe records both the explicit-config
result and the bare-default result for the same matmul.

    <venv>/bin/python stack_numerics.py --out old.npz        # on each stack
    python stack_numerics.py --compare old.npz new.npz
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import sys
from pathlib import Path

import numpy as np

SEED = 0
INPUTS = Path("/tmp/b2z_numerics_inputs.npz")


def make_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    def r(*shape):
        return rng.standard_normal(shape, dtype=np.float32) * 0.5
    return {
        "a": r(1, 1, 512, 128),      # pair-track activation, tile aligned
        "b": r(1, 1, 128, 128),      # projection weight
        "big_a": r(1, 4, 512, 512),  # attention-scale operand
        "big_b": r(1, 4, 512, 128),
        "ln_w": r(128).astype(np.float32),
        "ln_b": r(128).astype(np.float32),
    }


def run(out: Path) -> int:
    import torch
    import ttnn
    torch.set_grad_enabled(False)
    if not INPUTS.exists():
        np.savez(INPUTS, **make_inputs())
    src = dict(np.load(INPUTS))
    dev = ttnn.open_device(device_id=0)
    res: dict[str, np.ndarray] = {}
    try:
        def _np(t_):
            # bf16 -> fp32 is lossless, so widening here cannot hide a difference between stacks,
            # and numpy has no bf16 to compare in.
            return ttnn.to_torch(t_).float()

        def to_dev(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(torch.from_numpy(x), dtype=dtype,
                                   layout=ttnn.TILE_LAYOUT, device=dev)

        a, b = to_dev(src["a"]), to_dev(src["b"])
        ba, bb = to_dev(src["big_a"]), to_dev(src["big_b"])

        # 1. matmul at the library default -- the changed-default hypothesis lands here
        res["mm_default"] = _np(ttnn.matmul(a, b))
        # 2. the same matmul with tt-bio's explicit HiFi4 config
        cfg = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)
        res["mm_hifi4"] = _np(
            ttnn.matmul(a, b, compute_kernel_config=cfg))
        # 3. a big matmul, where fidelity loss compounds over K
        res["mm_big"] = _np(ttnn.matmul(ba, bb))
        # 4. softmax on the last axis -- the fold's numerics are sensitive to it
        res["softmax"] = _np(ttnn.softmax(ba, dim=-1))
        # 5. layer_norm, which every block runs dozens of times
        res["layer_norm"] = _np(ttnn.layer_norm(
            a, weight=to_dev(src["ln_w"]), bias=to_dev(src["ln_b"]), epsilon=1e-5))
        # 6. a bf16 round trip, the floor: any difference here is not arithmetic at all
        res["roundtrip"] = _np(a)
    finally:
        ttnn.close_device(dev)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ttnn_version=np.array(md.version("ttnn")), **res)
    for k, v in res.items():
        print(f"{k:12s} shape={v.shape} mean={v.mean():+.6f} std={v.std():.6f}")
    return 0


def pcc(x: np.ndarray, y: np.ndarray) -> float:
    a, b = x.ravel().astype(np.float64), y.ravel().astype(np.float64)
    if np.allclose(a, b):
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def compare(pa: Path, pb: Path) -> int:
    A, B = np.load(pa), np.load(pb)
    print(f"{str(A['ttnn_version'])} vs {str(B['ttnn_version'])}")
    print(f"{'op':12s} {'PCC':>10s} {'max|diff|':>12s} {'identical':>10s}")
    for k in A.files:
        if k == "ttnn_version":
            continue
        x, y = A[k], B[k]
        same = bool(np.array_equal(x, y))
        print(f"{k:12s} {pcc(x, y):10.7f} {np.abs(x - y).max():12.6g} {str(same):>10s}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", type=Path, nargs=2)
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.out:
        ap.error("--out or --compare")
    return run(args.out)


if __name__ == "__main__":
    sys.exit(main())
