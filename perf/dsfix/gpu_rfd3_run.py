"""One RFD3 design invocation, with the atom-attention paths counted.

Run as a subprocess so counters and the model share a fresh interpreter:

    python perf/dsfix/gpu_rfd3_run.py --counts out/counts.json -- \
        design out_dir=... inputs=... diffusion_batch_size=8

Everything after `--` goes to the `rfd3` CLI verbatim.

Counters, not flags. RFD3 picks its atom-attention path at runtime from tensor shapes and free
VRAM (`use_dense_sdpa_pairbias` in rfd3/model/layers/attention.py), so setting an env var proves
nothing about what ran. These wrappers count real calls:

  dense_sdpa_pairbias_attention   the SDPA fast path, upstream production HEAD only
  sparse_pairbias_attention       the original gather path, and the only path in 0.2.0
  _fused_full_pairbias_attention  the cuEquivariance pair-bias kernel
  F.scaled_dot_product_attention  what the dense path defers to

The cueq counter reads 0 on every arm. `use_kernel` is a hard-coded False literal inside
`PairBiasAttention.forward`, so that branch is unreachable whether or not cuequivariance is
installed. Counting it is how that gets proven instead of asserted.
"""

import argparse
import atexit
import json
import pathlib
import sys

COUNTS: dict[str, int] = {}


def _wrap(mod, name: str) -> None:
    """Count calls to mod.name, if it exists. Patches the module attribute, which is what the
    call sites resolve -- they refer to the module global, not to an imported copy."""
    fn = getattr(mod, name, None)
    if fn is None:
        COUNTS[name] = -1  # -1 == not present in this build
        return
    COUNTS[name] = 0

    def wrapper(*a, **kw):
        COUNTS[name] += 1
        return fn(*a, **kw)

    setattr(mod, name, wrapper)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    import torch
    import torch.nn.functional as F

    import foundry
    import rfd3.model.layers.attention as A

    for name in ("dense_sdpa_pairbias_attention", "sparse_pairbias_attention",
                 "_fused_full_pairbias_attention", "kernel_pairbias_attention"):
        _wrap(A, name)

    _sdpa = F.scaled_dot_product_attention
    COUNTS["scaled_dot_product_attention"] = 0

    def sdpa_wrapper(*a, **kw):
        COUNTS["scaled_dot_product_attention"] += 1
        return _sdpa(*a, **kw)

    F.scaled_dot_product_attention = sdpa_wrapper
    torch.nn.functional.scaled_dot_product_attention = sdpa_wrapper

    env = {"should_use_cuequivariance": bool(getattr(foundry, "SHOULD_USE_CUEQUIVARIANCE", False)),
           "torch": torch.__version__,
           "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
           "cudnn_tf32": torch.backends.cudnn.allow_tf32,
           "float32_matmul_precision": torch.get_float32_matmul_precision()}
    for pkg in ("cuequivariance_torch", "cuequivariance_ops_torch", "rc_foundry"):
        try:
            from importlib.metadata import version

            env[pkg] = version(pkg.replace("_", "-"))
        except Exception:
            env[pkg] = None

    def dump() -> None:
        p = pathlib.Path(args.counts)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"counts": COUNTS, "env": env}, indent=2) + "\n")

    atexit.register(dump)

    from rfd3.cli import app

    sys.argv = ["rfd3"] + rest
    app()


if __name__ == "__main__":
    main()
