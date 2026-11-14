"""Capture the autoencoder encoder's real input batch from upstream's own generate path.

Partial diffusion encodes the input structure into local latents before sampling. A pre-hook on
EncoderTransformer saves that batch (every tensor it carries) and stops the run, so the CPU never
pays for the sampler. Runs in the upstream reference env:

    python -I perf/rfd4/vae/capture_real.py --cif <input.cif[.zst]> --out <batch.pt>
"""

import argparse
import sys
import tempfile
from pathlib import Path

import torch


class Captured(Exception):
    pass


def force_cpu_trainer() -> None:
    """`run_generation` hardcodes accelerator="gpu"; run its Trainer on CPU fp32 (as rfd4-ref's capture.py)."""
    import lightning as L

    base = L.Trainer

    class CPUTrainer(base):
        def __init__(self, *args, **kwargs):
            kwargs.update(accelerator="cpu", devices=1, num_nodes=1, precision="32-true")
            super().__init__(*args, **kwargs)

    L.Trainer = CPUTrainer


def _caused_by(exc, cls) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, cls):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="nvidia/RFD4-Proteina")
    a = ap.parse_args()
    import rfd4_proteina as rfd4

    force_cpu_trainer()

    model = rfd4.Proteina.from_pretrained(a.ckpt)
    enc = [m for m in model.modules() if type(m).__name__ == "EncoderTransformer"]
    assert len(enc) == 1, len(enc)

    def hook(mod, args):
        batch = args[0]
        d = {k: v.detach().clone() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        torch.save({"cif": a.cif, "batch": d, "upstream_sha": "2db07bb444d91695f8c10e6acb7648e9297f9221"}, a.out)
        print("captured", {k: tuple(v.shape) for k, v in d.items()}, flush=True)
        raise Captured()

    enc[0].register_forward_pre_hook(hook)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            rfd4.generate(model, overrides=[f"inputs={a.cif}", "generation.args.partial_t_backbone=0.8",
                                            "generation.args.partial_t_latent=0.8", "num_replicates=1",
                                            f"out_dir={tmp}"])
        except Exception as e:  # Lightning may wrap the hook's exception
            if _caused_by(e, Captured):
                return 0
            raise
    print("encoder never ran", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
