"""Profile model-specific stages that sit outside the shared Pairformer trunk."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import ttnn
import yaml


def _protenix(args):
    from tt_bio.protenix import Protenix
    from tt_bio.protenix_data import build_protein_features
    from tt_bio.tenstorrent import get_device

    spec = yaml.safe_load(Path(args.input).read_text())
    sequence = spec["sequences"][0]["protein"]["sequence"]
    feats = build_protein_features(sequence)
    device = get_device()
    config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    model = Protenix.load_from_checkpoint(
        args.checkpoint, compute_kernel_config=config, device=device
    )

    confidence_samples = []
    original_confidence = model.confidence_head.confidence

    def timed_confidence(*call_args, **call_kwargs):
        ttnn.synchronize_device(device)
        started = time.perf_counter()
        result = original_confidence(*call_args, **call_kwargs)
        ttnn.synchronize_device(device)
        confidence_samples.append(time.perf_counter() - started)
        return result

    model.confidence_head.confidence = timed_confidence

    def fold():
        confidence_samples.clear()
        ttnn.synchronize_device(device)
        started = time.perf_counter()
        model.fold(
            feats,
            n_step=args.steps,
            n_sample=args.samples,
            seed=0,
            return_confidence=True,
        )
        ttnn.synchronize_device(device)
        return time.perf_counter() - started, list(confidence_samples)

    warm_total, _ = fold()
    total, confidence = fold()
    confidence_total = sum(confidence)
    print(
        json.dumps(
            {
                "model": "protenix-v2",
                "tokens": len(sequence),
                "sampling_steps": args.steps,
                "samples": args.samples,
                "warmup_total_s": warm_total,
                "timed_total_s": total,
                "confidence_samples_s": confidence,
                "confidence_total_s": confidence_total,
                "confidence_share": confidence_total / total,
                "free_confidence_ceiling": total / (total - confidence_total),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("protenix-v2",))
    parser.add_argument("--input", default="examples/prot.yaml")
    parser.add_argument("--checkpoint", default="/home/moritz/.boltz/protenix-v2.pt")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    _protenix(args)


if __name__ == "__main__":
    main()
