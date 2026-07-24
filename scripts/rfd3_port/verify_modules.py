"""Verify RFD3 atom encoder, decoder, and sequence-head ttnn ports.

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_modules.py [capture_dir]
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3 import build_atom_encoder, build_decoder, build_sequence_head


def load(capture_dir, name):
    return torch.load(
        os.path.join(capture_dir, name + ".pt"),
        map_location="cpu",
        weights_only=True,
    )


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-12))


def report(name, actual, expected, threshold=0.99):
    value = pcc(actual, expected)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    print(
        f"{name:24s} PCC={value:.6f}  max_abs={max_abs:.4e}  "
        f"shape={tuple(actual.shape)}"
    )
    if value < threshold:
        raise AssertionError(f"{name} PCC {value:.6f} < {threshold}")
    return value


def scoped(weights, prefix):
    prefix = prefix + "."
    return {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}


def main(capture_dir):
    weights = load(capture_dir, "diffusion_module.real_weights")

    encoder = build_atom_encoder(scoped(weights, "encoder"))
    encoder_out = encoder(
        load(capture_dir, "encoder.in_Q_L"),
        load(capture_dir, "encoder.in_C_L"),
        load(capture_dir, "encoder.in_P_LL"),
        load(capture_dir, "encoder.in_indices"),
    )
    report("encoder.Q_L", encoder_out, load(capture_dir, "encoder.out_Q_L"))

    decoder = build_decoder(scoped(weights, "decoder"))
    decoder_a, decoder_q = decoder(
        load(capture_dir, "decoder.in_A_I"),
        load(capture_dir, "decoder.in_S_I"),
        load(capture_dir, "decoder.in_Q_L"),
        load(capture_dir, "decoder.in_C_L"),
        load(capture_dir, "decoder.in_P_LL"),
        load(capture_dir, "decoder.in_tok_idx").long(),
        load(capture_dir, "decoder.in_indices"),
    )
    report("decoder.A_I", decoder_a, load(capture_dir, "decoder.out_A_I"))
    report("decoder.Q_L", decoder_q, load(capture_dir, "decoder.out_Q_L"))

    sequence_head = build_sequence_head(scoped(weights, "sequence_head"))
    logits, indices = sequence_head(load(capture_dir, "sequence_head.in_A_I"))
    golden_pcc = report(
        "sequence_head.logits",
        logits,
        load(capture_dir, "sequence_head.out_logits"),
        threshold=0.98,
    )
    sequence_input = load(capture_dir, "sequence_head.in_A_I")
    host_logits = torch.nn.functional.linear(
        sequence_input,
        weights["sequence_head.linear.weight"].bfloat16(),
        weights["sequence_head.linear.bias"].bfloat16(),
    )
    report("sequence_head.host", logits, host_logits, threshold=0.999)
    if golden_pcc < pcc(host_logits, load(capture_dir, "sequence_head.out_logits")) - 1e-3:
        raise AssertionError("device sequence-head PCC falls below the CPU reference floor")
    expected_indices = load(capture_dir, "sequence_head.out_indices")
    valid = weights["sequence_head.valid_out_mask"].bool().view(1, 1, -1)
    host_indices = host_logits.float().masked_fill(~valid, float("-inf")).argmax(dim=-1)
    host_exact = float((indices == host_indices).float().mean())
    golden_exact = float((indices == expected_indices).float().mean())
    print(
        f"{'sequence_head.indices':24s} host_exact={host_exact:.6f}  "
        f"golden_exact={golden_exact:.6f}  shape={tuple(indices.shape)}"
    )
    if host_exact != 1.0:
        raise AssertionError(f"sequence-head host index exactness {host_exact:.6f} != 1.0")


if __name__ == "__main__":
    default = os.path.join(
        os.path.dirname(__file__), "..", "..", ".scratch",
        "rfd3-ref", "goldens", "capture",
    )
    main(sys.argv[1] if len(sys.argv) > 1 else default)
