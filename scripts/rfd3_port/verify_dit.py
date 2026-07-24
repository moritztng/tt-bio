"""Verify the RFD3 18-block token DiT (LocalTokenTransformer) ttnn port against a
vendored torch reference (scripts/rfd3_port/rfd3_ref.py DiTBlockRef).

Real checkpoint weights are loaded into BOTH the ttnn port (RFD3AtomBlock at
c_token=768, c_s=384, c_tokenpair=128, n_head=16, head_dim=48) and the vendored torch
reference. Identical (shared) random activations A_I [1,144,768], S_I [1,144,384],
Z_II [1,144,144,128] + a shared sparse indices tensor [1,144,128] (-> dense additive
mask, the path proven numerically equivalent to upstream gather-sparse in state 2b.5)
are fed to both. PCC > 0.99 on every block proves the DiT port is wired correctly.
This is the shared-activations wiring gate (no vast.ai golden needed for the per-block
wiring check); the multi-step trajectory parity (shared RNG draws) is the next leg.

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_dit.py [capture_dir] [n_blocks]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3 import build_dit

sys.path.insert(0, os.path.dirname(__file__))
from rfd3_ref import build_dit_block_ref, _indices_to_mask


def load(capture_dir, name):
    return torch.load(os.path.join(capture_dir, name + ".pt"), map_location="cpu", weights_only=True)


def scoped(weights, prefix):
    prefix = prefix + "."
    return {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-12))


def build_shared_inputs(I=144, n_keys=128, seed=42):
    """Shared random DiT-block inputs + sparse indices (self + n_keys-1 random
    neighbours per query) -> dense additive mask. Every row includes self so no row is
    fully masked (avoids softmax NaN)."""
    g = torch.Generator().manual_seed(seed)
    A_I = (torch.randn(1, I, 768, generator=g) * 0.1).bfloat16()
    S_I = (torch.randn(1, I, 384, generator=g) * 0.1).bfloat16()
    Z_II = (torch.randn(1, I, I, 128, generator=g) * 0.1).bfloat16()
    idx = torch.zeros(1, I, n_keys, dtype=torch.long)
    for i in range(I):
        others = torch.randperm(I, generator=g)
        others = others[others != i][: n_keys - 1]
        idx[0, i] = torch.cat([torch.tensor([i]), others])
    return A_I, S_I, Z_II, idx


def main(capture_dir, n_blocks):
    weights = load(capture_dir, "diffusion_module.real_weights")
    dit_weights = scoped(weights, "diffusion_transformer")

    A_I, S_I, Z_II, indices = build_shared_inputs()
    valid_mask = _indices_to_mask(indices)  # [1, I, I] bool — same pattern for both backends

    # --- vendored torch reference, block by block, fp32, shared inputs + shared mask ---
    ref_sd = scoped(dit_weights, "blocks")  # keys "{i}.attention_pair_bias..."
    ref_outputs = []
    a_ref = A_I.clone().float()
    s_ref = S_I.clone().float()
    z_ref = Z_II.clone().float()
    with torch.no_grad():
        for i in range(n_blocks):
            block_sd = scoped(ref_sd, str(i))  # "attention_pair_bias..."
            ref_block = build_dit_block_ref().eval()
            missing, unexpected = ref_block.load_state_dict(block_sd, strict=False)
            if missing or unexpected:
                print(f"[ref block {i}] missing={list(missing)} unexpected={list(unexpected)}")
            a_ref = ref_block(a_ref, s_ref, z_ref, valid_mask=valid_mask)
            ref_outputs.append(a_ref)
    print(f"[ref] ran {n_blocks} DiT blocks (fp32, dense+mask)")

    # --- ttnn port: full 18-block stack on device, shared inputs ---
    dit = build_dit(dit_weights)
    # LocalTokenTransformer.__call__ builds the dense additive mask from indices and
    # runs all blocks. To get per-block outputs for parity, run block-by-block here.
    from tt_bio.rfd3 import _dense_attention_mask, _tt  # noqa: F401
    import ttnn
    dev = dit.device
    dt = dit.dtype
    a = _tt(A_I, dev, dt)
    s = _tt(S_I, dev, dt)
    z = _tt(Z_II, dev, dt)
    mask = _tt(_dense_attention_mask(indices), dev, dt)
    tt_outputs = []
    for i, block in enumerate(dit.blocks):
        a = block(a, s, z, mask)
        tt_outputs.append(ttnn.to_torch(a).float())
    print(f"[ttnn] ran {n_blocks} DiT blocks on device (bf16, dense+additive-mask)")

    # --- compare per-block + final ---
    worst = 1.0
    for i in range(n_blocks):
        v = pcc(tt_outputs[i], ref_outputs[i])
        worst = min(worst, v)
        flag = "OK" if v >= 0.99 else "LOW"
        if i < 3 or i >= n_blocks - 2 or v < 0.99:
            print(f"  block {i:2d}: PCC={v:.6f}  {flag}")
    print(f"  final (block {n_blocks-1}): PCC={pcc(tt_outputs[-1], ref_outputs[-1]):.6f}")
    print(f"  worst-block PCC = {worst:.6f}")
    if worst < 0.99:
        raise AssertionError(f"DiT worst-block PCC {worst:.6f} < 0.99")
    print("DiT port PARITY OK (all blocks PCC >= 0.99 vs vendored torch reference)")


if __name__ == "__main__":
    default = os.path.join(os.path.dirname(__file__), "..", "..", ".scratch", "rfd3-ref", "goldens", "capture")
    cap = sys.argv[1] if len(sys.argv) > 1 else default
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    main(cap, n)
