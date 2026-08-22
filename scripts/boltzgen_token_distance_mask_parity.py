#!/usr/bin/env python3
"""Do BoltzGen's two token-distance paths hand their pairformer the same mask?

`TokenDistanceModule.forward` (the host recycle loop) passes the trunk's `pair_mask`.
`tenstorrent.TokenDistanceRecycle` is the device-resident copy of the same stage and is
the BoltzGen default, and it used to build its own mask from the tile padding alone --
and build it 1-D, which reaches only the outgoing TriangleMultiplication
(VERDICT-TRIMULCURE in state/opendde-pairformer-z-parity-drop.md).

Both arms are handed the same `v`, so the only thing that can differ is the mask. Weights
come from the shipped design checkpoint; the features are synthetic because the token mask
is the only feature the mask depends on.

  PYTHONPATH=<worktree> TT_VISIBLE_DEVICES=0 python3 scripts/boltzgen_token_distance_mask_parity.py
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch

torch.set_grad_enabled(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path.home() / ".boltz/boltzgen/boltzgen1_diverse.ckpt")
    ap.add_argument("--seq-len", type=int, default=100,
                    help="token count; not a multiple of 64 adds tile padding on top")
    ap.add_argument("--masked", type=int, default=0,
                    help="how many of the seq_len tokens carry token_pad_mask=0")
    ap.add_argument("--seed", type=int, default=893)
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent
    from tt_bio.boltzgen.adapter import _legacy_pickle_compat
    from tt_bio.boltzgen.model.modules.trunk import TokenDistanceModule

    with _legacy_pickle_compat():
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    hp = ckpt["hyper_parameters"]
    sd = {k[len("token_distance_module."):]: v for k, v in ckpt["state_dict"].items()
          if k.startswith("token_distance_module.")}
    assert sd, "no token_distance_module weights in the checkpoint"

    torch.manual_seed(args.seed)
    mod = TokenDistanceModule(hp["token_z"], **hp["token_distance_args"]).eval()
    missing, unexpected = mod.load_state_dict(sd, strict=False)
    print(f"[weights] missing={len(missing)} unexpected={len(unexpected)}")

    S = args.seq_len
    token_mask = torch.ones(1, S)
    if args.masked:
        token_mask[0, torch.randperm(S)[: args.masked]] = 0.0
    pair_mask = token_mask[:, :, None] * token_mask[:, None, :]
    n_atoms = 4 * S
    feats = {
        "token_pad_mask": token_mask,
        "center_coords": 20 * torch.randn(1, S, 3),
        "token_distance_mask": pair_mask.clone(),
        "token_to_bb4_atoms": torch.eye(n_atoms).unsqueeze(0),
        "coords": 20 * torch.randn(1, n_atoms, 3),
    }
    rpe = torch.randn(1, S, S, hp["token_z"])
    z = 4 * torch.randn(1, S, S, hp["token_z"])

    device = tenstorrent.get_device()
    pad = (-S) % tenstorrent.PAIRFORMER_PAD_MULTIPLE
    fpad = torch.nn.functional.pad

    # the pairformer input, built once on host and given to both arms unchanged
    rec = tenstorrent.TokenDistanceRecycle(mod, mod.pairformer.compute_kernel_config)
    # one instrument, both arms: before this fix `precompute` built its own masks and took
    # (feats, rpe, seq_len, seq_pad); it now takes the trunk's masks instead.
    takes_masks = "mask_tt" in inspect.signature(rec.precompute).parameters

    def precompute(mask_tt, attn_tt):
        return (rec.precompute(feats, rpe, pad, mask_tt, attn_tt) if takes_masks
                else rec.precompute(feats, rpe, S, pad))

    td = precompute(None, None)   # a_ij only; masks taken from the second call below
    a_ij = torch.Tensor(ttnn.to_torch(td["a_ij_tt"])).to(torch.float32)
    v = fpad(mod.z_proj(mod.z_norm(z)), (0, 0, 0, pad, 0, pad)) + a_ij

    # arm 1: what the reference recipe asks for -- the trunk's pair mask
    out_ref = mod.pairformer(v[:, :S, :S], pair_mask)

    # arm 2: whatever masks TokenDistanceRecycle would have used
    mask_1d_p = fpad(token_mask, (0, pad)) if pad else token_mask
    pair_mask_p = fpad(pair_mask, (0, pad, 0, pad)) if pad else pair_mask

    def up(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    trunk_mask_tt = up(pair_mask_p)
    trunk_attn_tt = up((1 - mask_1d_p).unsqueeze(1).unsqueeze(1) * -1e9)
    td = precompute(trunk_mask_tt, trunk_attn_tt)
    _, out_tt = rec.pairformer(None, up(v), td["mask_tt"], td["attn_tt"], td["attn_tt"])
    out_res = torch.Tensor(ttnn.to_torch(out_tt)).to(torch.float32)[:, :S, :S]

    keep = token_mask[0].bool()
    for label, a, b in (
        ("whole tensor", out_res, out_ref),
        ("unmasked block", out_res[:, keep][:, :, keep], out_ref[:, keep][:, :, keep]),
    ):
        rel = ((a - b).norm() / b.norm()).item()
        pcc = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
        print(f"[{label}] rel Frobenius {rel:.4e}   1-PCC {1 - pcc:.4e}")


if __name__ == "__main__":
    main()
