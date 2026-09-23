#!/usr/bin/env python3
"""of3t-fullstep64: the bijection from our walked device tensors to upstream's parameters, whole model.

    bijmap.py --weights weights_walked.pt --checkpoint of3-p2-155k.pt --out BIJECTION.json

The campaign's matcher, `perf/of3t_gradients/bijection_device.py`, run over every tensor the full
step's walk reaches, plus `perf/of3t_trunkg043/dev_grad.py`'s two derived rules for the fused
AttentionPairBias leaves. Nothing is transcribed: every placement is confirmed by value.

SCOPE. Matching is per block wherever both sides have blocks, so an all-zero or repeated tensor
cannot pair across blocks. Everything outside a block stack is matched against the upstream
tensors outside every block stack.

AMBIGUITY. A device tensor that whole-matches more than one upstream tensor (identical values,
e.g. two untouched LayerNorm gains) cannot say whose gradient it carries; its upstream tensors are
dropped to `ambiguous` and counted as unread, never guessed.

DERIVED. `qkv_weight`/`qkv_bias` (q, k, v fused, head dim padded, transposed; the diffusion
transformer's `qkv_w`/`qkv_b` are the same layout) and `z_weight`
(linear_z transposed and scaled by the card) are placed by `dev_grad.apb_inverse` and by
transpose-and-unscale, each ACCEPTED only if rebuilding the weight from the checkpoint through the
same rule is bit-identical to what the card holds. n_heads, padded head dim and the applied scale
are searched, not assumed, and the search result is the verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[0:0] = [str(REPO / "perf" / "of3t_gradients"), str(REPO / "perf" / "of3t_trunkg043")]
from bijection_device import device_bijection  # noqa: E402

STACKS = [("trunk.pairformer.blocks.", "pairformer_stack.blocks."),
          ("trunk.msa_module.blocks.", "msa_module.blocks."),
          ("trunk.template.ps.blocks.", "template_embedder.template_pair_stack.blocks."),
          ("confidence_head.pf.blocks.", "aux_heads.pairformer_embedding.pairformer_stack.blocks."),
          ("sampler.dm.dit.blocks.", "diffusion_module.diffusion_transformer.blocks.")]
APB_DEV = "attention_pair_bias."
APB_UP = "attn_pair_bias."


def apb_inverse(dev_t, leaf, H, d, D, c_s):
    """`dev_grad.apb_inverse`, restated because importing dev_grad pulls ttnn at module scope."""
    if leaf.endswith("linear_q.bias"):
        return dev_t[:H * D].reshape(H, D)[:, :d].reshape(H * d).contiguous()
    x = dev_t.t().reshape(3 * H, D, c_s)
    off = {"linear_q.weight": 0, "linear_k.weight": H, "linear_v.weight": 2 * H}[leaf]
    return x[off:off + H, :d, :].reshape(H * d, c_s).contiguous()


def bf16(t):
    return t.to(torch.bfloat16).to(torch.float32)


def same(dev_t, w):
    """The card holds `w` as bf16 (trunk) or as fp32 (the fp32 diffusion module)."""
    return torch.equal(dev_t, bf16(w)) or torch.equal(dev_t, w)


def trunc_bf16_scalar(x: float) -> float:
    b = torch.tensor([x], dtype=torch.float32).view(torch.int32) & ~0xFFFF
    return float(b.view(torch.float32))


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.load(a.weights, weights_only=False)
    sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    atoms = {(k[6:] if k.startswith("model.") else k): v.detach().to(torch.float32)
             for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()}

    scopes = []                       # (device prefix, upstream prefix)
    for dp, up in STACKS:
        js = sorted({int(k[len(dp):].split(".")[0]) for k in dev if k.startswith(dp)})
        scopes += [(f"{dp}{j}.", f"{up}{j}.") for j in js]
    in_block_dev = lambda k: any(k.startswith(dp) for dp, _ in STACKS)  # noqa: E731
    in_block_up = lambda k: any(k.startswith(up) for _, up in STACKS)  # noqa: E731
    scopes.append(("", ""))

    placements, device_serves, device_unmatched = {}, {}, []
    for dp, up in scopes:
        if dp:
            d_s = {k[len(dp):]: v for k, v in dev.items() if k.startswith(dp)}
            a_s = {k[len(up):]: v for k, v in atoms.items() if k.startswith(up)}
        else:
            d_s = {k: v for k, v in dev.items() if not in_block_dev(k)}
            a_s = {k: v for k, v in atoms.items() if not in_block_up(k)}
        bj = device_bijection(d_s, a_s)
        for k, pls in bj["placements"].items():
            for pl in pls:
                pl = dict(pl, device_path=dp + pl["device_path"], rule="by_value")
                placements.setdefault(up + k, []).append(pl)
                device_serves.setdefault(pl["device_path"], set()).add(up + k)
        device_unmatched += [dp + x["device_path"] for x in bj["device_unmatched"]]

    # A device tensor serving several upstream tensors WHOLE is ambiguous.
    ambiguous = set()
    for path, keys in device_serves.items():
        wholes = [k for k in keys
                  if any(p["device_path"] == path and p["start"] == 0
                         and p["length"] == int(dev[path].shape[p["axis"]]) for p in placements[k])]
        if len(wholes) > 1:
            ambiguous.update(wholes)
    for k in ambiguous:
        placements.pop(k, None)

    # Derived AttentionPairBias leaves: trunk and confidence Pairformers, diffusion transformer.
    derived, derived_fail = {}, []
    for dp, up in scopes:
        if dp.startswith("trunk.pairformer") or dp.startswith("confidence_head.pf"):
            qk, qb = dp + APB_DEV + "qkv_weight", dp + APB_DEV + "qkv_bias"
            zk, upa = dp + APB_DEV + "z_weight", up + APB_UP
        elif dp.startswith("sampler.dm.dit"):
            qk, qb, zk, upa = dp + "qkv_w", dp + "qkv_b", None, up + APB_DEV
        else:
            continue
        wq = atoms.get(upa + "mha.linear_q.weight")
        if qk in dev and wq is not None:
            c_s = int(wq.shape[1])
            found = None
            for H in (16, 8, 4, 12, 24, 32):
                if wq.shape[0] % H or dev[qk].shape[1] % (3 * H):
                    continue
                d, D = wq.shape[0] // H, dev[qk].shape[1] // (3 * H)
                if D < d:
                    continue
                if all(same(apb_inverse(dev[qk], f"linear_{x}.weight", H, d, D, c_s),
                            atoms[upa + f"mha.linear_{x}.weight"]) for x in "qkv"):
                    found = (H, d, D)
                    break
            if found is None:
                derived_fail.append(qk)
            else:
                H, d, D = found
                for x in "qkv":
                    key = upa + f"mha.linear_{x}.weight"
                    if key not in placements:
                        derived[key] = {"device_path": qk, "rule": "apb_inverse",
                                        "leaf": f"linear_{x}.weight", "H": H, "d": d, "D": D,
                                        "c_s": c_s}
                bkey = upa + "mha.linear_q.bias"
                if qb in dev and bkey in atoms and bkey not in placements:
                    if same(apb_inverse(dev[qb], "linear_q.bias", H, d, D, c_s), atoms[bkey]):
                        derived[bkey] = {"device_path": qb, "rule": "apb_inverse",
                                         "leaf": "linear_q.bias", "H": H, "d": d, "D": D,
                                         "c_s": c_s}
                    else:
                        derived_fail.append(qb)
        zkey = upa + "linear_z.weight"
        if zk in dev and zkey in atoms and zkey not in placements:
            w = atoms[zkey]
            H = int(w.shape[0])
            d = int(atoms[upa + "mha.linear_q.weight"].shape[0]) // H
            ok = None
            for s in (trunc_bf16_scalar(math.sqrt(d)), float(bf16(torch.tensor(math.sqrt(d)))),
                      math.sqrt(d), trunc_bf16_scalar(1 / math.sqrt(d)), 1 / math.sqrt(d), 1.0):
                if torch.equal(dev[zk], bf16(bf16(w).t().contiguous() * s)):
                    ok = s
                    break
            if ok is None:
                derived_fail.append(zk)
            else:
                derived[zkey] = {"device_path": zk, "rule": "transpose_and_unscale", "scale": ok}

    placed = set(placements) | set(derived)
    rep = {"weights": {"file": a.weights, "sha256": sha256_file(a.weights), "n": len(dev)},
           "checkpoint": {"file": a.checkpoint, "sha256": sha256_file(a.checkpoint),
                          "n": len(atoms)},
           "n_scopes": len(scopes), "n_by_value": len(placements), "n_derived": len(derived),
           "n_placed": len(placed), "n_ambiguous": len(ambiguous),
           "ambiguous": sorted(ambiguous), "derived_fail": derived_fail,
           "device_unmatched": sorted(device_unmatched),
           "upstream_unplaced": sorted(set(atoms) - placed - ambiguous),
           "placements": placements, "derived": derived}
    Path(a.out).write_text(json.dumps(rep, indent=0, default=str) + "\n")
    print(f"placed {len(placed)} of {len(atoms)} upstream tensors ({len(placements)} by value, "
          f"{len(derived)} derived), {len(ambiguous)} ambiguous, {len(derived_fail)} derived "
          f"failures, {len(device_unmatched)} device tensors unmatched", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
