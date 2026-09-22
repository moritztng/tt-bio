#!/usr/bin/env python3
"""Does `checkpoint_blocks` change the gradient? The break control for R141.

R141 named the last structural difference between `grads_f64_043.pt`'s backward and every
`ref_grad.py` replay: the reference runs the pairformer stack through
`openfold3.core.utils.checkpointing.checkpoint_blocks` with `blocks_per_ckpt=1` and
`use_reentrant=None` (the `train` preset), while the replay runs a bare Python loop.
`of3t-frameself` has since eliminated the tree and the per-block call, so this is the only one
left.

A located difference that does not MOVE the reading is not the cause. This moves it or it does
not, on a chain small enough to run in seconds, using UPSTREAM'S OWN function rather than a
reimplementation of it -- a reimplementation would test my model of checkpointing, which is
exactly the thing in question.

The chain deliberately carries the features that make checkpointing non-trivial: two tensors
threaded block to block, non-differentiable mask kwargs bound into a `partial` rather than
passed positionally (which is how `_prep_blocks` does it), LayerNorm so the backward is not
merely linear, and float64 so the only differences that can appear are structural.

  ckpt_break.py --tree <dir containing openfold3/> --blocks 8 --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
from functools import partial
from pathlib import Path

import torch
from torch import nn


class Blk(nn.Module):
    """Two tensors in, two out, masks as non-differentiable kwargs. LayerNorm on both."""

    def __init__(self, c_s: int, c_z: int):
        super().__init__()
        self.ln_s, self.ln_z = nn.LayerNorm(c_s), nn.LayerNorm(c_z)
        self.lin_s, self.lin_z = nn.Linear(c_s, c_s), nn.Linear(c_z, c_z)
        self.mix = nn.Linear(c_s, c_z)

    def forward(self, s, z, single_mask=None, pair_mask=None, _mask_trans=True):
        s = s + self.lin_s(self.ln_s(s)) * single_mask.unsqueeze(-1)
        z = z + self.lin_z(self.ln_z(z)) * pair_mask.unsqueeze(-1)
        z = z + self.mix(s).unsqueeze(1) * pair_mask.unsqueeze(-1)
        return s, z


def build(n, c_s, c_z, seed):
    torch.manual_seed(seed)
    return [Blk(c_s, c_z).to(torch.float64).eval() for _ in range(n)]


def run(mods, s0, z0, sm, pm, cot_s, cot_z, mode, ckpt_blocks, blocks_per_ckpt, use_reentrant):
    for m in mods:
        for p in m.parameters():
            p.grad = None
    s_in = s0.detach().clone().requires_grad_(True)
    z_in = z0.detach().clone().requires_grad_(True)

    if mode == "bare":
        s, z = s_in, z_in
        for m in mods:
            s, z = m(s, z, sm, pm)
    else:
        # exactly `_prep_blocks`: masks bound as kwargs into a partial, not passed positionally
        blocks = [partial(b, single_mask=sm, pair_mask=pm, _mask_trans=True) for b in mods]
        s, z = ckpt_blocks(blocks, args=(s_in, z_in), blocks_per_ckpt=blocks_per_ckpt,
                           use_reentrant=use_reentrant)

    loss = (s * cot_s).sum() + (z * cot_z).sum()
    loss.backward()
    grads = {f"{i}.{k}": p.grad.detach().clone()
             for i, m in enumerate(mods) for k, p in m.named_parameters() if p.grad is not None}
    return {"loss": float(loss), "grads": grads,
            "ds_in": s_in.grad.detach().clone(), "dz_in": z_in.grad.detach().clone(),
            "s_out_norm": float(s.norm()), "z_out_norm": float(z.norm())}


def score(a, b):
    """b against a, mass-weighted relative L2 over the shared parameter set, worst named."""
    keys = sorted(set(a["grads"]) & set(b["grads"]))
    ref_sq = err_sq = 0.0
    worst, worst_name, n_bit = 0.0, None, 0
    for k in keys:
        x, y = a["grads"][k], b["grads"][k]
        if torch.equal(x, y):
            n_bit += 1
        rn = float(torch.linalg.vector_norm(x))
        en = float(torch.linalg.vector_norm(y - x))
        ref_sq += rn ** 2
        err_sq += en ** 2
        r = en / rn if rn > 0 else (0.0 if en == 0 else float("inf"))
        if r > worst:
            worst, worst_name = r, k
    return {"compared": len(keys), "n_bit_identical": n_bit,
            "missing_on_one_side": sorted(set(a["grads"]) ^ set(b["grads"])),
            "mass_weighted_rel_l2": (err_sq / ref_sq) ** 0.5 if ref_sq else None,
            "worst_rel_l2": worst, "worst_tensor": worst_name}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--c-s", type=int, default=16)
    ap.add_argument("--c-z", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--real-block", action="store_true",
                    help="use upstream's own PairFormerBlock at small dims instead of the "
                         "proxy chain. The proxy shows checkpointing is exact for a chain of "
                         "that SHAPE; only the real block shows it is exact for the thing the "
                         "reference actually ran.")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    sys.path.insert(0, a.tree)
    import openfold3
    if not openfold3.__file__.startswith(a.tree):
        raise SystemExit(f"wrong tree: {openfold3.__file__} is not under {a.tree}")
    from openfold3.core.utils.checkpointing import checkpoint_blocks

    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    t, cs, cz = a.tokens, a.c_s, a.c_z
    s0 = torch.randn(1, t, cs, dtype=torch.float64)
    z0 = torch.randn(1, t, t, cz, dtype=torch.float64)
    sm = torch.ones(1, t, dtype=torch.float64)
    sm[:, t // 2:] = 0.0                       # pads, as the real frame has
    pm = sm.unsqueeze(-1) * sm.unsqueeze(1)
    cot_s = torch.randn(1, t, cs, dtype=torch.float64) * 1e-5
    cot_z = torch.randn(1, t, t, cz, dtype=torch.float64) * 1e-5

    if a.real_block:
        from openfold3.core.model.latent.pairformer import PairFormerBlock
        torch.manual_seed(a.seed)
        nh = 4
        dims = dict(c_s=cs, c_z=cz, c_hidden_pair_bias=cs // nh, no_heads_pair_bias=nh,
                    c_hidden_mul=cz, c_hidden_pair_att=cz // nh, no_heads_pair=nh,
                    transition_type="swiglu", transition_n=2, pair_dropout=0.25,
                    fuse_projection_weights=False, inf=1e9)
        mods = [PairFormerBlock(**dims).to(torch.float64).eval() for _ in range(a.blocks)]
        s0 = torch.randn(1, t, cs, dtype=torch.float64)
        z0 = torch.randn(1, t, t, cz, dtype=torch.float64)
        cot_s = torch.randn(1, t, cs, dtype=torch.float64) * 1e-5
        cot_z = torch.randn(1, t, t, cz, dtype=torch.float64) * 1e-5
    else:
        mods = build(a.blocks, cs, cz, a.seed)
    common = dict(ckpt_blocks=checkpoint_blocks, blocks_per_ckpt=1)
    bare = run(mods, s0, z0, sm, pm, cot_s, cot_z, "bare", use_reentrant=None, **common)

    arms = {}
    for label, ur in (("use_reentrant=None", None), ("use_reentrant=True", True),
                      ("use_reentrant=False", False)):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                got = run(mods, s0, z0, sm, pm, cot_s, cot_z, "ckpt", use_reentrant=ur, **common)
            except Exception as exc:                      # a raise IS a result here
                arms[label] = {"raised": f"{type(exc).__name__}: {exc}"}
                continue
        arms[label] = {
            "forward_bit_identical": (got["s_out_norm"] == bare["s_out_norm"]
                                      and got["z_out_norm"] == bare["z_out_norm"]),
            "loss_bit_identical": got["loss"] == bare["loss"],
            "gradient": score(bare, got),
            "ds_in_rel_l2": float(torch.linalg.vector_norm(got["ds_in"] - bare["ds_in"])
                                  / torch.linalg.vector_norm(bare["ds_in"])),
            "dz_in_rel_l2": float(torch.linalg.vector_norm(got["dz_in"] - bare["dz_in"])
                                  / torch.linalg.vector_norm(bare["dz_in"])),
            "warnings": sorted({str(x.message)[:160] for x in w}),
        }

    # A break control on the break control: a chain that is deliberately WRONG must be caught by
    # `score`, or a clean reading says nothing. Perturb one parameter and re-score.
    with torch.no_grad():
        first = next(p for p in mods[0].parameters() if p.numel())
        first.view(-1)[0] += 1e-9
    perturbed = run(mods, s0, z0, sm, pm, cot_s, cot_z, "bare", use_reentrant=None, **common)
    sanity = score(bare, perturbed)

    rep = {
        "what": __doc__.strip().splitlines()[0],
        "host": socket.gethostname(),
        "device_involved": False,
        "torch": torch.__version__,
        "tree": a.tree, "openfold3_file": openfold3.__file__,
        "shape": {"blocks": a.blocks, "tokens": t, "real_tokens": int(sm.sum()),
                  "c_s": cs, "c_z": cz, "dtype": "float64",
                  "block": "upstream PairFormerBlock" if a.real_block else "proxy chain"},
        "checkpoint_blocks_is_upstreams_own": True,
        "baseline_bare_loop": {"loss": bare["loss"], "s_out_norm": bare["s_out_norm"],
                               "z_out_norm": bare["z_out_norm"],
                               "n_grad_tensors": len(bare["grads"])},
        "arms_against_the_bare_loop": arms,
        "scorer_break_control": {
            "what": "one weight moved by 1e-9 in the bare arm, re-scored against the baseline",
            "mass_weighted_rel_l2": sanity["mass_weighted_rel_l2"],
            "fires": bool(sanity["mass_weighted_rel_l2"] and
                          sanity["mass_weighted_rel_l2"] > 0.0),
        },
    }
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("host", "torch", "arms_against_the_bare_loop",
                                          "scorer_break_control")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
