"""Is the trunk's residual add bit-exact against torch, and if not, is its error one-sided?

The in-chain substitution instrument ran out of suspects: with the four extra-MSA blocks and all
nine Evoformer op classes moved into host torch, the only device arithmetic left in the trunk is
the residual `ttnn.add_` -- 9 per Evoformer block, 432 over the stack -- and the per-block error
growth stayed at the device arm's rate instead of returning to the torch arm's. So the add is the
last thing standing, and a *biased* add is the mechanism that fits: symmetric rounding noise
grows like a random walk, a one-sided rounding error grows linearly.

This measures one add on the real thing: `z + update` at the trunk's own shapes and magnitudes,
with the update scaled to the fraction of `z` a residual branch actually contributes.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/af2_port/residual_add_probe.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _round(exact: torch.Tensor, mode: str) -> torch.Tensor:
    """`exact` (float32) narrowed to bfloat16 under one candidate rounding rule.

    float32 is sign-magnitude, so adding half an output ulp to the bit pattern and truncating
    the low 16 bits is round-half-away-from-zero, and truncating alone is round-toward-zero.
    `rne` is what torch and JAX do.
    """
    if mode == "rne":
        return exact.to(torch.bfloat16)
    bits = exact.view(torch.int32).to(torch.int64)
    if mode == "half_away":
        bits = bits + 0x8000
    return (bits & ~0xFFFF).to(torch.int32).view(torch.float32).to(torch.bfloat16)


ROUNDING_MODES = ("rne", "half_away", "truncate")


def probe(rows: int, cols: int, channels: int, ratio: float, seed: int) -> dict:
    import ttnn
    from tt_bio.tenstorrent import get_device

    device = get_device()
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(1, rows, cols, channels, generator=generator).to(torch.bfloat16)
    update = (ratio * torch.randn(1, rows, cols, channels, generator=generator)).to(torch.bfloat16)

    def up(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    zt, ut = up(z), up(update)
    got = torch.Tensor(ttnn.to_torch(ttnn.add_(zt, ut))).to(torch.bfloat16)
    want = z + update

    # The exact sum, to say which of the two roundings each arm took.
    exact = z.float() + update.float()
    got_err = (got.float() - exact).reshape(-1)
    want_err = (want.float() - exact).reshape(-1)
    scale = exact.abs().reshape(-1).clamp(min=1e-30)
    out = {"shape": [rows, cols, channels], "ratio": ratio,
           "mismatched": int((got != want).sum()), "elements": int(got.numel())}
    out["mismatch_fraction"] = out["mismatched"] / out["elements"]
    # Which rounding rule the card is actually using. `rne` is the reference's.
    out["ttnn_vs"] = {mode: int((got != _round(exact, mode)).sum()) for mode in ROUNDING_MODES}
    out["torch_vs"] = {mode: int((want != _round(exact, mode)).sum()) for mode in ROUNDING_MODES}
    # The perturbation one add hands the next op, which is what a 48-block chain compounds.
    diff = (got.float() - want.float()).reshape(-1)
    out["arm_difference"] = {"rms_rel": float((diff / scale).square().mean().sqrt()),
                             "mean_signed_rel": float((diff / scale).mean())}
    for name, err in (("ttnn", got_err), ("torch", want_err)):
        rel = err / scale
        out[name] = {"mean_signed_rel": float(rel.mean()),
                     "rms_rel": float(rel.square().mean().sqrt()),
                     # A rounding that always shrinks the magnitude is truncation toward zero.
                     "shrunk": float((err.sign() == -exact.reshape(-1).sign()).float().mean()),
                     "exact": float((err == 0).float().mean())}
    return out


def boundary(rows: int, cols: int, channels: int, seed: int) -> dict:
    """Is `_up`/`_down` lossless? The attribution rests on it.

    With every op substituted into host torch the trunk's only remaining device arithmetic is
    the residual add -- but only if a bfloat16 tensor survives the round trip bit for bit, since
    each substituted op crosses the boundary twice.
    """
    import ttnn
    from tt_bio.tenstorrent import get_device

    device = get_device()
    generator = torch.Generator().manual_seed(seed)
    out = {}
    for name, shape in (("pair", (1, rows, cols, channels)), ("msa", (1, 2, rows, 256))):
        x = torch.randn(*shape, generator=generator).to(torch.bfloat16)
        up = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        back = torch.Tensor(ttnn.to_torch(up)).to(torch.bfloat16)
        out[name] = {"shape": list(shape), "mismatched": int((back != x).sum()),
                     "elements": int(x.numel())}
    return out


def variants(rows: int, cols: int, channels: int, ratio: float, seed: int) -> dict:
    """Does the rounding depend on which add, or on the shape? A shared-kernel claim needs both.

    Every model in the repo builds a bfloat16 residual chain out of these calls, so if the rule
    is the op's rather than this call site's, it is not an AF2-IG finding.
    """
    import ttnn
    from tt_bio.tenstorrent import get_device

    device = get_device()
    generator = torch.Generator().manual_seed(seed)
    out = {}
    for name, shape in (("trunk_pair", (1, rows, cols, channels)),
                        ("one_tile", (1, 1, 32, 32)),
                        ("flat", (1, 1, 4096, 128))):
        z = torch.randn(*shape, generator=generator).to(torch.bfloat16)
        u = (ratio * torch.randn(*shape, generator=generator)).to(torch.bfloat16)
        exact = z.float() + u.float()

        def up(t):
            return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device,
                                   dtype=ttnn.bfloat16)

        for call, fn in (("add_", ttnn.add_), ("add", ttnn.add)):
            got = torch.Tensor(ttnn.to_torch(fn(up(z), up(u)))).to(torch.bfloat16)
            out[f"{name}/{call}"] = {
                "elements": int(z.numel()),
                "vs_rne": int((got != _round(exact, "rne")).sum()),
                "vs_half_away": int((got != _round(exact, "half_away")).sum())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=208)
    ap.add_argument("--cols", type=int, default=208)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--ratios", default="1.0,0.1,0.01")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    report = {"mode": "af2ig_residual_add",
              "boundary": boundary(args.rows, args.cols, args.channels, args.seed),
              "variants": variants(args.rows, args.cols, args.channels, 1.0, args.seed),
              "probes": [probe(args.rows, args.cols, args.channels, float(r), args.seed)
                         for r in args.ratios.split(",")]}
    report["verdict"] = ("EXACT" if all(p["mismatched"] == 0 for p in report["probes"])
                         else "DIVERGES")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
