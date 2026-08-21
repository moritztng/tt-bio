"""Which bfloat16 elementwise ops round ties away from zero, and which round to even?

`scripts/af2_port/residual_add_probe.py` found it in `ttnn.add`/`ttnn.add_`: two bfloat16
operands, ties broken away from zero where torch and JAX break them to even, and a narrower
datapath than float32 on top. Fixing the trunk's 432 residual adds took the pair error growth
from 1.0740 to 1.0465 per block against the torch arm's 1.0359.

The residue is not one op class -- pass 10's per-class substitution screen moved at most 43% of
it with any single class -- which is what a rule shared by every op looks like. Each Evoformer op
ends in a bfloat16 elementwise call of its own: the triangle attentions and multiplications gate
with `ttnn.multiply_(o, g, sigmoid)`, the transitions relu. If those carry the add's rule they
carry the add's bias, once per op instead of once per residual.

So this asks the same question of the ops the trunk actually calls, on the trunk's own shape.
`rne` is what torch and JAX do; `half_away` is what the add was measured doing.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=. env/bin/python3 scripts/af2_port/eltwise_rounding_probe.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from residual_add_probe import ROUNDING_MODES, _round  # noqa: E402


def _cases():
    """`(name, ttnn call, torch twin, needs a second operand)`.

    Every one of these runs in the trunk. `mul_sigmoid` is the gate the two triangle attentions
    and the two triangle multiplications all end on; `mul` and `add` are its unfused halves;
    `relu` and `sigmoid` are single-operand, where a tie cannot arise from the arithmetic but the
    output narrowing can still differ.
    """
    import ttnn
    return [
        ("add", lambda a, b: ttnn.add(a, b), lambda a, b: a + b, True),
        ("mul", lambda a, b: ttnn.multiply(a, b), lambda a, b: a * b, True),
        ("mul_", lambda a, b: ttnn.multiply_(a, b), lambda a, b: a * b, True),
        ("mul_sigmoid",
         lambda a, b: ttnn.multiply(a, b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]),
         lambda a, b: a * torch.sigmoid(b.float()).to(torch.bfloat16), True),
        ("relu", lambda a, _b: ttnn.relu(a), lambda a, _b: torch.relu(a), False),
        ("sigmoid", lambda a, _b: ttnn.sigmoid(a), lambda a, _b: torch.sigmoid(a), False),
        # The same widening the residual add needed. torch computes a bfloat16 sigmoid in
        # float32 and narrows once, so if the SFPU's bfloat16 approximation is the whole of the
        # difference this closes it and the gate multiply can be fixed the same way.
        ("sigmoid_fp32",
         lambda a, _b: ttnn.typecast(ttnn.sigmoid(ttnn.typecast(a, ttnn.float32)),
                                     ttnn.bfloat16),
         lambda a, _b: torch.sigmoid(a), False),
        ("mul_sigmoid_fp32",
         lambda a, b: ttnn.multiply(
             a, ttnn.typecast(ttnn.sigmoid(ttnn.typecast(b, ttnn.float32)), ttnn.bfloat16)),
         lambda a, b: a * torch.sigmoid(b.float()).to(torch.bfloat16), True),
    ]


def probe(rows: int, cols: int, channels: int, seed: int) -> list[dict]:
    import ttnn
    from tt_bio.tenstorrent import get_device

    device = get_device()
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(1, rows, cols, channels, generator=generator).to(torch.bfloat16)
    b = torch.randn(1, rows, cols, channels, generator=generator).to(torch.bfloat16)

    def up(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    out = []
    for name, call, twin, binary in _cases():
        got = torch.Tensor(ttnn.to_torch(call(up(a), up(b)))).to(torch.bfloat16)
        want = twin(a, b)
        # The exact result in float32, so each arm's narrowing rule can be named. For the
        # activation cases the transcendental itself differs between the two libraries, so only
        # the elementwise ops get a rounding-rule verdict; the rest get the mismatch rate.
        if name in ("add", "mul", "mul_"):
            exact = (a.float() + b.float()) if name == "add" else (a.float() * b.float())
            rules = {mode: int((got != _round(exact, mode)).sum()) for mode in ROUNDING_MODES}
        else:
            exact, rules = None, None
        row = {"op": name, "binary": binary, "elements": int(a.numel()),
               "mismatched": int((got != want).sum())}
        row["mismatch_fraction"] = row["mismatched"] / row["elements"]
        # One-sidedness is the mechanism, not the mismatch rate: symmetric rounding noise grows
        # like a random walk over 48 blocks and a biased one grows linearly.
        diff = (got.float() - want.float()).reshape(-1)
        scale = want.float().abs().reshape(-1).clamp(min=1e-30)
        row["difference"] = {"mean_signed_rel": float((diff / scale).mean()),
                             "rms_rel": float((diff / scale).square().mean().sqrt()),
                             "grew": float((diff.sign() == want.reshape(-1).sign()).float()
                                           [diff != 0].mean()) if int((diff != 0).sum()) else 0.0}
        if rules is not None:
            row["ttnn_vs"] = rules
            row["torch_vs"] = {mode: int((want != _round(exact, mode)).sum())
                               for mode in ROUNDING_MODES}
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=208)
    ap.add_argument("--cols", type=int, default=208)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = probe(args.rows, args.cols, args.channels, args.seed)
    print(json.dumps({"mode": "af2ig_eltwise_rounding", "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
