"""float64 reference and grader for AF2's triangle multiplication, forward and VJP.

The yardstick the fused block kernels are graded on (bc4-bet-block): never another approximation.
`reference(z, mask, w, ending)` returns the float64 forward and, given a cotangent, the float64 VJP
(dz and every weight gradient) of `tt_bio.af2_reference.TriangleMultiplication`, the module AF2's
`modules.py:1017` defines. `grade(ref, got)` is rel L2 per tensor.

    python perf/bc4_block/trimul_ref.py --params <npz> --block 0 --n 224 [--ending]

prints the reference's own bf16 envelope: the same module run in bfloat16 (torch, CPU) against
float64, which is the bar a kernel's error is quoted beside.
"""
import argparse
import torch

from tt_bio.af2_reference import TriangleMultiplication

DIM = HIDDEN = 128


def module(w: dict, ending: bool, dtype=torch.float64) -> TriangleMultiplication:
    m = TriangleMultiplication(DIM, HIDDEN, ending)
    m.load_state_dict(w, strict=True)
    return m.to(dtype)


def weights(params: str, block: int, ending: bool) -> dict:
    from tt_bio.af2_weights import load_af2_state_dict
    sd = load_af2_state_dict(params, multimer=True)
    name = "tri_mul_in" if ending else "tri_mul_out"
    pre = next(k[:k.index(name)] for k in sd
               if k.startswith("evoformer") and f".{block}." in k and name in k) + name + "."
    return {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}


def reference(z, mask, w, ending, cot=None, dtype=torch.float64) -> dict:
    """Forward (and VJP if `cot`) of the module at `dtype`, inputs cast to it."""
    m = module(w, ending, dtype)
    zz = z.detach().to(dtype).requires_grad_(cot is not None)
    out = m(zz, mask.to(dtype))
    res = {"out": out.detach()}
    if cot is not None:
        out.backward(cot.to(dtype))
        res["dz"] = zz.grad.detach()
        res.update({f"d_{k}": p.grad.detach() for k, p in m.named_parameters()})
    return res


def rel_l2(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.double(), got.double()
    return float((got - ref).norm() / ref.norm().clamp_min(1e-300))


def grade(ref: dict, got: dict) -> dict:
    return {k: rel_l2(ref[k], got[k]) for k in ref if k in got}


def inputs(n: int, seed: int = 0, valid: int | None = None):
    """A pair with LayerNorm-scale activations and a ragged mask (the last n - valid rows padded)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, n, DIM, generator=g, dtype=torch.float64) * 2.0
    mask = torch.ones(n, n, dtype=torch.float64)
    if valid is not None:
        mask[valid:, :] = 0
        mask[:, valid:] = 0
    cot = torch.randn(n, n, DIM, generator=g, dtype=torch.float64)
    return z, mask, cot


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--n", type=int, default=224)
    ap.add_argument("--valid", type=int, default=195)
    ap.add_argument("--ending", action="store_true")
    a = ap.parse_args()
    w = weights(a.params, a.block, a.ending)
    z, mask, cot = inputs(a.n, valid=a.valid)
    ref = reference(z, mask, w, a.ending, cot)
    bf = reference(z.bfloat16(), mask, w, a.ending, cot, dtype=torch.bfloat16)
    f32 = reference(z.float(), mask, w, a.ending, cot, dtype=torch.float32)
    print({"n": a.n, "block": a.block, "ending": a.ending,
           "bf16_envelope": {k: round(v, 5) for k, v in grade(ref, bf).items()},
           "fp32_envelope": {k: round(v, 7) for k, v in grade(ref, f32).items()}})
