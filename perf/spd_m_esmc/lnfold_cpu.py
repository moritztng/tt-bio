"""CPU precision check: the input LayerNorm folded into the next projection, against today's LN -> bf16 -> matmul.

LN(x) W = ((x - mu) / s * g + b) W = (x Wc) / s + b W, with Wc = diag(g) W centred over its input axis (column means of
diag(g) W removed), exact in float64. The device would need only the per-row s (a stats pass) and a row scale in the
consumer. Here: every block's two input LNs of the reference ESMC, real activations of the page sequence, each path
emulated in bf16 operands with fp32 accumulation, rel l2 against float64.

    python3 perf/spd_m_esmc/lnfold_cpu.py --model esmc-300m
"""
import argparse
import sys
from pathlib import Path

import torch

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "scripts"), str(R / "tests"), str(R)]
from esmc_embed_parity import load_reference, reference_state_dict  # noqa: E402
from tt_bio import esmc  # noqa: E402


def bf(t):
    return t.to(torch.bfloat16).to(torch.float32)


def rel(a, b):
    return float((a.double() - b).norm() / b.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m")
    ap.add_argument("--seq-file", default=str(R / "scripts/gpu_vs_tt/fixtures/prot512.seq"))
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    ref = load_reference(a.model, reference_state_dict(a.model))
    caps = []
    for i, blk in enumerate(ref.transformer.blocks):
        for name, seq in (("qkv", blk.attn.layernorm_qkv), ("ffn", blk.ffn)):
            ln, lin = seq[0], seq[1]
            seq[0].register_forward_hook(lambda m, inp, out, i=i, name=name, lin=lin:
                                         caps.append((i, name, inp[0][0].clone(), m, lin)))
    ref(esmc.tokenize(Path(a.seq_file).read_text().strip()))
    worst = {}
    print("layer path  base_rel  fold_rel  |mu|/s(max)")
    for i, name, x, ln, lin in caps:
        xd, g, b, W = x.double(), ln.weight.double(), ln.bias.double(), lin.weight.double().t()  # W [d_in, d_out]
        mu, var = xd.mean(-1, keepdim=True), xd.var(-1, unbiased=False, keepdim=True)
        s = (var + ln.eps).sqrt()
        true = ((xd - mu) / s * g + b) @ W
        # today: LN in fp32 from bf16 x, written bf16, bf16 weights, fp32 accumulate
        x16 = bf(x)
        m16, v16 = x16.mean(-1, keepdim=True), x16.var(-1, unbiased=False, keepdim=True)
        xn = bf((x16 - m16) / (v16 + ln.eps).sqrt() * ln.weight + ln.bias)
        base = xn @ bf(W.float())
        # fold: bf16 x, bf16 centred gamma-folded weight, fp32 accumulate, row scale 1/s (fp32 stats), fp32 bias
        Wc = g[:, None] * W
        Wc = Wc - Wc.mean(0, keepdim=True)
        fold = (x16 @ bf(Wc.float())) / (v16 + ln.eps).sqrt() + (b @ W).float()
        r = (rel(base, true), rel(fold, true))
        worst[name] = max(worst.get(name, (0, 0)), r, key=lambda t: t[1] / t[0])
        print(f"{i:5d} {name:4s} {r[0]:.3e} {r[1]:.3e} {float((m16.abs() / v16.sqrt()).max()):.2f}")
    print("worst fold/base ratio per path:", {k: (f"{v[0]:.3e}", f"{v[1]:.3e}") for k, v in worst.items()})


if __name__ == "__main__":
    main()
