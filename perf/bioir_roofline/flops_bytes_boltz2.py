#!/usr/bin/env python3
"""FLOPs and bytes per Boltz-2 phase at a given token count, counted on the real modules.

Not a hand-derived formula: each phase is one real `tt_bio.reference` / `tt_bio.boltz2` module
run once on CPU under torch's own FlopCounterMode, with a dispatch mode underneath that sums the
bytes every aten op reads and writes. That gives two byte numbers per phase, and the roofline
placement depends on which one you use:

  resident_bytes  parameters only, i.e. the compulsory DRAM traffic if every activation stays on
                  chip for the whole phase. The lower bound, the fully fused machine.
  eager_bytes     every aten op's inputs + outputs, i.e. what an unfused implementation moves if
                  no activation is ever reused from cache. The upper bound.

Any real implementation is between them, and the distance between them is what fusion and
on-chip residency are worth. Config comes from the shipped checkpoint's own hyper_parameters.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch                                                                  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode                          # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode                    # noqa: E402
from torch.utils._pytree import tree_flatten                                  # noqa: E402


class ByteCounter(TorchDispatchMode):
    """Sum bytes read+written by every aten op, as an unfused implementation would move them."""

    def __init__(self):
        self.bytes = 0
        self.ops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = str(func)
        if not name.startswith("aten.") or "detach" in name:
            return out
        n = 0
        for t in tree_flatten((args, kwargs))[0]:
            if isinstance(t, torch.Tensor):
                n += t.numel() * t.element_size()
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor):
                n += t.numel() * t.element_size()
        self.bytes += n
        self.ops += 1
        return out


def measure(name, mod, run, mult, notes=""):
    params = sum(p.numel() * p.element_size() for p in mod.parameters()) if mod is not None else 0
    bc = ByteCounter()
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc, bc:
        run()
    fl = fc.get_total_flops()
    row = {"phase": name, "per_call_flops": fl, "per_call_eager_bytes": bc.bytes,
           "per_call_aten_ops": bc.ops, "param_bytes": params, "calls_per_fold": mult,
           "fold_flops": fl * mult, "fold_eager_bytes": bc.bytes * mult,
           "fold_resident_bytes": params * mult, "notes": notes}
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--msa-depth", type=int, default=35)
    ap.add_argument("--atoms", type=int, default=None, help="default 8 x tokens, rounded to 32")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    N, D = a.tokens, a.msa_depth
    A = a.atoms or ((N * 8 + 31) // 32) * 32
    passes = a.recycles + 1          # boltz runs the trunk recycling_steps+1 times

    from tt_bio.reference import PairformerLayer, MSALayer
    from tt_bio.boltz2 import DiffusionTransformerLayer

    torch.manual_seed(0)
    rows = []

    # --- trunk: one pairformer block, 64 of them per trunk pass -------------------------
    pf = PairformerLayer(token_s=384, token_z=128, num_heads=16, dropout=0.0,
                         pairwise_head_width=32, pairwise_num_heads=4, v2=True).eval()
    s = torch.randn(1, N, 384)
    z = torch.randn(1, N, N, 128)
    mask = torch.ones(1, N)
    pair_mask = torch.ones(1, N, N)
    rows.append(measure("pairformer_block", pf, lambda: pf(s, z, mask, pair_mask),
                        64 * passes, "64 blocks x %d trunk passes" % passes))
    del pf

    # --- MSA module: one block, 4 per trunk pass ----------------------------------------
    ms = MSALayer(msa_s=64, token_z=128, msa_dropout=0.0, z_dropout=0.0,
                  pairwise_head_width=32, pairwise_num_heads=4).eval()
    m = torch.randn(1, D, N, 64)
    rows.append(measure("msa_block", ms, lambda: ms(z, m, mask, torch.ones(1, D, N)),
                        4 * passes,
                        "4 blocks x %d trunk passes, msa depth %d" % (passes, D)))
    del ms, m

    # --- confidence: pairformer with 8 blocks, once per fold ----------------------------
    conf = dict(rows[0])
    conf.update({"phase": "confidence_pairformer_block", "calls_per_fold": 8,
                 "fold_flops": rows[0]["per_call_flops"] * 8,
                 "fold_eager_bytes": rows[0]["per_call_eager_bytes"] * 8,
                 "fold_resident_bytes": rows[0]["param_bytes"] * 8,
                 "notes": "8 blocks once per fold, same block shape as the trunk"})
    rows.append(conf)
    print(json.dumps(conf), flush=True)

    # --- diffusion: one token-transformer layer, 24 of them per sampling step -----------
    dt = DiffusionTransformerLayer(heads=16, dim=768, dim_single_cond=768).eval()
    aa = torch.randn(1, N, 768)
    ss = torch.randn(1, N, 768)
    bias = torch.randn(1, N, N, 16)
    rows.append(measure("diffusion_token_layer", dt,
                        lambda: dt(aa, ss, bias=bias, mask=torch.ones(1, N)),
                        24 * a.steps, "24 layers x %d sampling steps" % a.steps))
    del dt, aa, ss, bias

    # --- diffusion: one atom-transformer layer, 6 of them per sampling step -------------
    # atom attention is windowed: A/32 windows of 32 queries over 128 keys, dim 128.
    at = DiffusionTransformerLayer(heads=4, dim=128, dim_single_cond=128).eval()
    NW = A // 32
    q = torch.randn(NW, 32, 128)
    c = torch.randn(NW, 32, 128)
    abias = torch.randn(NW, 32, 32, 4)
    rows.append(measure("atom_transformer_layer", at,
                        lambda: at(q, c, bias=abias, mask=torch.ones(NW, 32)),
                        6 * a.steps,
                        "3 encoder + 3 decoder layers x %d steps, %d windows of 32 queries; "
                        "keys truncated to the 32-wide window, so the attention term alone is "
                        "undercounted up to 4x" % (a.steps, NW)))
    del at

    total_fl = sum(r["fold_flops"] for r in rows)
    total_eager = sum(r["fold_eager_bytes"] for r in rows)
    total_res = sum(r["fold_resident_bytes"] for r in rows)
    summary = {"tokens": N, "atoms": A, "msa_depth": D, "recycles": a.recycles,
               "trunk_passes": passes, "sampling_steps": a.steps,
               "fold_flops": total_fl, "fold_eager_bytes": total_eager,
               "fold_resident_bytes": total_res,
               "AI_eager_flop_per_byte": total_fl / total_eager,
               "AI_resident_flop_per_byte": total_fl / total_res,
               "rows": rows}
    print("SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    Path(a.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
