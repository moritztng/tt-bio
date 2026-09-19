#!/usr/bin/env python3
"""Does bf16 STOP rather than go wrong? The question a gradient error cannot answer.

`hallgrad-bf16-design-state-floor`: a step smaller than bf16's spacing simply does not land,
so the loss stops moving and the run reads as converged. That failure is invisible to a
gradcheck -- the gradient can be perfectly accurate and the training still be dead -- so it
has to be measured by actually optimising something and watching the curve.

Three configurations of the same DiT block fitting the same fixed target, same seed, same
initialisation, same number of steps:

  fp32/fp32   fp32 activations, fp32 master weights   -- what the module serves today
  bf16/fp32   bf16 activations, fp32 master weights   -- tt-train's adamw_full_precision
  bf16/bf16   bf16 activations, bf16 weights          -- the naive port

`kept` per step is the fraction of weight elements the step actually moved. A configuration
that descends has `kept` near 1; a configuration that stalls has the loss flat and `kept`
collapsing, and the two together are what distinguish a stall from a converged run.

Adam is host-side on the master weights, which is where it has to be: `moreh_adamw` cannot
hold an fp32 master (`ptxft-build`, and tt-train's own
`optimizers/adamw_full_precision.cpp:26-31` builds the masters in fp32 for exactly this
reason). The device holds the visible weight at the arm's dtype.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import dit


def aiclk():
    root = Path("/sys/class/tenstorrent")
    return {p.name.split("!")[1]: int((p / "tt_aiclk").read_text().strip())
            for p in sorted(root.glob("tenstorrent!*"))}


class Adam:
    """Adam on host masters, at the master's dtype. betas/eps are torch's defaults, which are
    Protenix's (`configs/configs_base.py` adam_beta1 0.9, beta2 0.95, eps 1e-8) except beta2;
    beta2 is set here to Protenix's 0.95."""

    def __init__(self, w, dtype, lr, b1=0.9, b2=0.95, eps=1e-8):
        self.dtype, self.lr, self.b1, self.b2, self.eps = dtype, lr, b1, b2, eps
        self.w = {k: torch.from_numpy(v).to(dtype) for k, v in w.items()}
        # Moments stay fp32 whatever the master is: rounding them as well would confound the
        # master's floor with the moments', and tt-train allocates both in fp32 regardless.
        self.m = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in self.w.items()}
        self.v = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in self.w.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        moved = total = 0
        for k, g in grads.items():
            g = torch.from_numpy(g).to(torch.float32).reshape(self.w[k].shape)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            upd = (self.lr * mh / (vh.sqrt() + self.eps)).to(self.dtype)
            new = (self.w[k] - upd).to(self.dtype)
            moved += int((new != self.w[k]).sum()); total += new.numel()
            self.w[k] = new
        return moved / total


def run(arm, ttnn, ag, tt, nt, steps, lr, seed):
    act32, mas32 = arm["act32"], arm["master32"]
    torch_dt = torch.float32 if act32 else torch.bfloat16
    tt_dt = ttnn.float32 if act32 else ttnn.bfloat16
    precise = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    cfg = precise if act32 else ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=True)

    rng = np.random.default_rng(seed)
    raw = dit.params(rng, nt)
    target = rng.standard_normal((nt, dit.C)) * 0.3      # the fixed thing it has to fit
    opt = Adam({k: raw[k] for k in dit.WEIGHTS},
               torch.float32 if mas32 else torch.bfloat16, lr)

    def up(v, d=torch_dt):
        d_tt = ttnn.float32 if d == torch.float32 else ttnn.bfloat16
        return ttnn.from_torch(v.to(d) if torch.is_tensor(v) else torch.from_numpy(v).to(d),
                               dtype=d_tt, layout=ttnn.TILE_LAYOUT, device=tt.get_device())

    const = {k: ag.Tensor(up(raw[k]), requires_grad=False) for k in dit.INPUTS + ("_bias",)}
    tgt = torch.from_numpy(target).to(torch.float64)
    hist = []
    for i in range(steps):
        dev = dict(const)
        for k in dit.WEIGHTS:
            # The visible device weight is the master rounded to the arm's dtype, which is
            # `adamw_full_precision.cpp:98-100`'s typecast-back. With a bf16 master the two
            # are the same tensor and there is nothing to round.
            dev[k] = ag.Tensor(up(opt.w[k].to(torch_dt)), requires_grad=True)
        out = dit.block_tape(ag, dev, nt, cfg=cfg, bwcfg=precise, attn_cfg=None)
        y = ttnn.to_torch(out.value).to(torch.float64)
        diff = y - tgt
        loss = float((diff * diff).mean())
        # dL/dy for a mean square error; the seed crosses to the device at the arm's dtype,
        # which is itself part of what the arm is measuring.
        out.backward(seed=up(2.0 * diff / diff.numel()))
        grads = {k: ttnn.to_torch(dev[k].grad).to(torch.float64).numpy() for k in dit.WEIGHTS}
        kept = opt.step(grads)
        gn = float(np.sqrt(sum((g ** 2).sum() for g in grads.values())))
        hist.append({"step": i, "loss": loss, "kept": round(kept, 6), "grad_norm": gn})
        for k in dit.WEIGHTS:
            ttnn.deallocate(dev[k].value)
        if i % 10 == 0 or i == steps - 1:
            print(f"  {arm['name']:10s} step {i:4d} loss {loss:.6e} kept {kept:.4f} "
                  f"|g| {gn:.3e}", flush=True)
    return hist


ARMS = {
    "fp32_fp32": {"name": "fp32/fp32", "act32": True, "master32": True},
    "bf16_fp32": {"name": "bf16/fp32", "act32": False, "master32": True},
    "bf16_bf16": {"name": "bf16/bf16", "act32": False, "master32": False},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="fp32_fp32,bf16_fp32,bf16_bf16")
    ap.add_argument("--nt", type=int, default=128)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag
    tt.get_device()

    rows = {}
    t0 = time.time()
    clk0 = aiclk()
    for a in args.arms.split(","):
        if a:
            rows[a] = run(ARMS[a], ttnn, ag, tt, args.nt, args.steps, args.lr, args.seed)
    rep = {"nt": args.nt, "steps": args.steps, "lr": args.lr, "seed": args.seed,
           "aiclk_before": clk0, "aiclk_after": aiclk(),
           "seconds": round(time.time() - t0, 1),
           "summary": {k: {"loss_first": v[0]["loss"], "loss_last": v[-1]["loss"],
                           "drop": v[0]["loss"] / v[-1]["loss"],
                           "kept_first": v[0]["kept"], "kept_last": v[-1]["kept"],
                           "kept_mean": round(float(np.mean([r["kept"] for r in v])), 6),
                           "grad_norm_last": v[-1]["grad_norm"]}
                       for k, v in rows.items()},
           "history": rows}
    print(json.dumps(rep["summary"], indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
