#!/usr/bin/env python3
"""Is the template pair stack's VJP right on card?

The port's hard stop. `tt_bio.af2.AF2DeviceModel` already builds the two template `PairBlock`s
on card (`device_template`, c=64, 4 heads of 16, Evoformer order on multimer), and the fold path
runs them under `torch.no_grad`. A BindCraft 2 gradient round needs them under `tt_bio`'s tape,
and 83 % of the seconds this row is after are in the backward, so the backward is the thing that
has to be graded before any of it is worth porting.

Graded against a **float64** CPU reference, never against another approximation, with a
**bfloat16** CPU arm beside it as the control. bf16 on the host and bf16 on the card are two
realisations of the same arithmetic, so the reference's distance to the CPU bf16 arm is the
floor no correct device implementation can beat. The campaign's bar is the one `bcx-extramsa`
and `bcx-p10-bfp8` were graded on: the device arm within 1.1-1.2x of that floor.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-p10-tmplemb \
        PYTHONPATH=. python3 perf/bcx_p10_tmplemb/vjp.py --n 288 --out out/vjp
"""
from __future__ import annotations

import argparse
import json
import pathlib
import os
import statistics as st
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PARAMS = "/home/ttuser/bcx_e2e/af2_params/params_model_1_multimer_v3.npz"


def aiclk() -> dict:
    """The clock the device arm actually ran at, off this card's own sysfs node.

    `perf/bcx_round/meter.py`'s sampler, not `tt-smi -ls`: an all-chip probe inherits the worst
    chip's hang and mmap-holds the rest. A perf number without a clock is not a measurement.
    """
    sys.path.insert(0, str(ROOT / "perf" / "bcx_stack"))
    from stack import sysfs_node

    node, pci = sysfs_node()
    reads = []
    for _ in range(3):
        try:
            reads.append(int(open(f"{node}/tt_aiclk").read().split()[0]))
        except OSError:
            pass
        time.sleep(0.05)
    return {"pci": pci, "min": min(reads) if reads else None,
            "max": max(reads) if reads else None, "load1": round(os.getloadavg()[0], 2)}


def distance(a: torch.Tensor, b: torch.Tensor) -> dict:
    """`a` against `b`, in float64 so the report is not itself an approximation."""
    x, y = a.detach().double().flatten(), b.detach().double().flatten()
    d = x - y
    ny = float(y.norm())
    return {"rel_l2": float(d.norm()) / max(ny, 1e-300),
            "max_abs": float(d.abs().max()),
            "cos": float(torch.dot(x, y)) / max(float(x.norm()) * ny, 1e-300)}


def host_blocks(model, stack):
    return model.template.pair_stack if stack == "template" else model.extra_msa


def device_blocks(model, stack):
    return model.device_template if stack == "template" else model.device_extra_msa


def host_arm(model, act, mask, cotangent, dtype, stack):
    """The torch `PairBlock`s at `dtype`, forward and backward, on the CPU."""
    blocks = host_blocks(model, stack).to(dtype)
    a = act.to(dtype).detach().requires_grad_(True)
    m = mask.to(dtype)
    t0 = time.perf_counter()
    z = a
    for block in blocks:
        z = block(z, m)
    z.backward(cotangent.to(dtype))
    return z.detach(), a.grad.detach(), round(time.perf_counter() - t0, 3)


def device_arm(model, act, mask, cotangent, reps=1, stack="template"):
    """The same two blocks on card, under `tt_bio.autograd`'s tape -- the ported path.

    `reps` repeats forward and backward on one device context. The first pass carries ttnn's
    kernel compile, which is a one-off a design campaign pays once and a per-round number must
    not carry, so the cold pass is reported apart from the warm median.
    """
    import ttnn

    from tt_bio import autograd as ag
    from tt_bio import taped_ttnn
    from tt_bio.af2 import af2_pair_masks

    dev = model._device
    up = lambda t: ttnn.from_torch(t.detach().unsqueeze(0).to(torch.bfloat16),   # noqa: E731
                                   layout=ttnn.TILE_LAYOUT, device=dev,
                                   dtype=ttnn.bfloat16)
    shape = tuple(act.shape)

    def down(t):
        x = torch.Tensor(ttnn.to_torch(t)).float()
        while x.dim() > len(shape) and x.shape[0] == 1:
            x = x.squeeze(0)
        return x

    masks = af2_pair_masks(mask, dev)
    fwd, bwd = [], []
    out = grad = None
    clocks = []
    for _ in range(reps):
        clocks.append(aiclk())
        leaf = ag.Tensor(up(act), requires_grad=True)
        t0 = time.perf_counter()
        with taped_ttnn.tape():
            z = leaf
            for block in device_blocks(model, stack):
                z = block(z, *masks)
        ttnn.synchronize_device(dev)
        fwd.append(time.perf_counter() - t0)
        out = down(z.value)

        t1 = time.perf_counter()
        ag.backward([z], [up(cotangent)])
        ttnn.synchronize_device(dev)
        bwd.append(time.perf_counter() - t1)
        grad = down(leaf.grad)
        ag.release_pins()
    return out, grad, fwd, bwd, clocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=288, help="token axis; must be a multiple of 32")
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=4,
                    help="device forward+backward repeats; pass 1 carries ttnn's kernel compile")
    ap.add_argument("--bar", type=float, default=1.2,
                    help="how many times the bf16 CPU control the device arm may be")
    ap.add_argument("--stack", default="template", choices=["template", "extra_msa"],
                    help="extra_msa is the control: the campaign measured its four blocks at "
                         "1.054 s of device in a real round, so a harness that reads far above "
                         "that is the thing being measured, not the stack")
    ap.add_argument("--triatt-fused", default="inherit",
                    choices=["inherit", "none", "trunk", "all"],
                    help="which pair stacks take the fused SDPA triangle attention; 'all' is "
                         "the only value that reaches the template stack")
    ap.add_argument("--rne-fold", dest="rne_fold", type=int, default=0,
                    help="grade `AF2PairBlock.rne_fold_cast`: the residual's trailing typecast "
                         "folded onto the add's output. The fold DELETES a taped node, so the "
                         "forward's bit-exactness says nothing about the VJP and this is where "
                         "it is graded")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.n % 32 == 0, "the token axis buckets to a multiple of 32"

    from tt_bio.af2 import AF2PairBlock, load_af2_device_model
    AF2PairBlock.rne_fold_cast = bool(args.rne_fold)
    from tt_bio.af2_weights import load_af2_state_dict

    state = load_af2_state_dict(args.params, multimer=True)
    model = load_af2_device_model(state, template=True, multimer=True, structure=False,
                                  trunk_dtype=torch.bfloat16).eval()
    if args.triatt_fused != "inherit":
        from tt_bio.af2 import TRIATT_FUSED_ARMS
        model.set_triatt_fused(TRIATT_FUSED_ARMS[args.triatt_fused])
    c_t = (model.template.output_norm.weight.shape[0] if args.stack == "template"
           else model.evoformer[0].tri_mul_out.norm_in.weight.shape[0])

    g = torch.Generator().manual_seed(args.seed)
    act = torch.randn(args.n, args.n, c_t, generator=g)
    mask = torch.ones(args.n, args.n)
    cotangent = torch.randn(args.n, args.n, c_t, generator=g)

    # float64 has to come off a float64 copy of the weights, so the reference is built first and
    # the two lower-precision arms are cast from the same parameters afterwards.
    # The extra-MSA arm is a DEVICE-TIMING control only. Its torch block is
    # `(msa, pair, extra_mask, pair_mask) -> (msa, pair)`, not the `(act, mask) -> act` the
    # ported cut has, so there is no host arm to grade it against and none is wanted: what it
    # controls is whether this harness can reproduce the 1.054 s the campaign measured.
    grade = args.stack == "template"
    if grade:
        ref_out, ref_grad, ref_s = host_arm(model, act, mask, cotangent, torch.float64,
                                            args.stack)
        bf16_out, bf16_grad, bf16_s = host_arm(model, act, mask, cotangent, torch.bfloat16,
                                               args.stack)
        host_blocks(model, args.stack).to(torch.bfloat16)
    else:
        ref_out = ref_grad = bf16_out = bf16_grad = None
        ref_s = bf16_s = None
    dev_out, dev_grad, fwd, bwd, clocks = device_arm(model, act, mask, cotangent,
                                                     args.reps, args.stack)
    warm = lambda xs: round(st.median(xs[1:] or xs), 3)  # noqa: E731

    if grade:
        control = distance(bf16_grad, ref_grad)
        device = distance(dev_grad, ref_grad)
        ratio = device["rel_l2"] / max(control["rel_l2"], 1e-300)
        accuracy = {
            "forward": {"cpu_bf16_vs_f64": distance(bf16_out, ref_out),
                        "device_vs_f64": distance(dev_out, ref_out)},
            "vjp": {"cpu_bf16_vs_f64": control, "device_vs_f64": device,
                    "device_over_control": round(ratio, 4), "bar": args.bar,
                    "verdict": "PASS" if ratio <= args.bar else "FAIL"},
        }
    else:
        accuracy = {"graded": False,
                    "why": "device-timing control; the extra-MSA block has a different cut"}
    report = {
        "n": args.n, "c_t": c_t, "stack": args.stack, "triatt_fused": args.triatt_fused,
        "rne_fold": bool(args.rne_fold),
        "params": args.params, "seed": args.seed,
        **accuracy,
        "seconds": {"cpu_f64_fwd_bwd": ref_s, "cpu_bf16_fwd_bwd": bf16_s,
                    "device_cold_fwd": round(fwd[0], 3), "device_cold_bwd": round(bwd[0], 3),
                    "device_warm_fwd": warm(fwd), "device_warm_bwd": warm(bwd),
                    "device_warm_total": round(warm(fwd) + warm(bwd), 3),
                    "device_fwd_all": [round(x, 3) for x in fwd],
                    "device_bwd_all": [round(x, 3) for x in bwd],
                    "aiclk_during": clocks},
        "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"vjp_{args.stack}_{args.triatt_fused}_n{args.n}"
                 f"{'_fold' if args.rne_fold else ''}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
