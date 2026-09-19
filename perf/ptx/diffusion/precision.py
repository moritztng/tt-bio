#!/usr/bin/env python3
"""Does the diffusion module's backward need fp32, and what does bf16 actually do?

The claim inherited from `ptxft-build` is "the diffusion module's backward needs fp32". The
forward already runs fp32 by default (`protenix.py:884`, `PROTENIX_DIFFUSION_FP32_DEVICE`
defaults on, matching upstream's autocast boundary), so the backward's requirement has been
carried along with it rather than measured. This measures it.

Three things, because "needs fp32" is three different questions:

1. IS THE GRADIENT WRONG IN bf16? One DiT block's gradient against a float64 reference of
   the same composition, the reference itself checked against float64 central differences
   first. Scored per op class on the bars `perf/train_a1_defork/gradcheck_dispatch.py`
   pre-registered from the bf16 mantissa -- reused, not restated, so the two rows'
   numbers are comparable.

2. DOES IT STALL RATHER THAN DIVERGE? `hallgrad-bf16-design-state-floor` records a step
   below bf16 spacing simply stopping, which reads as convergence. So the gradient error
   is only half the question: the other half is whether an Adam step built from that
   gradient survives being written back into a bf16 weight. `kept` below is the fraction of
   weight elements that actually CHANGED, measured on the block's real gradient at its real
   parameter scale, for a bf16 weight and for an fp32 master.

3. WHERE, not whether. Arm `mixed` puts fp32 only where (1) says it is needed and leaves
   the rest bf16. A stack of perturbations is sub-additive, so the mixed arm is measured as
   a stack rather than predicted from the per-site readings.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
_spec = importlib.util.spec_from_file_location("_gc", REPO / "perf" / "hallgrad" / "gradcheck.py")
GC = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(GC)
import dit

FD_BAR = 2e-6

# `gradcheck_dispatch.py`'s bars, imported by value rather than restated. Every gradient in
# this block is reduction-carrying -- there is no pure-eltwise leaf in a DiT block, because
# even the two sigmoid gates sit on a linear -- so the reduction row is the one that scores.
BARS = {
    "float32":  (2.5e-02, 0.9999, "3.5x the measured fp32 HiFi2 matmul floor 7.05e-03 "
                                  "(gradcheck_dispatch.py); fp32 does not mean exact here"),
    "bfloat16": (1.0e-02, 0.9999, "the bf16 mantissa, sqrt(2)*2^-9 = 2.76e-03"),
}


def aiclk():
    """Every card's AICLK off sysfs, sampled at the moment of the call. Opens no device."""
    root = Path("/sys/class/tenstorrent")
    out = {}
    for p in sorted(root.glob("tenstorrent!*")):
        try:
            out[p.name.split("!")[1]] = int((p / "tt_aiclk").read_text().strip())
        except (OSError, ValueError):
            pass
    return out


def arms(ttnn):
    """The four precision arms, each a (tensor dtype, forward config, backward config) triple.

    `precise` is tt-train's `ComputeKernelConfig::precise()`: HiFi4, fp32 destination
    accumulation, packer L1 accumulation (core/compute_kernel_config.cpp:9-16).
    """
    def ckc(fid, acc):
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
            fp32_dest_acc_en=acc, packer_l1_acc=True)

    precise, hifi2 = ckc("HiFi4", True), ckc("HiFi2", False)
    return {
        # what the diffusion module serves today: fp32 tensors, HiFi4 + fp32 dest acc
        "fp32":  dict(dt=ttnn.float32, torch_dt=torch.float32, cfg=precise, bwcfg=precise,
                      attn_cfg=None),
        # the shipped opt-out, PROTENIX_DIFFUSION_FP32_DEVICE=0, with the tape's own
        # precise() backward on top -- i.e. bf16 operands, best available accumulation
        "bf16":  dict(dt=ttnn.bfloat16, torch_dt=torch.bfloat16, cfg=hifi2, bwcfg=precise,
                      attn_cfg=None),
        # bf16 operands AND a bf16-grade backward: the accumulation lever alone, isolated
        "bf16_lofi": dict(dt=ttnn.bfloat16, torch_dt=torch.bfloat16, cfg=hifi2, bwcfg=hifi2,
                          attn_cfg=None),
        # bf16 everywhere except the attention matmuls and softmax, which is the site the
        # shipped forward already singles out with fp32_raw_matmul_attention=True
        "mixed": dict(dt=ttnn.bfloat16, torch_dt=torch.bfloat16, cfg=hifi2, bwcfg=precise,
                      attn_cfg=precise),
        # The two halves of the operand question, separated. `ptxft-build` measured an fp32
        # weight against a bf16 activation as a 26x defect on one linear; these two arms ask
        # which half a whole block's gradient is actually paying for.
        "w32_a16": dict(dt=ttnn.bfloat16, torch_dt=torch.bfloat16, cfg=precise, bwcfg=precise,
                        attn_cfg=None, fp32=dit.WEIGHTS),
        "w16_a32": dict(dt=ttnn.bfloat16, torch_dt=torch.bfloat16, cfg=precise, bwcfg=precise,
                        attn_cfg=None, fp32=dit.INPUTS + ("_bias",)),
    }


def run_arm(name, spec, ttnn, ag, tt, nt, seed):
    rng = np.random.default_rng(seed)
    raw = dit.params(rng, nt)
    dt, torch_dt = spec["dt"], spec["torch_dt"]

    # Inputs rounded to the device dtype FIRST, and the reference fed the rounded values, so
    # the comparison isolates the backward's error from input quantisation.
    fp32_names = set(spec.get("fp32") or ())
    # A tensor named in `fp32` is rounded to fp32 and uploaded as fp32; every other tensor
    # takes the arm's dtype. The reference is fed whatever each tensor was actually rounded
    # to, so a split arm is scored against its own quantisation, not the uniform one.
    dt_of = {k: (torch.float32 if k in fp32_names else torch_dt) for k in raw}
    rounded = {k: torch.from_numpy(v).to(dt_of[k]).to(torch.float64) for k, v in raw.items()}
    ref = {k: v.clone().requires_grad_(k != "_bias") for k, v in rounded.items()}
    wt = torch.from_numpy(rng.standard_normal((nt, dit.C)))

    def loss_fn(_r=ref):
        return (dit.block_ref(_r, nt) * wt).sum()

    t0 = time.time()
    leaves = [v for k, v in ref.items() if k != "_bias"]
    fd_worst, n_probed, n_elig = GC.fd_check(loss_fn, leaves, n_probe=24, seed=seed)
    res = {"arm": name, "nt": nt, "seed": seed, "fd_worst": float(fd_worst),
           "fd_probed": n_probed, "fd_eligible": n_elig, "fd_ok": bool(fd_worst < FD_BAR),
           "fd_seconds": round(time.time() - t0, 1), "grads": {}}
    if not res["fd_ok"]:
        res["pass"] = False
        return res, None
    ref_grads = {k: v.grad.detach().numpy().copy() for k, v in ref.items() if k != "_bias"}

    def up(v, d=torch_dt):
        tt_dt = ttnn.float32 if d == torch.float32 else ttnn.bfloat16
        return ttnn.from_torch(v.to(d), dtype=tt_dt, layout=ttnn.TILE_LAYOUT,
                               device=tt.get_device())

    dev = {k: ag.Tensor(up(v, dt_of[k]), requires_grad=(k != "_bias"))
           for k, v in rounded.items()}
    clk0 = aiclk()
    out = dit.block_tape(ag, dev, nt, cfg=spec["cfg"], bwcfg=spec["bwcfg"],
                         attn_cfg=spec["attn_cfg"])
    out.backward(seed=up(wt, dt_of["a"]))
    ttnn.synchronize_device(tt.get_device())
    res["aiclk_before"], res["aiclk_after"] = clk0, aiclk()

    got = {k: (ttnn.to_torch(v.grad).to(torch.float64).numpy() if v.grad is not None else None)
           for k, v in dev.items() if k != "_bias"}
    dts = "float32" if (dt == ttnn.float32 or not set(dt_of.values()) - {torch.float32}) \
        else "bfloat16"
    rel_bar, cos_bar, why = BARS[dts]
    res["dtype_split"] = {k: str(v).replace("torch.", "") for k, v in sorted(dt_of.items())} \
        if fp32_names else None
    ok = True
    for k, r in ref_grads.items():
        if got[k] is None:
            res["grads"][k] = {"pass": False, "error": "no gradient reached this input"}
            ok = False
            continue
        m = GC.metrics(got[k].reshape(r.shape), r)
        m["rel_l2_bar"], m["cos_bar"], m["bar_from"] = rel_bar, cos_bar, why
        m["pass"] = bool(m["rel_l2"] <= rel_bar and m["cos"] >= cos_bar)
        ok = ok and m["pass"]
        res["grads"][k] = m
    res["bar"] = {"rel_l2": rel_bar, "cos": cos_bar, "from": why}
    res["worst_rel_l2"] = max(v["rel_l2"] for v in res["grads"].values() if "rel_l2" in v)
    res["worst_cos"] = min(v["cos"] for v in res["grads"].values() if "cos" in v)
    res["pass"] = bool(ok)
    return res, (got, {k: rounded[k].numpy() for k in ref_grads})


def stall(grads, weights, lrs):
    """`kept`: the fraction of weight elements an Adam step actually MOVES.

    The question `hallgrad-bf16-design-state-floor` forces. A gradient can be perfectly
    accurate and the training still stop, because the update is written back into a weight
    whose spacing is larger than the update. bf16 carries 8 total mantissa bits, so its
    spacing at |w| is 2^-8 |w| = 3.9e-03 |w|; fp32 carries 24, spacing 6.0e-08 |w|.

    Adam's first step is sign(g) * lr elementwise (m/sqrt(v) = 1 at step 1 for any g), so
    the update magnitude is lr and does not depend on the gradient's scale at all. That is
    what makes this a WEIGHT-dtype question rather than a gradient-dtype one, and it is why
    an accurate bf16 gradient does not rescue a bf16 master.
    """
    out = {}
    for lr in lrs:
        row = {}
        for dtname, dtype in (("bfloat16", torch.bfloat16), ("float32", torch.float32)):
            moved = total = 0
            for k, w in weights.items():
                t = torch.from_numpy(w).to(dtype)
                step = torch.from_numpy(np.sign(grads[k]).reshape(w.shape) * lr).to(dtype)
                new = (t - step).to(dtype)
                moved += int((new != t).sum()); total += t.numel()
            row[dtname] = {"kept": round(moved / total, 6), "elements": total}
        out[f"lr={lr:g}"] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="fp32,bf16,bf16_lofi,mixed,w32_a16,w16_a32")
    ap.add_argument("--nt", type=int, default=128,
                    help="tokens. The gradcheck's float64 reference is host-side and O(nt^2) "
                         "in the attention, so the precision arms run small; the shapes that "
                         "have to be production-sized are the throughput row's.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lrs", default="1.8e-3,1e-3,1e-4,1e-5")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag
    tt.get_device()

    A = arms(ttnn)
    rows, stalls = [], None
    for name in args.arms.split(","):
        if not name:
            continue
        r, pair = run_arm(name, A[name], ttnn, ag, tt, args.nt, args.seed)
        print(f"{name:10s} pass={r['pass']} worst_rel_l2="
              f"{r.get('worst_rel_l2')} worst_cos={r.get('worst_cos')} "
              f"fd={r['fd_worst']:.2e}", flush=True)
        rows.append(r)
        if name == "fp32" and pair is not None:
            # The stall test wants a TRUSTED gradient, so it uses the fp32 arm's.
            g, w = pair
            stalls = stall(g, {k: v for k, v in w.items() if k in dit.WEIGHTS},
                           [float(x) for x in args.lrs.split(",")])

    rep = {"nt": args.nt, "seed": args.seed, "fd_bar": FD_BAR, "bars": BARS,
           "blocks_in_module": dit.BLOCKS, "arms": rows, "stall": stalls,
           "aiclk_end": aiclk(),
           "all_pass": bool(all(r["pass"] for r in rows))}
    print(json.dumps(rep, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
