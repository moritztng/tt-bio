#!/usr/bin/env python3
"""Gradcheck for `tt_bio.train.bridge`: host torch glue around a taped device op, as RFD4 trains.

host Linear A (conditioning stand-in) -> `up` -> device `ttnn.linear` with a registered weight W, taped
-> `down` -> host Linear B (a head) plus a head reading `s.detach()` (RFD4's sequence readout) -> loss.
Graded by `tt_bio.train.checks.gradcheck` against float64, per parameter: A.weight, A.bias (host, through
the device), W (device), B.weight (host). The detached head's own weight must get a gradient and must NOT
reach W. `--break` seeds the tape with ones instead of the host's gradient, which must FAIL.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.train import checks as CK  # noqa: E402

N, DI, DS, DO = 64, 32, 768, 16
CASES = {"bridge": None}


def build(name, rng):
    return {"x": rng.standard_normal((1, N, DI)), "A": rng.standard_normal((DS, DI)) / DI ** 0.5,
            "a": rng.standard_normal((DS,)) * 0.1, "W": rng.standard_normal((DS, DS)) / DS ** 0.5,
            "B": rng.standard_normal((DO, DS)) / DS ** 0.5, "D": rng.standard_normal((DO, DS)) / DS ** 0.5}


def ref_forward(t, wt, frozen):
    """`frozen` is the unperturbed inputs: the detached head reads s computed from them, which is what
    `detach` means and what central differences (which see through `detach`) can also check."""
    s = (t["x"] @ t["A"].t() + t["a"]) @ t["W"]
    s0 = ((frozen["x"] @ frozen["A"].t() + frozen["a"]) @ frozen["W"])
    return (s @ t["B"].t() * wt[0]).sum() + (s0 @ t["D"].t() * wt[1]).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--break", dest="brk", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import autograd as ag, taped_ttnn as T, tenstorrent as tt
    from tt_bio.train.bridge import Bridge
    dev = tt.get_device()
    rng = CK.case_rng(a.seed, "bridge")
    bf = lambda v: torch.from_numpy(v).to(torch.bfloat16).to(torch.float64)  # noqa: E731
    r64 = {k: bf(v) for k, v in build("bridge", rng).items()}
    wt = (torch.from_numpy(rng.standard_normal((1, N, DO))), torch.from_numpy(rng.standard_normal((1, N, DO))))
    ref = {k: v.clone().requires_grad_(k != "x") for k, v in r64.items()}

    A = torch.nn.Linear(DI, DS); B = torch.nn.Linear(DS, DO, bias=False); D = torch.nn.Linear(DS, DO, bias=False)
    with torch.no_grad():
        A.weight.copy_(r64["A"]); A.bias.copy_(r64["a"]); B.weight.copy_(r64["B"]); D.weight.copy_(r64["D"])
    Wl = ag.parameter(ttnn.from_torch(r64["W"].float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev))
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    br = Bridge(dev)
    with T.tape():
        cond = br.up(A(r64["x"].float()))
        s = br.down(ttnn.linear(cond, Wl, compute_kernel_config=ckc))
    loss = (B(s) * wt[0].float()).sum() + (D(s.detach()) * wt[1].float()).sum()
    if a.brk:
        for t, h in br.downs:
            h.register_hook(lambda g: torch.ones_like(g))
    br.backward(loss)
    got = {"A": A.weight.grad, "a": A.bias.grad, "W": ttnn.to_torch(Wl.grad).reshape(DS, DS),
           "B": B.weight.grad, "D": D.weight.grad}
    got = {k: v.to(torch.float64).numpy() for k, v in got.items()}
    names = [k for k in ref if k != "x"]
    rep = CK.gradcheck("bridge/bfloat16", ref_loss=lambda: ref_forward(ref, wt, r64),
                       ref_params=[ref[k] for k in names], tt_grads={k: got[k] for k in names},
                       op_class="reduction", dtype="bfloat16", n_probe=24, seed=a.seed)
    print(rep, flush=True)
    out = {"break": a.brk, "seed": a.seed, "pass": rep.passed, "fd_worst": rep.fd_worst,
           "grads": {k: {q: g[q] for q in ("rel_l2", "cos", "max_rel", "pass")} for k, g in rep.grads.items()},
           "text": str(rep)}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
    print("ALL_PASS" if rep.passed else "SOME_FAIL")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
