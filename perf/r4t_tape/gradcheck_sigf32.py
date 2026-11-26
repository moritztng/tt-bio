#!/usr/bin/env python3
"""Gradcheck for the `dit_sigf32` gate linear as RFD4's trunk issues it, through the tape's linear entry.

The forward under test is `tt_bio.rfd4.trunk.AdaptiveScale` (x * sigmoid(W c + b) * mask), the production
module, run under a lever set holding `dit_sigf32` so `Module._lin` hands `ops.linear` the
`tenstorrent._SIGMOID_F32` epilogue (a `ttnn.UnaryWithParam`, which the tape used to refuse). Graded by
`tt_bio.train.checks.gradcheck` against a float64 reference validated by central differences.
Arms: bf16 and fp32 module dtypes; `--break` keeps the sigmoid forward but drops the (1 - s) factor from its
backward, which must FAIL. The taped forward composes the sigmoid instead of fusing it, so it is not
bit-identical to the served one: its deviation is reported (`fwd_max_abs`), and grad-off production is
untouched because the tape entry is only reached under a tape.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.train import checks as CK  # noqa: E402

DIM, DC, N = 768, 256, 64


CASES = {"sigf32_gate": None}
MASK = torch.ones(1, N, 1, dtype=torch.float64)
MASK[:, N - 7:] = 0.0          # seven padded tokens, as a design shorter than its bucket has


def build(name, rng):
    return {"x": rng.standard_normal((1, N, DIM)), "c": rng.standard_normal((1, N, DC)),
            "W": rng.standard_normal((DIM, DC)) / DC ** 0.5, "b": rng.standard_normal((DIM,)) * 0.5 - 1.0}


def ref_forward(t):
    return t["x"] * torch.sigmoid(t["c"] @ t["W"].t() + t["b"]) * MASK


def run(dt_name, seed, ttnn, ag, T, TR, tt):
    dt = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}[dt_name]
    tdt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dt_name]
    rng = CK.case_rng(seed, "sigf32_gate")
    r64 = {k: torch.from_numpy(v).to(tdt).to(torch.float64) for k, v in build("sigf32_gate", rng).items()}
    ref = {k: v.clone().requires_grad_(True) for k, v in r64.items()}
    wt = torch.from_numpy(rng.standard_normal((1, N, DIM)))
    dev = tt.get_device()

    def up(v):
        return ttnn.from_torch(v.to(tdt), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

    sd = {"to_adaln_zero_gamma.0.weight": r64["W"].float(), "to_adaln_zero_gamma.0.bias": r64["b"].float()}
    m = TR.AdaptiveScale(tt.WeightScope(sd), TR.compute_kernel_config(), dtype=dt)
    xv, cv, mv = up(r64["x"]), up(r64["c"]), up(MASK)
    act = tt.mm_sigmoid(m.compute_kernel_config)
    shipped = ttnn.to_torch(m(xv, cv, mv)).to(torch.float64)
    leaves = {"x": ag.Tensor(xv, requires_grad=True), "c": ag.Tensor(cv, requires_grad=True),
              "W": ag.parameter(m.w), "b": ag.parameter(m.b)}
    with T.tape():
        out = m(leaves["x"], leaves["c"], mv)
    fwd = ttnn.to_torch(out.value).to(torch.float64)
    out.backward(seed=up(wt))
    # device W is the transposed [DC, DIM] torch_to_tt layout; b is [DIM] (any leading 1s dropped)
    got = {"x": ttnn.to_torch(leaves["x"].grad), "c": ttnn.to_torch(leaves["c"].grad),
           "W": ttnn.to_torch(leaves["W"].grad).reshape(DC, DIM).t(),
           "b": ttnn.to_torch(leaves["b"].grad).reshape(DIM)}
    got = {k: v.to(torch.float64).numpy().reshape(r64[k].shape) for k, v in got.items()}

    def ref_loss():
        return (ref_forward(ref) * wt).sum()

    exact = ref_forward(r64)
    rep = CK.gradcheck(f"sigf32_gate/{dt_name}", ref_loss=ref_loss, ref_params=list(ref.values()),
                       tt_grads=dict(zip(ref, (got[k] for k in ref))), op_class="reduction",
                       dtype=dt_name, n_probe=24, seed=seed)
    return {"case": "sigf32_gate", "dtype": dt_name, "pass": rep.passed,
            "epilogue": repr(act), "epilogue_is_sigf32": not isinstance(act, str),
            "fwd_max_abs_taped_vs_served": float((fwd - shipped).abs().max()),
            "fwd_max_abs_served_vs_f64": float((shipped - exact).abs().max()),
            "fwd_max_abs_taped_vs_f64": float((fwd - exact).abs().max()),
            "fd_worst": rep.fd_worst, "fd_probed": rep.fd_probed,
            "grads": {k: {q: g[q] for q in ("rel_l2", "cos", "max_rel", "pass")} for k, g in rep.grads.items()},
            "text": str(rep)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default="bfloat16,float32")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--break", dest="brk", action="store_true",
                    help="negative control: sigmoid backward without its (1 - s) factor")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import autograd as ag, taped_ttnn as T, tenstorrent as tt
    from tt_bio.rfd4 import trunk as TR
    if a.brk:
        def broken(x):
            def make():
                def bw(g):
                    x.add_grad(ttnn.multiply(g, ttnn.sigmoid(x.value)))
                return bw
            return ag._tape(ttnn.sigmoid(x.value), [x], make, reads=(0,))
        ag._ACTIVATIONS["sigmoid"] = broken
    with TR.lever_scope(TR.mode_levers() | {"dit_sigf32"}):
        rows = [run(d, a.seed, ttnn, ag, T, TR, tt) for d in a.dtypes.split(",")]
    for r in rows:
        print(r["text"], "| epilogue:", r["epilogue"], "| fwd |taped-served|:",
              f"{r['fwd_max_abs_taped_vs_served']:.3e}", flush=True)
    rep = {"break": a.brk, "seed": a.seed, "rows": rows,
           "all_pass": all(r["pass"] and r["epilogue_is_sigf32"] for r in rows)}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=1))
    print("ALL_PASS" if rep["all_pass"] else "SOME_FAIL")
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
