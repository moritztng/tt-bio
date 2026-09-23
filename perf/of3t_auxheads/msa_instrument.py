#!/usr/bin/env python3
"""Instrument A at `msa_module` scope, against the 0.4.3 reference, on its own boundary.

Same shape as `aux_instrument.py` and the same order: the forward is measured first
(A18s first clause), then the gradient seeded with THEIR cotangent on the module output.
The boundary is `capture_boundary.py`s, whose msa_module parameter gradients are
BIT-IDENTICAL to `grads_f64_043.pt` on all 227 tensors, so it is the references own.

The bijection is taken from TENSOR IDENTITY, not from a name map. `remap_msa_module`
routes every weight through five separate per-primitive remappers before it reaches a
device upload, and hand-inverting that chain is a second thing to get wrong. Instead
`ttnn.from_torch` is wrapped for the duration of construction and each uploaded tensor is
matched back to the checkpoint by a transpose-invariant fingerprint -- the same technique
`of3t-diffusion` used to go from 283 of 870 device weights to all of them.

SS3a is honoured at the fusion boundary: TriangleMultiplication uploads a concatenation of
the checkpoints `linear_a_p` and `linear_b_p` (and the `_g` pair), so an unmatched upload
is re-fingerprinted as two halves and each half compared against its own reference tensor.
A fused comparison would let one halfs agreement mask the others error.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "of3t_confidence"))

PER_TENSOR_BAR = 5.0e-2
MEDIAN_BAR = 2.0e-2
A14_FLOOR_RATIO = 1e-8


def rel_l2(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a - b).norm() / (b.norm() + 1e-30))


def fingerprint(x):
    x = x.double()
    return (tuple(sorted(x.shape)), round(float(x.sum()), 9),
            round(float(x.abs().max()), 9), x.numel())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.path.expanduser("~/of3-weights/of3-p2-155k.pt")))
    ap.add_argument("--reference-grads", type=Path)
    ap.add_argument("--zero-model", action="store_true")
    ap.add_argument("--perturb", default=None, help="NAME:FACTOR, the SS3e control")
    ap.add_argument("--scramble-cot", action="store_true", dest="scramble_cot",
                    help="negative control: seed the backward with the SAME cotangent numbers written into the wrong positions (the flattened cotangent reversed). Norm and shape are preserved exactly, and the forward, the weights and the arithmetic are untouched, so a comparison that cannot tell this apart from the real run is not reading their seed at all. A zero-model baseline cannot catch that: it breaks OUR side.")
    ap.add_argument("--dump-grads", default="", dest="dump_grads",
                    help="write the compared device gradient TENSORS to this .pt, keyed by full checkpoint name. Without it this arm publishes only rel_l2 against the one float64 reference it was run against, so its gradient can never be compared to upstream's own bf16 training gradient -- and two distances from a shared reference do not order each other (D72).")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    t0 = time.perf_counter()

    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_msa_embedder import MSAModule
    from tt_bio.openfold3_weights import is_openbind
    import grad_device as GD

    B = torch.load(a.boundary, map_location="cpu", weights_only=False)
    pos, kw = B["inputs"]["args"], B["inputs"]["kwargs"]
    m_ref = pos[0] if pos else kw["m"]
    z_ref = pos[1] if len(pos) > 1 else kw["z"]
    pair_mask = kw["pair_mask"]
    ref_out = B["outputs"]
    cot = B["cotangents"]["out"]
    ref_grads = B["param_grads"]
    n = int(z_ref.shape[-2])
    n_seq = int(m_ref.shape[-3])
    print("boundary: N=%d tokens, %d MSA rows, z %s, cotangent %s"
          % (n, n_seq, tuple(z_ref.shape), tuple(cot.shape)), flush=True)

    sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    msd = {k: v for k, v in sd.items() if k.startswith("msa_module.")}

    fp_name, fp_clash = {}, set()
    for k, v in msd.items():
        if not torch.is_tensor(v) or not v.is_floating_point():
            continue
        f = fingerprint(v)
        if f in fp_name:
            fp_clash.add(f)
        fp_name[f] = k

    reg = {}
    orig_from_torch = ttnn.from_torch

    def recording_from_torch(tensor, *args, **kwargs):
        out = orig_from_torch(tensor, *args, **kwargs)
        try:
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                f = fingerprint(tensor)
                if f in fp_name and f not in fp_clash:
                    reg[id(out)] = (fp_name[f], None)
                elif tensor.dim() >= 1 and tensor.shape[0] % 2 == 0:
                    # SS3a: a TriangleMultiplication upload is a cat of the checkpoints
                    # a/b halves along dim 0. Fingerprint each half on its own.
                    h = tensor.shape[0] // 2
                    fa, fb = fingerprint(tensor[:h]), fingerprint(tensor[h:])
                    if (fa in fp_name and fb in fp_name
                            and fa not in fp_clash and fb not in fp_clash):
                        reg[id(out)] = (fp_name[fa], fp_name[fb])
        except Exception:
            pass
        return out

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    tb = not is_openbind(sd)
    ttnn.from_torch = recording_from_torch
    try:
        msa = MSAModule(sd, ckc, transpose_bias=tb)
    finally:
        ttnn.from_torch = orig_from_torch

    walked = GD.walk(msa)
    named, unnamed = {}, []
    for path, _owner, _attr, t in walked:
        rec = reg.get(id(t))
        if rec is None:
            unnamed.append([path, list(t.shape)])
            continue
        ag.parameter(t)
        named[id(t)] = (path, rec, t)
    print("[%.0fs] built with transpose_bias=%s: %d device weights reachable, %d named, "
          "%d unnamed" % (time.perf_counter() - t0, tb, len(walked), len(named),
                          len(unnamed)), flush=True)

    up = lambda x, dt=ttnn.bfloat16: ttnn.from_torch(
        x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
    m_d = up(m_ref.float().reshape(1, n_seq, n, -1))
    z_d = up(z_ref.float().reshape(1, n, n, -1))
    pm_d = up(pair_mask.float().reshape(1, n, n))
    # The additive token-axis mask our PairWeightedAveraging takes, built from the same
    # 1-D token mask the pair mask is the outer product of (tt_bio/token_axis.py).
    m1 = torch.diagonal(pair_mask.reshape(n, n)).reshape(1, n).clamp(0, 1)
    attn_d = up(((1 - m1).reshape(1, 1, 1, n) * -1e9))

    dn = lambda t: torch.Tensor(ttnn.to_torch(getattr(t, "value", t))).double()
    m_t, z_t = ag.Tensor(m_d), ag.Tensor(z_d)
    with ag.tape():
        m_out, z_out = msa(m_t, z_t, pair_mask=ag.Tensor(pm_d), attn_mask=ag.Tensor(attn_d))

    their_z = ref_out["out"] if isinstance(ref_out, dict) else ref_out
    their_z = their_z.double().reshape(n, n, -1)
    ours_z = dn(z_out).reshape(n, n, -1)
    tokm = m1.reshape(-1).bool()
    fwd = {"stage": "z_out", "rel_l2": rel_l2(ours_z, their_z),
           "rel_l2_real_block": rel_l2(ours_z[tokm][:, tokm], their_z[tokm][:, tokm]),
           "ref_norm": float(their_z.norm()), "our_norm": float(ours_z.norm())}
    print("  FWD z_out rel_l2 %.4e   real-block %.4e   |ref| %.6g |ours| %.6g"
          % (fwd["rel_l2"], fwd["rel_l2_real_block"], fwd["ref_norm"], fwd["our_norm"]),
          flush=True)

    cot_seed = cot.double().reshape(tuple(int(d) for d in z_out.shape))
    if a.scramble_cot:
        cot_seed = cot_seed.reshape(-1).flip(0).reshape(cot_seed.shape).contiguous()
    seed = up(cot_seed.float())
    ag.backward([z_out], [seed])
    print("[%.0fs] backward done" % (time.perf_counter() - t0), flush=True)

    pert_name, pert_factor = (a.perturb.rsplit(":", 1) if a.perturb else (None, None))
    rows, skipped, dumped = [], [], {}
    for _tid, (path, rec, t) in named.items():
        leaf = ag.parameter_for(t)
        if leaf is None or leaf.grad is None:
            skipped.append({"path": path, "skip": "no gradient reached this leaf"})
            continue
        gd = dn(leaf.grad)
        if a.zero_model:
            gd = torch.zeros_like(gd)
        na, nb_ = rec
        targets = [(na, None)] if nb_ is None else [(na, 0), (nb_, 1)]
        for name, half in targets:
            ref = ref_grads.get(name)
            if ref is None:
                skipped.append({"path": path, "name": name,
                                "skip": "no reference gradient under this name"})
                continue
            ref = ref.double()
            g = gd
            if half is not None:
                h = g.shape[0] // 2
                g = g[:h] if half == 0 else g[h:]
            if tuple(g.shape) != tuple(ref.shape):
                if tuple(g.t().shape) == tuple(ref.shape):
                    g = g.t()
                else:
                    skipped.append({"path": path, "name": name,
                                    "skip": "shape %s vs ref %s"
                                            % (tuple(g.shape), tuple(ref.shape))})
                    continue
            if pert_name and name == pert_name:
                g = g * float(pert_factor)
            rows.append({"name": name, "path": path, "rel_l2": rel_l2(g, ref),
                         "ref_norm": float(ref.norm()), "our_norm": float(g.norm()),
                         "ref_sq": float((ref ** 2).sum())})
            dumped[name] = g

    norms = sorted(r["ref_norm"] for r in rows) or [1.0]
    floor = A14_FLOOR_RATIO * norms[len(norms) // 2]
    a14 = [r for r in rows if r["ref_norm"] < floor]
    rows = [r for r in rows if r["ref_norm"] >= floor]
    rows.sort(key=lambda r: -r["rel_l2"])
    meds = sorted(r["rel_l2"] for r in rows)
    tot = sum(r["ref_sq"] for r in rows) or 1.0
    mw = math.sqrt(sum(r["rel_l2"] ** 2 * r["ref_sq"] for r in rows) / tot)
    inside = sum(r["ref_sq"] for r in rows if r["rel_l2"] <= PER_TENSOR_BAR) / tot

    reach = {}
    if a.reference_grads:
        ref_all = torch.load(a.reference_grads, map_location="cpu", weights_only=False)
        sec = sum(float((v.double() ** 2).sum()) for k, v in ref_all.items()
                  if k.startswith("msa_module.") and v is not None)
        model = sum(float((v.double() ** 2).sum()) for v in ref_all.values() if v is not None)
        reach = {"msa_module_squared_norm": sec, "model_squared_norm": model,
                 "msa_module_share_of_model": sec / model,
                 "compared_squared_norm": tot,
                 "share_of_msa_module_own_norm": tot / sec,
                 "share_of_model_squared_norm": tot / model,
                 "n_msa_module_tensors": sum(1 for k in ref_all if k.startswith("msa_module."))}

    report = {
        "instrument": "instrument A at msa_module scope, 0.4.3 boundary",
        "boundary": {"file": str(a.boundary), "n_tokens": n, "n_msa_rows": n_seq,
                     "transpose_bias": tb},
        "bars": {"per_tensor": PER_TENSOR_BAR, "median": MEDIAN_BAR},
        "bijection": {"n_device_weights": len(walked), "n_named": len(named),
                      "n_unnamed": len(unnamed), "unnamed": unnamed[:40]},
        "forward": fwd,
        "gradient": {
            "n_scored": len(rows), "n_a14_excluded": len(a14), "n_skipped": len(skipped),
            "mass_weighted_rel_l2": mw,
            "median_rel_l2": meds[len(meds) // 2] if meds else None,
            "mass_inside_per_tensor_bar": inside,
            "n_over_bar": sum(1 for r in rows if r["rel_l2"] > PER_TENSOR_BAR),
            "worst": rows[0] if rows else None,
            "heaviest": sorted(rows, key=lambda r: -r["ref_sq"])[:5],
            "a14_excluded": [{"name": r["name"], "ref_norm": r["ref_norm"]} for r in a14],
            "rows": rows, "skipped": skipped, "reach": reach,
            "zero_model_arm": bool(a.zero_model), "perturbation": a.perturb,
            "cotangent_scrambled": bool(a.scramble_cot),
        },
    }
    if rows:
        print("PER-PARAMETER vs float64 0.4.3: %d scored, %d A14-excluded, %d skipped"
              % (len(rows), len(a14), len(skipped)))
        print("  mass-weighted %.4e   median %.4e   over bar %d of %d   mass inside %.6f %%"
              % (mw, report["gradient"]["median_rel_l2"],
                 report["gradient"]["n_over_bar"], len(rows), inside * 100))
        print("  worst   %s  %.4e" % (rows[0]["name"], rows[0]["rel_l2"]))
        for h in report["gradient"]["heaviest"][:3]:
            print("  heavy   %8.4f %% of compared mass  rel_l2 %.4e  %s"
                  % (h["ref_sq"] / tot * 100, h["rel_l2"], h["name"]))
        if reach:
            print("  reach: %.4f %% of msa_module own squared norm = %.6f %% of the model"
                  % (reach["share_of_msa_module_own_norm"] * 100,
                     reach["share_of_model_squared_norm"] * 100))
    if a.dump_grads:
        Path(a.dump_grads).parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in dumped.items()}, a.dump_grads)
        report["grads_dumped_to"] = a.dump_grads
        print("[%.0fs] wrote %d gradient tensors to %s"
              % (time.perf_counter() - t0, len(dumped), a.dump_grads), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("[%.0fs] written %s" % (time.perf_counter() - t0, a.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
