#!/usr/bin/env python3
"""Replay the one backward call that returns NaN. Deterministic, seconds, no campaign.

The NaN appears after 3 to 5 optimisation rounds and the round varies run to run, so
reproducing it through BindCraft 2 costs a 6-minute trajectory and gives a different
answer each time. splice.py under BCX_NANCAP writes the exact inputs of the first failing
backward; this replays that single call, and it is the same numbers every time.

  capture   BCX_NANCAP=<dir> BCX_NANLOG=1 perf/bcx_predictor/trace_exit.py
  replay    perf/bcx_predictor/replay_nan.py <dir>/nan_backward_inputs.npz

Prints where the non-finite values are and, with --per-block, the gradient magnitude after
each of the 48 blocks, which is what separates an overflow that grows from a single bad op.
"""
import argparse, json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np, torch
import afgrad as A, stack as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--depths", default="",
                    help="comma list of Evoformer depths to bisect, e.g. 1,2,4,8,16,32,48. "
                         "The captured inputs make every depth deterministic, so the first "
                         "depth whose backward is non-finite is the block the NaN is born in.")
    ap.add_argument("--k-evo", type=int, default=48)
    args = ap.parse_args()
    z = np.load(args.npz)
    print({k: (z[k].shape if z[k].ndim else z[k].item()) for k in z.files}, flush=True)

    lv = S.Levers(); dm, _ = A.load_models(A.DEFAULT_PARAMS)
    dev = A.Dev(dm.to_device()); lv.arm("stack")
    m = torch.from_numpy(z["msa_leaf"]).float()
    p_ = torch.from_numpy(z["pair_leaf"]).float()
    mask_host = torch.from_numpy(z["mask"]).float()
    FRESH = bool(__import__("os").environ.get("BCX_FRESH_MASK"))

    def mask_t():
        """One upload per call when BCX_FRESH_MASK is set, otherwise one shared tensor.

        The shared tensor is what splice.py does -- it caches the uploaded mask by shape
        and hands the same device tensor to every round of a trajectory. If an op in the
        trunk writes through it, the second call sees a mutated mask, which is the shape
        of the clean/NaN alternation this replay shows on identical inputs.
        """
        return dev.up(mask_host)

    mask = mask_t()
    n = int(z["n"])

    out = {"npz": args.npz, "n": n, "k_evo": args.k_evo}
    ml, zl = dev.leaf(m), dev.leaf(p_)
    with dev.tt.tape():
        mo, zo = dev.stack(ml, zl, 0, args.k_evo, ckpt=True, msa_mask=mask)
    dev.sync()
    fm = dev.down(mo.value, tuple(m.shape)); fz = dev.down(zo.value, tuple(p_.shape))
    out["forward_finite"] = {"msa": bool(torch.isfinite(fm).all()),
                             "pair": bool(torch.isfinite(fz).all()),
                             "msa_absmax": float(fm[torch.isfinite(fm)].abs().max()),
                             "pair_absmax": float(fz[torch.isfinite(fz)].abs().max())}
    print("forward:", json.dumps(out["forward_finite"]), flush=True)

    # The cotangents are at the REAL width and the leaves are padded, exactly as
    # _backward sees them, so they are zero-extended the same way before seeding.
    gm = torch.zeros(tuple(m.shape)); gz = torch.zeros(tuple(p_.shape))
    gm[:, :n] = torch.from_numpy(z["cot_msa"]).float()
    gz[:n, :n] = torch.from_numpy(z["cot_pair"]).float()
    out["cotangent_finite"] = {"msa": bool(torch.isfinite(gm).all()),
                               "pair": bool(torch.isfinite(gz).all()),
                               "msa_absmax": float(gm.abs().max()),
                               "pair_absmax": float(gz.abs().max())}
    print("cotangent:", json.dumps(out["cotangent_finite"]), flush=True)

    dev.ag.backward([mo, zo], [dev.seed(gm, mo), dev.seed(gz, zo)])
    dev.sync()
    dm_ = dev.grad(ml, tuple(m.shape)); dz_ = dev.grad(zl, tuple(p_.shape))
    dev.ag.release_pins()

    def rep(t, tag):
        a = t.double()
        fin = torch.isfinite(a)
        return {"tag": tag, "nan": int(torch.isnan(a).sum()), "inf": int(torch.isinf(a).sum()),
                "size": int(a.numel()),
                "absmax_finite": float(a[fin].abs().max()) if fin.any() else None}
    out["backward"] = [rep(dm_, "d_msa"), rep(dz_, "d_pair")]
    out["reproduced"] = not (torch.isfinite(dm_).all() and torch.isfinite(dz_).all())
    print("backward:", json.dumps(out["backward"]), flush=True)
    print("REPRODUCED" if out["reproduced"] else "did NOT reproduce", flush=True)
    if args.depths:
        # The forward is finite at full depth, so a depth sweep on the SAME inputs says
        # where in the 48-block chain the backward first goes bad. Each depth reseeds the
        # same cotangents at that depth's own roots.
        sweep = []
        for k in [int(x) for x in args.depths.split(",") if x.strip()]:
            ml2, zl2 = dev.leaf(m), dev.leaf(p_)
            with dev.tt.tape():
                mo2, zo2 = dev.stack(ml2, zl2, 0, k, ckpt=True,
                                     msa_mask=mask_t() if FRESH else mask)
            dev.sync()
            f_ok = bool(torch.isfinite(dev.down(mo2.value, tuple(m.shape))).all()
                        and torch.isfinite(dev.down(zo2.value, tuple(p_.shape))).all())
            dev.ag.backward([mo2, zo2], [dev.seed(gm, mo2), dev.seed(gz, zo2)])
            dev.sync()
            a = dev.grad(ml2, tuple(m.shape)); b = dev.grad(zl2, tuple(p_.shape))
            dev.ag.release_pins()
            fin_a, fin_b = torch.isfinite(a), torch.isfinite(b)
            row = {"k_evo": k, "forward_finite": f_ok,
                   "d_msa_nonfinite": int((~fin_a).sum()), "d_pair_nonfinite": int((~fin_b).sum()),
                   "d_msa_absmax": float(a[fin_a].abs().max()) if fin_a.any() else None,
                   "d_pair_absmax": float(b[fin_b].abs().max()) if fin_b.any() else None}
            sweep.append(row)
            print("depth", json.dumps(row), flush=True)
            del ml2, zl2, mo2, zo2
        out["depth_sweep"] = sweep
        out["first_bad_depth"] = next((r["k_evo"] for r in sweep
                                       if r["d_msa_nonfinite"] or r["d_pair_nonfinite"]), None)
        print("first bad depth:", out["first_bad_depth"], flush=True)
    out["stamp"] = A.stamp(int(__import__("os").environ.get("TT_VISIBLE_DEVICES", -1)))
    (HERE / "replay_nan.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
