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
    ap.add_argument("--per-block", action="store_true",
                    help="gradient magnitude after each block, to separate growth from one bad op")
    ap.add_argument("--k-evo", type=int, default=48)
    args = ap.parse_args()
    z = np.load(args.npz)
    print({k: (z[k].shape if z[k].ndim else z[k].item()) for k in z.files}, flush=True)

    lv = S.Levers(); dm, _ = A.load_models(A.DEFAULT_PARAMS)
    dev = A.Dev(dm.to_device()); lv.arm("stack")
    m = torch.from_numpy(z["msa_leaf"]).float()
    p_ = torch.from_numpy(z["pair_leaf"]).float()
    mask = dev.up(torch.from_numpy(z["mask"]).float())
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
    out["stamp"] = A.stamp(int(__import__("os").environ.get("TT_VISIBLE_DEVICES", -1)))
    (HERE / "replay_nan.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
