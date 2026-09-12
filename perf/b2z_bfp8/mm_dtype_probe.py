#!/usr/bin/env python3
"""Does `ttnn.experimental.minimal_matmul` actually serve a bfloat8_b pair projection, and at what
price?

`_pair_proj_minimal_matmul` refuses any operand that is not bf16 (`tenstorrent.py:3819`). The
predecessor read that refusal as a kernel limitation and priced widening it as tt-metal work. The
shipped 0.68.0 docstring says otherwise: the kernel takes BFLOAT16, BFLOAT8_B and BFLOAT4_B, and
requires only that the weight's dtype MATCHES the input's. So the widening is Python, and the one
real cost is that serving a bfloat8_b activation needs a bfloat8_b copy of the weight.

Three questions, and all three have to be answered before the widening is worth writing:

  1. does the matched bfp8_b call run at all, and is it faster than the bf16 call on the same shape
  2. does an UNMATCHED call (bf16 weight, bfp8_b activation) throw, or silently upcast -- a silent
     upcast would make the `w.dtype` half of the gate free to drop, and a throw makes the weight
     copy mandatory
  3. what does the narrower operand cost in accuracy at the op level, against the bf16 call

The pair projections at 512 aa: [1, 512, 512, c] x [c, c] with kt == 8, which is c_z = 256 after
the trimul's 2x fuse and c_z = 128 for the plain ones.
"""
from __future__ import annotations

import argparse, json, os, socket, statistics as st, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prodcfg                                                              # noqa: E402
from prodcfg import REPO, assert_checkout                                   # noqa: E402
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--reps", type=int, default=9)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio import tenstorrent as T

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    out: dict = {"env": {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "grid": [g.x, g.y], "arch": str(dev.arch()), "seq": args.seq,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tt_bio": assert_checkout(),
    }}
    try:
        import importlib.metadata as _md
        out["env"]["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    out["docstring"] = (ttnn.experimental.minimal_matmul.__doc__ or "")[:4000]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    S = args.seq
    # The pair projections run under `TorchWrapper.compute_kernel_config`: HiFi4, no approx,
    # fp32 accumulate, packer L1 acc. The probe has to use the same one or it prices a
    # different kernel than the model's.
    kcls = (ttnn.types.WormholeComputeKernelConfig
            if dev.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    ckc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)
    b8, bf = ttnn.bfloat8_b, ttnn.bfloat16

    def tt(x, dt):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)

    def timed(fn, reps):
        fn()                                              # compile
        ttnn.synchronize_device(dev)
        walls = []
        for _ in range(reps):
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            r = fn()
            ttnn.synchronize_device(dev)
            walls.append(time.perf_counter() - t)
            ttnn.deallocate(r)
        return st.median(walls), walls

    cases = []
    for c in (128, 256):
        torch.manual_seed(0)
        xt = (torch.randn(1, S, S, c) * 0.35)
        wt = (torch.randn(c, c) * (c ** -0.5))
        ref = (xt.to(torch.float64) @ wt.to(torch.float64))[0]

        cfg_probe = {}
        for xd, wd, lab in ((bf, bf, "bf16"), (b8, b8, "bfp8_b"), (b8, bf, "mixed_x8_wbf16")):
            x, w = tt(xt, xd), tt(wt, wd)
            cfg = T._qkv_mm_config(x, w)
            rec = {"case": f"c{c}", "x_dtype": str(xd), "w_dtype": str(wd), "label": lab,
                   "kt": -(-c // 32), "have_cfg": cfg is not None}
            if cfg is None:
                rec["skip"] = "no _qkv_mm_config for this shape"
                cases.append(rec); ttnn.deallocate(x); ttnn.deallocate(w); continue

            def call(x=x, w=w, cfg=cfg, xd=xd):
                return ttnn.experimental.minimal_matmul(
                    input_tensor=x, weight_tensor=w, bias_tensor=None,
                    compute_kernel_config=ckc, dtype=xd, config=cfg)
            try:
                med, walls = timed(call, args.reps)
                rec["ms"] = round(med * 1e3, 4)
                rec["ms_all"] = [round(v * 1e3, 4) for v in walls]
                y = ttnn.to_torch(call()).to(torch.float64)[0]
                err = (y - ref)
                rec["rmse_vs_fp64"] = float(err.pow(2).mean().sqrt())
                rec["rel_rmse"] = float(err.pow(2).mean().sqrt() / ref.std())
                rec["ok"] = True
            except Exception as e:
                rec["ok"] = False
                rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            ttnn.deallocate(x); ttnn.deallocate(w)
            cases.append(rec)
            out["cases"] = cases
            args.out.write_text(json.dumps(out, indent=1))
            print(json.dumps(rec)[:400], flush=True)
        cfg_probe.clear()

    by = {(c["case"], c["label"]): c for c in cases}
    out["verdict"] = {}
    for c in ("c128", "c256"):
        a, b = by.get((c, "bf16"), {}), by.get((c, "bfp8_b"), {})
        m = by.get((c, "mixed_x8_wbf16"), {})
        out["verdict"][c] = {
            "bfp8_b_runs": bool(b.get("ok")),
            "speedup_bf16_over_bfp8": (round(a["ms"] / b["ms"], 4)
                                       if a.get("ms") and b.get("ms") else None),
            "rel_rmse_bf16": a.get("rel_rmse"), "rel_rmse_bfp8": b.get("rel_rmse"),
            "mixed_operand_dtypes": "ran" if m.get("ok") else m.get("error", "n/a"),
        }
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
