#!/usr/bin/env python3
"""bcx-large: the largest AF2 design gradient step that completes on one Blackhole.

The step is `bcx-ckpt`'s: sequence logits -> AF2 embedding (host, torch autograd) -> 4 extra-MSA
+ 48 Evoformer blocks on the device, taped and checkpointed per block -> cotangent seeds on
both outputs -> backward through every block -> logit gradient on the host. Levers as the
`stack` arm, the program `bcx-ckpt` measured. Nothing is shortened: every block runs.

One size per process, so an allocation refusal at one size cannot colour the next. The
allocator reports what is allocated now, not a high-water mark, so DRAM is sampled after
every block forward and after every checkpoint recompute (inside the backward, where a
block's tape is at its largest), then after the backward. The recorded peak is therefore a
lower bound on the true instantaneous peak; the free headroom at that sample says how close.
Rep 0 pays the JIT compile; later reps are warm.

  TT_VISIBLE_DEVICES=2 python3 perf/bcx_large/ladder.py --n 512
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import resource
import sys
import time
import traceback

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_afgrad import afgrad as A  # noqa: E402
from perf.bcx_stack import stack as S  # noqa: E402

OUT = ROOT / "perf" / "bcx_large"
GB = 1e9


def rss_peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--params", default=A.DEFAULT_PARAMS)
    ap.add_argument("--extra", type=int, default=4)
    ap.add_argument("--evo", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="stack")
    ap.add_argument("--reps", type=int, default=2, help="rep 0 is cold (JIT compile), later reps warm")
    ap.add_argument("--save-grad", action="store_true", help="keep the logit gradient (.pt)")
    args = ap.parse_args()
    n = args.n
    rec = {"n": n, "k_extra": args.extra, "k_evo": args.evo, "ckpt": True, "arm": args.arm,
           "seed": args.seed, "stamp": A.stamp(os.environ.get("TT_VISIBLE_DEVICES", "?")),
           "pci": S.sysfs_node()[1], "completed": False, "reps": []}
    rec["stamp"].pop("subsystem_device", None)     # afgrad reads the naive node, wrong on qb1
    out = OUT / f"ladder_n{n}.json"
    OUT.mkdir(parents=True, exist_ok=True)

    def save():
        rec["host_rss_peak_gb"] = rss_peak_gb()
        out.write_text(json.dumps(rec, indent=1, default=str))

    t_open = time.time()
    lv = S.Levers()
    dm, ref = A.load_models(args.params)
    dev = A.Dev(dm)
    lv.arm(args.arm)
    ttnn, ag = dev.ttnn, dev.ag
    rec["open_and_load_s"] = time.time() - t_open
    rec["host_rss_after_load_gb"] = rss_peak_gb()

    def view():
        mv = ttnn.get_memory_view(dev.device, ttnn.BufferType.DRAM)
        nb = int(mv.num_banks)
        return (int(mv.total_bytes_allocated_per_bank) * nb, int(mv.total_bytes_free_per_bank) * nb,
                int(mv.total_bytes_per_bank) * nb, int(mv.largest_contiguous_bytes_free_per_bank))

    base, _, total, _ = view()
    rec["dram_total_gb"], rec["dram_weights_gb"] = total / GB, base / GB
    st = {"phase": "fwd", "peak": 0, "trace": []}

    def sample(where):
        used, free, _, lcf = view()
        st["trace"].append((st["phase"], where, round(used / GB, 4), round(free / GB, 4)))
        if used > st["peak"]:
            st.update(peak=used, peak_at=(st["phase"], where), peak_free_gb=free / GB,
                      peak_largest_free_per_bank_gb=lcf / GB)

    ex, ev = dev.extra, dev.evo

    def extra(i, z):
        r = ex(i, z)
        sample(f"extra{i}")
        return r

    def evo(i, m, z):
        r = ev(i, m, z)
        sample(f"evo{i}")
        return r

    dev.extra, dev.evo = extra, evo

    def peak_of():
        return {"gb": st["peak"] / GB, "at": st.get("peak_at"), "free_gb": st.get("peak_free_gb"),
                "largest_free_per_bank_gb": st.get("peak_largest_free_per_bank_gb")}

    torch.manual_seed(args.seed)                  # `whole`'s draws, in its order
    ridx = torch.arange(n)
    logits = torch.randn(n, 20) * 2.0
    wm = torch.randn(1, n, 256, dtype=torch.float64) / (n * 256) ** 0.5
    wz = torch.randn(n, n, 128, dtype=torch.float64) / (n * n * 128) ** 0.5
    clock = S.Clock()
    try:
        for rep in range(args.reps):
            st.update(phase="fwd", peak=0, trace=[])
            gc.collect()
            lgt = logits.clone().float().requires_grad_(True)
            m0, z0 = A.embed(ref["bf16"], lgt, ridx)
            ml, zl = dev.leaf(m0), dev.leaf(z0)
            dev.sync()
            t0 = time.time()
            with dev.tt.tape():
                mo, zo = dev.stack(ml, zl, args.extra, args.evo, ckpt=True)
            dev.sync()
            t1 = time.time()
            sample("end of forward")
            after_fwd = view()[0] / GB
            seeds = [dev.seed(wm, mo), dev.seed(wz, zo)]
            st["phase"] = "bwd"
            t2 = time.time()
            ag.backward([mo, zo], seeds)
            dev.sync()
            t3 = time.time()
            sample("end of backward")
            gm0, gz0 = dev.grad(ml, m0.shape), dev.grad(zl, z0.shape)
            out_finite = {"m": bool(torch.isfinite(dev.down(mo.value, m0.shape)).all()),
                          "z": bool(torch.isfinite(dev.down(zo.value, z0.shape)).all())}
            torch.autograd.backward([m0, z0], [gm0.to(m0.dtype), gz0.to(z0.dtype)])
            ag.release_pins()
            g = lgt.grad.double()
            r = {"rep": rep, "fwd_s": t1 - t0, "bwd_s": t3 - t2, "step_s": t3 - t0,
                 "aiclk": clock.window([(t0, t3)]), "load1_end": os.getloadavg()[0],
                 "dram_after_fwd_gb": after_fwd, "peak": peak_of(),
                 "grad": {"finite": bool(torch.isfinite(g).all()), "norm": float(g.norm()),
                          "absmax": float(g.abs().max()),
                          "nonzero_frac": float((g != 0).double().mean()),
                          "gm0_finite": bool(torch.isfinite(gm0).all()),
                          "gz0_finite": bool(torch.isfinite(gz0).all()),
                          "gz0_norm": float(gz0.double().norm())},
                 "out_finite": out_finite}
            if rec["reps"]:
                r["grad_bit_identical_to_rep0"] = bool(torch.equal(g, rec["_g0"]))
            else:
                rec["_g0"] = g
            if args.save_grad and rep == 0:
                torch.save({"logits": logits, "grad": g}, OUT / f"grad_n{n}.pt")
            del mo, zo, ml, zl, seeds, m0, z0, lgt, gm0, gz0
            gc.collect()
            r["dram_after_release_gb"] = view()[0] / GB
            r["trace"] = list(st["trace"])
            rec["reps"].append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "trace"}, default=str), flush=True)
            save()
        rec["completed"] = True
    except Exception as e:                         # an allocation refusal is the answer
        rec["error"] = str(e)[:2000]
        rec["error_type"] = type(e).__name__
        rec["frames"] = [f"{f.filename.split(chr(47))[-1]}:{f.lineno} {f.name}"
                         for f in traceback.extract_tb(e.__traceback__)]
        rec["traceback_tail"] = traceback.format_exc()[:3000]
        rec["failed_rep"] = {"rep": len(rec["reps"]), "phase": st["phase"], "peak": peak_of(),
                             "trace": list(st["trace"]), "load1": os.getloadavg()[0],
                             "aiclk": clock.window([(t_open, time.time())])}
    finally:
        clock.stop()
        rec.pop("_g0", None)
        save()
    print(json.dumps({k: rec.get(k) for k in ("n", "completed", "host_rss_peak_gb", "error_type",
                                              "error")}, default=str)[:1500], flush=True)
    return 0 if rec["completed"] else 3


if __name__ == "__main__":
    sys.exit(main())
