#!/usr/bin/env python3
"""The floor under the trunk-forward comparison: upstream's OWN 48-block forward, re-run.

Arm 1's reading only means something beside two numbers this instrument can produce itself:

  float64 re-run   upstream's own blocks, float64, from the captured block-0 input. Anything
                   this reads against the captured block-47 output is the INSTRUMENT: the
                   crop-64 convention, the replay, the re-composition. It is the floor.
  bf16 autocast    upstream's own blocks under `torch.autocast(bfloat16)` with float32
                   parameters, the recipe `bundle_min.py --dtype float32 --autocast bf16` runs
                   at model scope. This is what a bf16 arithmetic disagreement looks like when
                   the composition is upstream's own.

`of3t-pairformer`'s `upstream_arm.py` already published both, UNMASKED: float64 z 3.686695e-03,
bf16 z 4.443473e-03. Unmasked is the wrong denominator here -- the batch has 56 real tokens in a
64-wide window and the device arms are read masked -- so this recomputes the same two arms with
the masked metric, and the norm ratio and cosine beside it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
PRE = "pairformer_stack.blocks."


def metrics(ours, ref):
    a = ours.to(torch.float64).reshape(-1)
    b = ref.to(torch.float64).reshape(-1)
    na = float(torch.linalg.vector_norm(a))
    nb = float(torch.linalg.vector_norm(b))
    den = nb if nb > 0 else 1e-300
    return {"rel_l2": float(torch.linalg.vector_norm(a - b) / den),
            "norm_ratio": na / den,
            "cos": float((a @ b) / ((na * nb) or 1e-300)),
            "ours_norm": na, "ref_norm": nb}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", default="/home/ttuser/of3t_gradients/cap")
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--capall",
                    default="/home/ttuser/of3t_trunkfwd/capall/blocks_all_crop64.pt")
    ap.add_argument("--out", default="perf/of3t_trunkfwd/UPSTREAM_FLOOR.json")
    a = ap.parse_args()
    t0 = time.perf_counter()

    sys.path.insert(0, "/home/ttuser/of3t_pairformer")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "of3t_pairformer"))
    from upstream_arm import build  # upstream's own PairFormerBlock, their load path

    cap0 = torch.load(os.path.join(a.cap, "block0_boundary.pt"), map_location="cpu",
                      weights_only=False)
    last = a.blocks - 1
    capL = torch.load(os.path.join(a.cap, f"block{last}_boundary.pt"), map_location="cpu",
                      weights_only=False)
    c = a.crop
    s_in = cap0["args"][0][:, :c].contiguous()
    z_in = cap0["args"][1][:, :c, :c].contiguous()
    sm = cap0["kwargs"]["single_mask"][:, :c].contiguous()
    pm = cap0["kwargs"]["pair_mask"][:, :c, :c].contiguous()
    s_ref = capL["out"][0][:, :c].to(torch.float64).contiguous()
    z_ref = capL["out"][1][:, :c, :c].to(torch.float64).contiguous()
    N = int(z_in.shape[1])
    rep = {"what": __doc__.strip().splitlines()[0], "crop": c, "blocks": a.blocks,
           "tokens": N, "real_tokens": int(sm.sum()),
           "reference": "the captured block-47 output of upstream's own float64 step"}
    print(f"[{time.perf_counter()-t0:.0f}s] boundary N={N} real={int(sm.sum())}", flush=True)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    msk = sm.to(torch.float64).reshape(1, N, 1)
    pmk = pm.to(torch.float64).reshape(1, N, N, 1)

    for name, dt, ac in (("float64", torch.float64, False),
                         ("float32", torch.float32, False),
                         ("bf16_autocast", torch.float32, True)):
        mods, dims, load = build(sd, 0, a.blocks, dt)
        s = s_in.to(dt).contiguous()
        z = z_in.to(dt).contiguous()
        smd, pmd = sm.to(dt), pm.to(dt)
        ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if ac
               else torch.autocast("cpu", enabled=False))
        t1 = time.perf_counter()
        with ctx, torch.no_grad():
            for m in mods:
                s, z = m(s, z, smd, pmd)
        s, z = s.to(torch.float64), z.to(torch.float64)
        rep[name] = {"dtype": str(dt), "autocast_bf16": ac,
                     "seconds": time.perf_counter() - t1,
                     "s": metrics(s, s_ref), "z": metrics(z, z_ref),
                     "s_masked": metrics(s * msk, s_ref * msk),
                     "z_masked": metrics(z * pmk, z_ref * pmk)}
        print(f"[{time.perf_counter()-t0:.0f}s] {name:<14} "
              f"s_masked {rep[name]['s_masked']['rel_l2']:.6e}  "
              f"z_masked {rep[name]['z_masked']['rel_l2']:.6e}  "
              f"(unmasked s {rep[name]['s']['rel_l2']:.6e} z {rep[name]['z']['rel_l2']:.6e})",
              flush=True)
        del mods, s, z

    # ---- the PER-BLOCK bar -------------------------------------------------------------------
    # Arm 3 reads our per-block forward when block k is fed THEIR captured input. That figure is
    # unreadable without the same quantity for upstream's own bf16, and dividing the 48-block
    # bf16 figure by 48 would assume the accumulation is linear -- which is the thing arm 3 is
    # measuring, so it cannot be assumed on the way in.
    if os.path.isfile(a.capall):
        allb = torch.load(a.capall, map_location="cpu", weights_only=False)
        if int(allb["crop"]) != c:
            raise SystemExit(f"capall crop {allb['crop']} != --crop {c}")
        for name, dt, ac in (("per_block_float64", torch.float64, False),
                             ("per_block_bf16_autocast", torch.float32, True)):
            mods, _, _ = build(sd, 0, a.blocks, dt)
            ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if ac
                   else torch.autocast("cpu", enabled=False))
            rows = []
            for k in range(a.blocks):
                sk = allb["s_in"][k].to(dt).contiguous()
                zk = allb["z_in"][k].to(dt).contiguous()
                with ctx, torch.no_grad():
                    so, zo = mods[k](sk, zk, sm.to(dt), pm.to(dt))
                so, zo = so.to(torch.float64), zo.to(torch.float64)
                sr = allb["s_out"][k].to(torch.float64)
                zr = allb["z_out"][k].to(torch.float64)
                rows.append({"block": k,
                             "s_masked": metrics(so * msk, sr * msk),
                             "z_masked": metrics(zo * pmk, zr * pmk)})
            zs = sorted(r["z_masked"]["rel_l2"] for r in rows)
            ss = sorted(r["s_masked"]["rel_l2"] for r in rows)
            rep[name] = {"per_block": rows,
                         "median_z_masked": zs[len(zs) // 2], "worst_z_masked": zs[-1],
                         "median_s_masked": ss[len(ss) // 2], "worst_s_masked": ss[-1]}
            print(f"[{time.perf_counter()-t0:.0f}s] {name:<24} median z "
                  f"{zs[len(zs)//2]:.6e} worst {zs[-1]:.6e} | median s "
                  f"{ss[len(ss)//2]:.6e} worst {ss[-1]:.6e}", flush=True)
            del mods
    else:
        rep["per_block_bf16_autocast"] = {"ran": False, "why": f"{a.capall} absent"}

    rep["reading"] = {
        "instrument_floor_z_masked": rep["float64"]["z_masked"]["rel_l2"],
        "upstream_bf16_z_masked": rep["bf16_autocast"]["z_masked"]["rel_l2"],
        "why": "float64 is what re-composing 48 blocks from the captured input costs with "
               "upstream's own arithmetic: the instrument's floor. bf16_autocast is what a "
               "bf16 disagreement looks like when the composition is theirs."}
    rep["seconds"] = time.perf_counter() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(f"[{time.perf_counter()-t0:.0f}s] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
