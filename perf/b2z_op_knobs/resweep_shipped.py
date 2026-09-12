#!/usr/bin/env python3
"""Re-sweep the top matmul instances against the incumbent the fold ACTUALLY dispatches.

The first sweep (`sweep.py` -> `results/sweep_wh_c1_base.json`) built its incumbent as a bare
`ttnn.linear(a, b)`. The engine never makes that call: every matmul call site on Boltz-2's hot
path already passes `core_grid=CORE_GRID_MAIN` or a hand-tuned program config. So the 1.7x-7.2x
`grid=` column in that table is the distance between ttnn's default resolver and a grid the fold
already uses, and none of it is available.

This run fixes the incumbent (`op_replay.shipped_matmul_config`) and keeps one arm, `nogrid=1`,
which reinstates the straw man on purpose so the size of the correction is in the table rather
than only in a paragraph.

Every arm is also scored against a float32 torch reference computed from the same seeded
operands, so a knob that merely reorders accumulation is told apart from one that loses bits --
comparing an arm to the incumbent alone only says the two differ.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import op_replay as R                                                     # noqa: E402
from sweep import run_one                                                 # noqa: E402

ARMS = [
    "nogrid=1",                     # the straw man, on purpose
    "fp32acc=0",
    "fidelity=HiFi2",
    "fidelity=HiFi2,fp32acc=0",
    "fidelity=LoFi,fp32acc=0",
    "packerl1=0",
    "dstfull=1",
    "fp32acc=0,dstfull=1",
    "grid=8x8",
    "grid=8x4",
]


def torch_ref(torch, rec, seed=0):
    """float32 CPU reference from the same seeded operands every arm sees."""
    if rec["kind"] != "matmul":
        return None
    torch.manual_seed(seed)
    ts = []
    for s in rec["inputs"]:
        shape = [int(x) for x in s["shape"]]
        if s["dtype"] in ("uint32", "int32", "uint16"):
            ts.append(torch.zeros(shape, dtype=torch.int32))
        else:
            ts.append((torch.randn(shape, dtype=torch.float32) * 0.1))
    a = ts[0].bfloat16().float()
    b = ts[1].bfloat16().float()
    out = a @ b
    if rec.get("has_bias") and len(ts) > 2:
        out = out + ts[2].bfloat16().float()
    return out


def err_vs(o, ref):
    if o is None or ref is None or tuple(o.shape) != tuple(ref.shape):
        return {}
    d = (o.float() - ref).abs()
    scale = float(ref.abs().max()) or 1.0
    return {"rel_max_err": round(float(d.max()) / scale, 6),
            "rms_err": round(float((d ** 2).mean().sqrt()), 6)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "bench_manifest.json"))
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--bursts", type=int, default=5)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    R._init_tables(ttnn)

    recs = {r["id"]: r for r in json.load(open(a.manifest))}
    rows = [r for r in json.load(open(a.baseline))["rows"] if "ms_per_fold" in r]
    rows.sort(key=lambda r: -r["ms_per_fold"])
    rows = [r for r in rows[: a.top] if recs[r["id"]]["kind"] == "matmul"]

    device = R.open_device_retry(ttnn)
    g = device.compute_with_storage_grid_size()
    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "grid": f"{g.x}x{g.y}",
                   "arch": str(device.arch()), "reps": a.reps, "bursts": a.bursts,
                   "incumbent": "shipped (core_grid where the engine passes one)"},
           "instances": []}
    print(f"# grid {g.x}x{g.y}  arms={len(ARMS)}  instances={len(rows)}", flush=True)

    for br in rows:
        rec = recs[br["id"]]
        ship = R.shipped_matmul_config(rec)
        e = {"id": br["id"], "shapes": br["shapes"], "ms_per_fold": br["ms_per_fold"],
             "unit_path": rec["unit_path"], "api": rec["api"], "shipped_config": ship, "arms": []}
        print(f"== {br['id']} {'|'.join(br['shapes'])} shipped={ship} "
              f"({br['ms_per_fold']:.0f} ms/fold replayed)", flush=True)
        ref = torch_ref(torch, rec)

        incs, base_out = [], None
        try:
            us, _, o = run_one(ttnn, torch, rec, device, "", a.reps, a.bursts, True)
            incs.append(us)
            base_out = o
            e["incumbent_us"] = round(us, 2)
            e.update({f"incumbent_{k}": v for k, v in err_vs(o, ref).items()})
            print(f"   {'(incumbent)':26s} {us:9.2f} us"
                  + (f"  rel {e.get('incumbent_rel_max_err')}" if ref is not None else ""),
                  flush=True)
        except Exception as ex:                                           # noqa: BLE001
            e["error"] = f"{type(ex).__name__}: {str(ex)[:160]}"
            print(f"   incumbent FAILED: {e['error']}", flush=True)
            out["instances"].append(e)
            json.dump(out, open(a.out, "w"), indent=1)
            continue

        for knob in ARMS:
            row = {"knob": knob}
            try:
                us, _, o = run_one(ttnn, torch, rec, device, knob, a.reps, a.bursts, True)
                inc2, _, _ = run_one(ttnn, torch, rec, device, "", a.reps, a.bursts, False)
                incs.append(inc2)
                row["us"] = round(us, 2)
                row["ratio"] = round(st.median(incs[-2:]) / us, 4)
                row.update(err_vs(o, ref))
                if base_out is not None and o is not None \
                        and tuple(o.shape) == tuple(base_out.shape):
                    d = (o.float() - base_out.float()).abs()
                    row["max_abs_vs_incumbent"] = float(f"{float(d.max()):.3e}")
                    row["bit_exact"] = bool(float(d.max()) == 0.0)
            except Exception as ex:                                       # noqa: BLE001
                row["error"] = f"{type(ex).__name__}: {str(ex)[:160]}"
            e["arms"].append(row)
            msg = row.get("error") or (
                f"{row['ratio']:6.3f}x {row['us']:9.2f} us"
                + (f"  rel {row.get('rel_max_err')}" if "rel_max_err" in row else "")
                + ("  BITEX" if row.get("bit_exact") else ""))
            print(f"   {row['knob']:26s} {msg}", flush=True)
            json.dump(out | {"instances": out["instances"] + [e]}, open(a.out, "w"), indent=1)

        e["aa_floor"] = round(max(incs) / min(incs), 4)
        print(f"   {'A/A floor':26s} {e['aa_floor']:6.4f}", flush=True)
        out["instances"].append(e)
        json.dump(out, open(a.out, "w"), indent=1)

    ttnn.close_device(device)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
