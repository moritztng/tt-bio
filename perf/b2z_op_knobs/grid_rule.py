#!/usr/bin/env python3
"""Is the core_grid win a rule, and does it cost accuracy?

The knob sweep says passing `core_grid` to a matmul is worth 1.7x-9.1x when the matmul is
batched and ~1.06x when it is not. Two things have to be true before that can become a
derivation in the engine's config resolver:

  1. the win has to be a function of the shape, not of the particular grid guessed. This runs a
     grid ladder per instance, including the device's OWN full grid, which the sweep never tried.
  2. the arm has to be no less accurate than the incumbent. Comparing an arm to the incumbent
     only says they differ. This compares BOTH against a float32 torch reference, so a knob that
     is merely reordering accumulation is told apart from one that is losing bits.
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
import op_replay as R  # noqa: E402


def torch_ref(torch, rec, ins_t):
    """float32 CPU reference for the instance, or None if we do not model this op."""
    if rec["kind"] != "matmul":
        return None
    a, b = ins_t[0].float(), ins_t[1].float()
    out = a @ b
    if rec.get("has_bias") and len(ins_t) > 2:
        out = out + ins_t[2].float()
    return out


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
    rows = [r for r in rows[: a.top] if r["kind"] == "matmul"]

    device = R.open_device_retry(ttnn)
    g = device.compute_with_storage_grid_size()
    gx, gy = int(g.x), int(g.y)
    ladder = [f"{gx}x{gy}", f"{gx}x{gy-1}", "8x8", "8x4", "4x8", "8x2", "2x8", "4x4", "1x1"]
    seen, ladder = set(), [x for x in ladder if not (x in seen or seen.add(x))]
    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "grid": f"{gx}x{gy}",
                   "ladder": ladder}, "instances": []}
    print(f"# device grid {gx}x{gy}, ladder {ladder}", flush=True)

    for br in rows:
        rec = recs[br["id"]]
        batch = 1
        for d in rec["inputs"][0]["shape"][:-2]:
            batch *= int(d)
        b2 = len(rec["inputs"][1]["shape"])
        e = {"id": br["id"], "shapes": br["shapes"], "ms_per_fold": br["ms_per_fold"],
             "batch": batch, "rank_in1": b2, "arms": []}
        print(f"== {br['id']} {'|'.join(br['shapes'])} batch={batch} "
              f"({br['ms_per_fold']:.0f} ms/fold)", flush=True)

        # reference, from the same seeded operands every arm sees
        torch.manual_seed(0)
        ref = None
        try:
            ins_t = []
            for s in rec["inputs"]:
                shape = [int(x) for x in s["shape"]]
                ins_t.append((torch.randn(shape, dtype=torch.float32) * 0.1).bfloat16())
            ref = torch_ref(torch, rec, ins_t)
        except Exception as ex:                                          # noqa: BLE001
            e["ref_error"] = str(ex)[:120]

        incs = []
        for knob in [""] + ladder:
            row = {"knob": knob or "(default)"}
            try:
                from sweep import run_one
                us, _, o = run_one(ttnn, torch, rec, device, knob, a.reps, a.bursts, True)
                row["us"] = round(us, 2)
                if knob == "":
                    incs.append(us)
                else:
                    inc2, _, _ = run_one(ttnn, torch, rec, device, "", a.reps, a.bursts, False)
                    incs.append(inc2)
                    row["ratio"] = round(st.median(incs[-2:]) / us, 4)
                if ref is not None and o is not None and tuple(o.shape) == tuple(ref.shape):
                    d = (o - ref).abs()
                    row["rel_err_vs_fp32"] = round(float(d.max() / ref.abs().max()), 6)
                    row["rms_err_vs_fp32"] = round(float((d ** 2).mean().sqrt()), 6)
            except Exception as ex:                                      # noqa: BLE001
                row["error"] = f"{type(ex).__name__}: {str(ex)[:140]}"
            e["arms"].append(row)
            msg = row.get("error") or (
                f"{row.get('ratio', 1.0):6.3f}x  {row['us']:9.2f} us"
                + (f"  rel_err {row['rel_err_vs_fp32']:.2e}" if "rel_err_vs_fp32" in row else ""))
            print(f"   {row['knob']:12s} {msg}", flush=True)
            json.dump(out | {"instances": out["instances"] + [e]}, open(a.out, "w"), indent=1)
        out["instances"].append(e)
        json.dump(out, open(a.out, "w"), indent=1)

    ttnn.close_device(device)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
