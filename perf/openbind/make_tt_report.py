#!/usr/bin/env python3
"""Join perf/openbind/tt_results/*.json against gpu_reference.json and print the bar table.

The bar is `tt_device_s <= 4 * h200_device_s` at matched diffusion-sample count, on the
matched input (the cell name and the input sha256 both have to line up, or the row is
void). `x_h200` is tt_device_s / h200_device_s, so the bar is x_h200 <= 4.00.
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "perf/openbind/tt_results")
    ap.add_argument("--gpu", type=Path, default=ROOT / "perf/openbind/gpu_reference.json")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    g = json.load(open(a.gpu))
    gpu = {c["cell"]: c for c in g["results"]["ob"]["cells"]}
    gpu_confirm = {c["cell"]: c for c in g["results"].get("confirm-ob", {}).get("cells", [])}
    p2 = {r["cell"]: r for r in g["delta_ob_vs_p2"]["rows"]}
    sha = g["input_sha256"]

    tt = {}
    for p in sorted(a.results.glob("*.json")):
        d = json.load(open(p))
        if not d.get("device_s_median"):
            continue
        tt[(d["model"], d["input"], d["samples"])] = d

    rows = []
    for (model, _inp, _s), d in sorted(tt.items()):
        stem = d["input"].replace(".tt.yaml", "")
        cell = f"{stem}_s{d['samples']}"
        ok_input = (sha.get(d["input"]) == d["input_sha256"])
        gc = gpu.get(cell)
        row = {"model": model, "cell": cell, "samples": d["samples"],
               "n_tokens": d["runs"][0]["n_tokens"], "n_atoms": d["runs"][0]["n_atoms"],
               "tt_device_s": d["device_s_median"], "tt_fold_s": d["fold_s_median"],
               "tt_spread_pct": d["device_spread_pct"], "card": d["card"],
               "input_sha_ok": ok_input, "cif_reproducible": d["cif_reproducible"],
               "plddt": d["runs"][0]["plddt"]}
        if gc:
            row["h200_device_s"] = gc["h200_device_s"]
            row["tt_target_device_s"] = gc["tt_target_device_s"]
            row["x_h200"] = round(d["device_s_median"] / gc["h200_device_s"], 3)
            row["passes_4x"] = row["x_h200"] <= 4.0
            row["h200_trunk_s"] = gc.get("trunk_s")
            row["h200_rollout_s"] = gc.get("rollout_s")
        if cell in gpu_confirm:
            row["h200_confirm_device_s"] = gpu_confirm[cell]["h200_device_s"]
            row["x_h200_confirm"] = round(d["device_s_median"] / gpu_confirm[cell]["h200_device_s"], 3)
        if cell in p2:
            row["h200_p2_device_s"] = p2[cell]["p2_device_s"]
        rows.append(row)

    ob = {r["cell"]: r for r in rows if r["model"] == "openbind"}
    of3 = {r["cell"]: r for r in rows if r["model"] == "openfold3"}
    delta = []
    for cell in sorted(set(ob) & set(of3)):
        d_ob, d_p2 = ob[cell]["tt_device_s"], of3[cell]["tt_device_s"]
        delta.append({"cell": cell, "tt_ob_device_s": d_ob, "tt_p2_device_s": d_p2,
                      "tt_ob_speedup_x": round(d_p2 / d_ob, 4),
                      "gpu_ob_speedup_x": p2[cell]["ob_speedup_x"] if cell in p2 else None,
                      "tt_ob_spread_pct": ob[cell]["tt_spread_pct"],
                      "tt_p2_spread_pct": of3[cell]["tt_spread_pct"]})

    out = {"rows": rows, "tt_ob_vs_p2": delta,
           "bar": "tt_device_s <= 4 * h200_device_s at matched diffusion_samples"}
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))

    nan = float("nan")
    print(f"{'cell':22s} {'model':10s} {'tok':>5s} {'atoms':>6s} "
          f"{'tt_dev':>8s} {'spr%':>6s} {'h200':>7s} {'x_h200':>7s} {'4x?':>5s} {'card':>4s}")
    for r in rows:
        print(f"{r['cell']:22s} {r['model']:10s} {r['n_tokens']:5d} {r['n_atoms']:6d} "
              f"{r['tt_device_s']:8.3f} {r['tt_spread_pct'] or 0:6.2f} "
              f"{r.get('h200_device_s', nan):7.3f} {r.get('x_h200', nan):7.3f} "
              f"{('PASS' if r.get('passes_4x') else 'FAIL'):>5s} {str(r['card'] or '-'):>4s}")
    if delta:
        print(f"\n{'cell':22s} {'tt_ob':>8s} {'tt_p2':>8s} {'tt_x':>7s} {'gpu_x':>7s}")
        for d in delta:
            print(f"{d['cell']:22s} {d['tt_ob_device_s']:8.3f} {d['tt_p2_device_s']:8.3f} "
                  f"{d['tt_ob_speedup_x']:7.4f} {(d['gpu_ob_speedup_x'] or nan):7.4f}")
    bad = [r["cell"] for r in rows if not r["input_sha_ok"]]
    if bad:
        print("\nVOID (input sha256 mismatch):", bad)


if __name__ == "__main__":
    main()
