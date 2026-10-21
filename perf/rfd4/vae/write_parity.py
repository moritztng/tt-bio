"""Turn a device_parity.py run and its CPU bf16-weight floor (cpu_parity.py --bf16-weights) into the
campaign's parity record, in the trunk row's shape: one entry per graded module and case, each with its
bf16 floor beside it.

    python3 perf/rfd4/vae/write_parity.py --device <dp.json> --floor <cpu_bf16_floor.json> \
        --out state/rfd4/parity/vae.json --row state/rfd4/rows/rfd4-vae.json [--clock-mhz N --clock-note ...]

The continuous-output bar is the campaign's 0.05 (the trunk row's). device_parity.py's own 2e-2 sits
barely above what bf16 weights alone cost: the float64 fold with only its matrices rounded to bf16
already reads up to ~0.022 on the real encoder batches.

The synthetic batches (random features, gen_goldens.py) are out of distribution: bf16 weights alone put
their encoder at 0.10 (n 250) and 0.15 (n 500). Those are graded on the floor only, as the trunk row
grades its random-weight deep chain: the device may sit at most FLOOR_RATIO x the floor. Real inputs
carry the absolute bar.
"""

import argparse
import json
import subprocess
from pathlib import Path

BAR = {"max_rel_err": 0.05, "coord_rmsd_A": 0.05, "argmax_agree": 0.99, "tv_mean": 1e-2,
       "d_rmsd_A": 0.02, "d_seq_recovery": -0.01}
FLOOR_RATIO = 2.0
CONT = [("encoder", "mean"), ("encoder", "log_scale"), ("encoder", "z_latent"),
        ("decoder", "seq_logits"), ("decoder", "presence")]


def get(d, path):
    for k in path:
        d = d[k]
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--floor", required=True, help="the grading floor: cpu_parity.py --mixed (upstream's bf16-mixed)")
    ap.add_argument("--weights-floor", default="", help="optional cpu_parity.py --bf16-weights, recorded beside it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--row", required=True)
    ap.add_argument("--branch", default="wk/rfd4-vae")
    ap.add_argument("--card", default="qb2 chip 3 p300c")
    a = ap.parse_args()
    dev, flo = json.loads(Path(a.device).read_text()), json.loads(Path(a.floor).read_text())
    wfl = json.loads(Path(a.weights_floor).read_text()) if a.weights_floor else None

    modules, clocks = [], []
    for case, c in dev["cases"].items():
        f = flo["cases"][case]
        shape = f"{c['batch']} x {c['n_tokens']} tokens (bucketed to 32), 768-d, 12 heads; released AE weights"
        for p in CONT:
            m = {"name": f"{case} {'.'.join(p)}", "shape": shape, "ref": "float64",
                 "rel_err": get(c, p + ("max_rel_err",)), "rel_l2": get(c, p + ("rel_l2",)),
                 "bf16_floor": get(f, p + ("max_rel_err",)), "test": "perf/rfd4/vae/device_parity.py"}
            if wfl:
                m["bf16_weight_floor"] = get(wfl["cases"][case], p + ("max_rel_err",))
            if case.startswith("synthetic"):
                m["graded_on"] = (f"floor only (random out-of-distribution features; bf16 weights alone read "
                                  f"{m['bf16_floor']:.3g}): rel_err <= {FLOOR_RATIO} x bf16_floor")
            modules.append(m)
        r, rf = c["decoder"]["residue"], f["decoder"]["residue"]
        modules.append({"name": f"{case} residue head (147-class)", "shape": shape, "ref": "float64",
                        "argmax_agree": r["argmax_agree"], "bf16_floor_argmax_agree": rf["argmax_agree"],
                        "disagree_ref_margin": r["disagree_ref_margin"], "tv_mean": r["tv_mean"],
                        "tv_max": r["tv_max"], "kl_mean": r["kl_mean"],
                        "final_residue_type_agree": r["final_residue_type_agree"],
                        "test": "perf/rfd4/vae/device_parity.py"})
        modules.append({"name": f"{case} decoded coords", "shape": shape, "ref": "float64",
                        "rmsd_A": c["decoder"]["coords"]["rmsd_A"], "max_dev_A": c["decoder"]["coords"]["max_dev_A"],
                        "bf16_floor_rmsd_A": f["decoder"]["coords"]["rmsd_A"],
                        "atom_mask_agree": c["decoder"]["atom_mask_agree"],
                        "test": "perf/rfd4/vae/device_parity.py"})
        rt = c["roundtrip"]
        modules.append({"name": f"{case} roundtrip TT encode -> TT decode", "shape": shape, "ref": "float64",
                        "rel_err": rt["seq_logits"]["max_rel_err"], "rmsd_A": rt["coords"]["rmsd_A"],
                        "argmax_agree": rt["residue"]["argmax_agree"], "tv_mean": rt["residue"]["tv_mean"],
                        "test": "perf/rfd4/vae/device_parity.py"})
        if "recon" in c:
            rc = c["recon"]
            modules.append({"name": f"{case} reconstruction of the input structure", "shape": shape,
                            "ref": "upstream's own reconstruction of the same input",
                            "upstream_rmsd_A": rc["upstream"]["all_atom_rmsd_A"],
                            "tt_rmsd_A": rc["tt_decode"]["all_atom_rmsd_A"],
                            "tt_roundtrip_rmsd_A": rc["tt_roundtrip"]["all_atom_rmsd_A"],
                            "upstream_seq_recovery": rc["upstream"]["seq_recovery"],
                            "tt_seq_recovery": rc["tt_decode"]["seq_recovery"],
                            "tt_roundtrip_seq_recovery": rc["tt_roundtrip"]["seq_recovery"],
                            "d_rmsd_A": rc["d_rmsd_A"], "d_seq_recovery": rc["d_seq_recovery"],
                            "test": "perf/rfd4/vae/device_parity.py"})
        clocks.append(c["aiclk"])

    cont = [m for m in modules if "rel_err" in m and "bf16_floor" in m]
    barred = [m for m in cont if "graded_on" not in m]
    worst = max(barred, key=lambda m: m["rel_err"])
    floor_ok = all(m["rel_err"] <= FLOOR_RATIO * m["bf16_floor"] for m in cont if "graded_on" in m)
    rec = {
        "branch": a.branch, "modules": modules,
        "max_rel_err": worst["rel_err"], "max_rel_err_at": worst["name"],
        "max_rel_err_over": "every real-input module against the bar; synthetic ones floor-only (graded_on)",
        "max_floor_ratio": max(m["rel_err"] / m["bf16_floor"] for m in cont if m["bf16_floor"] > 0),
        "coord_rmsd_A": max(m["rmsd_A"] for m in modules if m["name"].endswith("decoded coords")),
        "argmax_agree": min(m["argmax_agree"] for m in modules if m["name"].endswith("(147-class)")),
        "tv_mean": max(m["tv_mean"] for m in modules if m["name"].endswith("(147-class)")),
        "bar": BAR["max_rel_err"], "bars": BAR,
    }
    real = [m for m in modules if m["name"].endswith("input structure")]
    rec["d_rmsd_A"] = max(m["d_rmsd_A"] for m in real) if real else None
    rec["d_seq_recovery"] = min(m["d_seq_recovery"] for m in real) if real else None
    ok = floor_ok and (rec["max_rel_err"] <= BAR["max_rel_err"] and rec["coord_rmsd_A"] <= BAR["coord_rmsd_A"]
          and rec["argmax_agree"] >= BAR["argmax_agree"] and rec["tv_mean"] <= BAR["tv_mean"]
          and (not real or (rec["d_rmsd_A"] <= BAR["d_rmsd_A"] and rec["d_seq_recovery"] >= BAR["d_seq_recovery"])))
    rec["verdict"] = "PASS" if ok else "FAIL"
    rec["card"] = a.card
    rec["aiclk_per_case"] = clocks
    rec["precision"] = ("bf16 weights (input projection, block linears, heads), fp32 LN affines and biases, "
                        "fp32 residual stream, fp32 attention scores and softmax; HiFi4, fp32 dest acc")
    rec["timing_s"] = {case: c["timing_s"] for case, c in dev["cases"].items()}
    rec["goldens"] = ("upstream AE (sha 2db07bb4) in float64 on CPU: 3 synthetic mixed batches (n 96/250/500, B=2) "
                      "and 9 real encoder batches captured inside upstream's own partial-diffusion runs "
                      "(perf/rfd4/vae/gen_goldens.py, capture_real.py)")
    rec["floor"] = ("bf16_floor is upstream's bf16-mixed precision modelled in float64 (cpu_parity.py --mixed: bf16 "
                    "weights and matmul inputs, full-precision LayerNorm/softmax/residual), a lower bound on a GPU "
                    "bf16-mixed run's deviation; bf16_weight_floor rounds only the weights")
    rec["sha"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent).stdout.strip()
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")

    row = {"row": "rfd4-vae", "branch": a.branch, "status": "done" if ok else "open", "card": a.card,
           "ref": "float64 upstream goldens", "bar": BAR["max_rel_err"], "max_rel_err": rec["max_rel_err"],
           "verdict": rec["verdict"], "modules": [m["name"] for m in modules],
           "summary": {k: rec[k] for k in ("max_floor_ratio", "coord_rmsd_A", "argmax_agree", "tv_mean",
                                           "d_rmsd_A", "d_seq_recovery")}}
    Path(a.row).write_text(json.dumps(row, indent=2) + "\n")
    print(rec["verdict"], json.dumps(row["summary"]), "max_rel_err", rec["max_rel_err"], rec["max_rel_err_at"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
