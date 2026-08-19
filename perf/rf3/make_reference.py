"""Turn the H200 sweep JSONL into perf/rf3/gpu_reference.json, the target the TT port is scored on.

    python perf/rf3/make_reference.py \
        --jsonl perf/rf3/results/m132116_full.jsonl \
        --control perf/rf3/results/m146533_devicephases.jsonl \
        --out perf/rf3/gpu_reference.json --measured-utc ...

THE PRIMARY REFERENCE IS `h200_device_s`, NOT end-to-end fold time.

That is a measured decision, not a preference. Two independently rented H200 boxes running the
identical stack agree on the GPU phases to within 2% at 512/768/1024 aa (512 aa trunk: 4.4845 s vs
4.4944 s) but disagree by 2.3x on the AtomWorks host featurisation (128 aa prep: 6.61 s vs ~2.9 s
implied). End-to-end fold time on a rented box is therefore a property of whichever CPU came with
the GPU, and publishing it as "the H200 number" would bake one landlord's host into the port's
target for the next six months. `h200_fold_s` is still published, with that caveat attached.

The TT bar: within 4x of one H200 per chip at MATCHED batch, the same bar as one 32-chip Galaxy
beating one 8-GPU DGX H200 (`perf-page-server-throughput-bar-equals-4x-gap`). So
`tt_target_device_s = 4 * h200_device_s`, per rung, per batch. A TT number is comparable only
against the row with the same `rung_aa` AND the same `batch`
(`perf-page-matched-batch-protocol-recurrence` has shipped three times).

Each input JSON is hashed in. If the TT side folds a file whose sha256 is not here, the comparison
is void and should say so rather than quietly print a ratio.
"""

import argparse
import hashlib
import json
import pathlib

# Leaf phases partition the fold. `network` (the trainer step) contains featinit..confidence, so
# it is reported as a cross-check and never summed.
LEAF = ("prep", "featinit", "trunk", "distogram", "diffusion", "confidence",
        "assemble", "confcompile", "write")
DEVICE = ("featinit", "trunk", "distogram", "diffusion", "confidence")
HOST = ("prep", "assemble", "confcompile", "write")
PHASES = LEAF + ("network",)


def load(path: str) -> list[dict]:
    recs = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip()]
    recs.sort(key=lambda r: (r["batch"], r["rung_aa"]))
    return recs


def dev_sum(r: dict) -> float:
    return round(sum(r.get(k + "_s") or 0 for k in DEVICE), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="primary sweep: full phase breakdown")
    ap.add_argument("--control", help="second box, same stack: cross-box reproducibility check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="H200")
    ap.add_argument("--measured-utc", required=True)
    ap.add_argument("--primary-box", default="")
    ap.add_argument("--control-box", default="")
    a = ap.parse_args()

    recs = load(a.jsonl)
    ctrl = {(r["rung_aa"], r["batch"]): r for r in load(a.control)} if a.control else {}
    repo = pathlib.Path(__file__).resolve().parents[2]

    inputs = {}
    for p in sorted((repo / "perf/rf3/inputs").glob("rf3_*.json")):
        spec = json.loads(p.read_text())
        inputs[p.name] = {
            "path": "perf/rf3/inputs/" + p.name,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "name": spec["name"],
            "n_chains": len(spec["components"]),
            "length_aa": len(spec["components"][0]["seq"]),
        }

    env = recs[0].get("env") or {}
    rows, xbox = [], []
    for r in recs:
        fold, dev = r["fold_s_median"], dev_sum(r)
        host = round(sum(r.get(k + "_s") or 0 for k in HOST), 4)
        row = {"rung_aa": r["rung_aa"], "batch": r["batch"],
               "input": "perf/rf3/inputs/rf3_%d.json" % r["rung_aa"],
               # --- the target -----------------------------------------------------------
               "h200_device_s": dev,
               "tt_target_device_s": round(4.0 * dev, 4),
               "h200_structures_per_s_device": round(r["batch"] / dev, 5),
               # --- e2e, host-dependent --------------------------------------------------
               "h200_fold_s": fold,
               "tt_target_fold_s": round(4.0 * fold, 4),
               "h200_host_s": host,
               "host_frac_of_fold": round(host / fold, 4),
               "h200_fold_s_min": r["fold_s_min"], "h200_fold_s_max": r["fold_s_max"],
               "h200_fold_s_spread_pct": round(100 * (r["fold_s_max"] - r["fold_s_min"]) / fold, 2),
               # --- provenance -----------------------------------------------------------
               "reps_warm": r["reps_warm"], "cold_rep_s": r["cold_rep_s"],
               "ckpt_load_s": r["load_s"],
               "peak_vram_alloc_GiB": r["peak_vram_alloc_GiB"],
               "peak_vram_reserved_GiB": r["peak_vram_reserved_GiB"],
               "power_W_median": r["power_W_median"], "power_W_max": r["power_W_max"],
               "util_pct_median": r["util_pct_median"],
               "clock_sm_median": r["clock_sm_median"],
               "phase_s": {k: r.get(k + "_s") for k in PHASES},
               "unattributed_s": r["other_s"],
               "counts": r["counts"], "confidence": r.get("confidence"),
               "sanity_ok": r["sanity_ok"], "sanity_why": r["sanity_why"],
               "gpu_exclusive": r.get("gpu_exclusive")}
        rows.append(row)

        c = ctrl.get((r["rung_aa"], r["batch"]))
        if c:
            cd = dev_sum(c)
            xbox.append({"rung_aa": r["rung_aa"], "batch": r["batch"],
                         "primary_device_s": dev, "control_device_s": cd,
                         "device_delta_pct": round(100 * (cd - dev) / dev, 2),
                         "primary_trunk_s": r.get("trunk_s"), "control_trunk_s": c.get("trunk_s"),
                         "trunk_delta_pct": round(100 * ((c.get("trunk_s") or 0)
                                                         - (r.get("trunk_s") or 0))
                                                  / (r.get("trunk_s") or 1), 2),
                         "primary_fold_s": fold, "control_fold_s": c["fold_s_median"],
                         "fold_delta_pct": round(100 * (c["fold_s_median"] - fold) / fold, 2)})

    doc = {
        "model": "rf3",
        "what": "RoseTTAFold3 single-GPU H200 reference, the denominator the tt-bio RF3 port is "
                "scored against.",
        "measured_utc": a.measured_utc,
        "gpu": a.gpu,
        "primary_box": a.primary_box,
        "control_box": a.control_box,
        "headline": {
            "primary_metric": "h200_device_s",
            "why": "Two independently rented H200 boxes on the identical stack agree on the GPU "
                   "phases to within 2% at 512/768/1024 aa but differ by up to 2.3x on the "
                   "AtomWorks host featurisation. End-to-end fold time on a rented box is a "
                   "property of the CPU that came with the GPU, so it is reported but is not the "
                   "target.",
            "secondary_metric": "h200_fold_s (end-to-end, host-dependent -- see host_frac_of_fold)",
        },
        "target_definition": {
            "rule": "tt_target_device_s = 4 * h200_device_s, per rung, at matched batch",
            "why": "The org bar is within 4x of one H200 per chip at matched batch, the same bar "
                   "as one 32-chip Galaxy beating one 8-GPU DGX H200.",
            "matched_batch": "A TT number is comparable only against the row with the same "
                             "rung_aa AND the same batch. Different batch, no comparison.",
            "e2e_caveat": "tt_target_fold_s is also published, but an e2e TT-vs-GPU comparison is "
                          "only meaningful if both arms' host prep ran on comparable CPUs, or if "
                          "host time is subtracted from both. The TT port inherits the same "
                          "AtomWorks pipeline, so host_s is common cost, not a TT deficit.",
        },
        "protocol": {
            "reps": "4 per rung in one process; rep 0 discarded as cold; median of the warm 3",
            "timing": "wall-clock around cuda-synchronised phase boundaries",
            "excluded": "checkpoint load (reported separately as ckpt_load_s)",
            "device_phases": list(DEVICE),
            "host_phases": list(HOST),
            "batch_meaning": "batch = diffusion_batch_size = structures in the ensemble from one "
                             "trunk forward pass. RF3 ships 5; 1 is the single-structure point.",
            "n_recycles": recs[0]["n_recycles"],
            "num_steps": recs[0]["num_steps"],
            "seed": recs[0]["seed"],
            "early_stopping_plddt_threshold": 0,
            "early_stopping_note": "RF3 ships 0.5, which truncates a no-MSA fold after one "
                                   "recycle. Forced to 0 so every rung runs the full network.",
            "msa": "none (single sequence), no templates",
            "gpu_exclusive": "every rung asserts it was the only compute app on the card. A "
                             "co-tenant on the same physical GPU voids absolute timings; one was "
                             "caught doing exactly that mid-campaign.",
        },
        "stack": {
            "rc_foundry": env.get("rc-foundry"), "atomworks": env.get("atomworks"),
            "torch": env.get("torch"), "cuda": env.get("torch_cuda"),
            "cudnn": env.get("cudnn"), "python": env.get("python"),
            "cuequivariance_torch": env.get("cuequivariance-torch"),
            "cuequivariance_ops_torch_cu12": env.get("cuequivariance-ops-torch-cu12"),
            "lightning": env.get("lightning"),
            "driver": (env.get("gpu_static") or {}).get("driver_version"),
            "power_limit_W": (env.get("gpu_static") or {}).get("power.limit"),
            "ckpt": "rf3_foundry_01_24_latest_remapped.ckpt (registry name `rf3`)",
            "should_use_cuequivariance": env.get("should_use_cuequivariance"),
            "float32_matmul_precision": env.get("float32_matmul_precision"),
        },
        "kernel_paths": {
            "note": "Counted, not assumed. RF3 picks per call on "
                    "`self.use_cuequivariance and SHOULD_USE_CUEQUIVARIANCE`.",
            "result": "RF3 reaches cuEquivariance on every triangle op: triangle_attention_cueq "
                      "and triangle_multiply_cueq carry 100% of calls, both vanilla counters are "
                      "0, and F.scaled_dot_product_attention is never called. This is the "
                      "opposite of RFD3, where the cueq counter read 0 with the same wheel "
                      "installed -- so the RF3 GPU baseline is a fused-kernel baseline and the "
                      "TT port is being compared against NVIDIA's tuned triangle kernels, not "
                      "against vanilla PyTorch einsums.",
            "counts_are_per_process": "each rung ran 4 reps, so a count is 4x the per-fold count",
        },
        "cross_box_reproducibility": {
            "what": "same stack, same inputs, two independently rented H200s on different "
                    "machines. `control` is the second box.",
            "conclusion": "device time reproduces (<=2% at >=512 aa); fold time does not, because "
                          "host prep does not.",
            "rungs": xbox,
        },
        "inputs": inputs,
        "rungs": rows,
    }
    pathlib.Path(a.out).write_text(json.dumps(doc, indent=2) + "\n")

    print("wrote %s: %d rungs" % (a.out, len(rows)))
    print("%5s %3s | %9s %11s | %9s %8s %6s | %5s" %
          ("aa", "b", "device_s", "TT target", "fold_s", "host_s", "host%", "GiB"))
    for r in rows:
        print("%5d %3d | %9.3f %11.3f | %9.3f %8.3f %5.0f%% | %5.1f" %
              (r["rung_aa"], r["batch"], r["h200_device_s"], r["tt_target_device_s"],
               r["h200_fold_s"], r["h200_host_s"], 100 * r["host_frac_of_fold"],
               r["peak_vram_alloc_GiB"]))
    if xbox:
        print("\ncross-box device agreement (control vs primary):")
        for x in xbox:
            print("  %4d aa b=%d  device %+6.2f%%   trunk %+6.2f%%   fold %+7.2f%%" %
                  (x["rung_aa"], x["batch"], x["device_delta_pct"], x["trunk_delta_pct"],
                   x["fold_delta_pct"]))


if __name__ == "__main__":
    main()
