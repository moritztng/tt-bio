"""Reduce the sweep JSONL into perf/pxdesign/gpu_reference.json — one cell per
(gpu, target size, preset, batch).

    python perf/pxdesign/make_reference.py --jsonl a.jsonl b.jsonl --out gpu_reference.json \
        --manifest perf/pxdesign/targets/manifest.json

Rep 0 of every cell is dropped as cold: PXDesign JIT-compiles two CUDA extensions on first use and
AF2-IG/ProteinMPNN pay a first-call JAX/torch compile, all of which land in rep 0 and in no later
rep. Everything reported is the median over the warm reps, and the cold rep is kept in the cell so
the discard is auditable rather than invisible.
"""

import argparse
import json
import pathlib
import statistics

LEAVES = ("prep_host", "model_init", "gen_feat", "gen_device", "gen_write", "tgt_template",
          "mpnn", "af2_complex", "af2_monomer", "ptx_mini", "ptx", "metrics_host", "rank_host")
SPLIT_KEYS = ("pxdesign_d_s", "protenix_s", "af2ig_s", "proteinmpnn_s", "host_data_s")


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="H200")
    ap.add_argument("--anchor-residues", type=int, default=116,
                    help="target residue count for the PD-L1 anchor cells")
    a = ap.parse_args()

    manifest = json.loads(pathlib.Path(a.manifest).read_text()) if a.manifest else {}
    sizes = {r["yaml"].replace(".yaml", ""): r["target_residues"]
             for r in manifest.get("rungs", {}).values()} if manifest else {}
    sizes.update({k: v["target_residues"] for k, v in manifest.get("rungs", {}).items()}
                 if manifest else {})

    recs = []
    for f in a.jsonl:
        p = pathlib.Path(f)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))

    by_cell = {}
    for r in recs:
        by_cell.setdefault(r["label"], []).append(r)

    cells = []
    for label, rs in sorted(by_cell.items()):
        rs.sort(key=lambda r: r["rep"])
        warm = [r for r in rs if not r.get("cold")]
        if not warm:
            warm = rs  # a single-rep probe: say so rather than drop the cell
        ref = warm[-1]

        target_res = a.anchor_residues
        for k, v in sizes.items():
            if k.split("_")[0] in label and str(v) in label.replace("lacz", ""):
                target_res = v
        for token in label.split("_"):
            if token.startswith("lacz") and token[4:].isdigit():
                target_res = int(token[4:])
        stages = {}
        for name in LEAVES + ("gen_total", "eval_total"):
            stages[name + "_s"] = med([(r.get("stages") or {}).get(name, {}).get("s")
                                       for r in warm])
        split = {k: med([(r.get("split") or {}).get(k) for r in warm]) for k in SPLIT_KEYS}
        total = med([r.get("total_s") for r in warm])
        pct = {k.replace("_s", "_pct"): (round(100.0 * v / total, 2) if v is not None and total
                                        else None) for k, v in split.items()}
        val = ref.get("validation") or {}
        cell = {
            "gpu": a.gpu,
            "label": label,
            "target": "PD-L1 5o45 A" if "pdl1" in label else "beta-galactosidase 1DP0 A",
            "target_residues": target_res,
            "binder_length": manifest.get("binder_length", 80),
            "preset": ref.get("preset"),
            "batch_n_sample": ref.get("n_sample"),
            "n_step": ref.get("n_step"),
            "dtype": ref.get("dtype"),
            "extra_argv": ref.get("extra") or None,
            "yaml": ref.get("yaml"),
            "yaml_sha256": ref.get("yaml_sha256"),
            "reps_total": len(rs),
            "reps_warm": len(warm),
            "cold_rep_total_s": rs[0].get("total_s") if rs[0].get("cold") else None,
            "total_s_median": total,
            "total_s_min": min([r["total_s"] for r in warm if r.get("total_s")], default=None),
            "total_s_max": max([r["total_s"] for r in warm if r.get("total_s")], default=None),
            "total_s_all_warm": [r.get("total_s") for r in warm],
            "s_per_design": (round(total / ref["n_sample"], 4)
                             if total and ref.get("n_sample") else None),
            "stages_s": stages,
            "split_s": split,
            "split_pct": pct,
            "unattributed_s": med([r.get("unattributed_s") for r in warm]),
            "gpu_util_pct_mean": med([(r.get("gpu_whole_run") or {}).get("util_pct_mean")
                                      for r in warm]),
            "gpu_power_W_median": med([(r.get("gpu_whole_run") or {}).get("power_W_median")
                                       for r in warm]),
            "gpu_power_W_max": med([(r.get("gpu_whole_run") or {}).get("power_W_max")
                                    for r in warm]),
            "gpu_per_stage_util": ref.get("gpu_per_stage"),
            "peak_vram_alloc_GiB": med([r.get("peak_vram_alloc_GiB") for r in warm]),
            "counts": ref.get("counts"),
            "subprocess_walls_s": ref.get("subprocesses"),
            "validation_ok": val.get("ok"),
            "n_designs_returned": val.get("summary_rows"),
            "filters_passed": val.get("filters"),
            "metrics": val.get("metrics"),
            "design_sequences": val.get("sequences"),
            "gpu_exclusive": all(r.get("gpu_exclusive") for r in warm),
            "sanity_ok": all(r.get("sanity_ok") for r in warm),
            "why": [w for r in warm for w in (r.get("why") or [])],
        }
        cells.append(cell)

    ref_env = next((r.get("env") for r in recs if r.get("env")), {})
    doc = {
        "model": "pxdesign",
        "what": "PXDesign GPU reference: seconds per design on one H200, split per pipeline stage",
        "measured_utc": "2026-08-19",
        "gpu": a.gpu,
        "stack": ref_env,
        "harness": {
            "runner": "perf/pxdesign/gpu_pxdesign_run.py",
            "sweep": "perf/pxdesign/gpu_pxdesign_sweep.py",
            "targets": "perf/pxdesign/make_targets.py",
            "setup": ["perf/pxdesign/gpu_pxdesign_setup.sh"]
                     + ["perf/pxdesign/gpu_pxdesign_setup_p%d.sh" % i for i in (2, 3, 4, 5)],
        },
        "protocol": {
            "reps": "rep 0 discarded as cold (two CUDA extensions JIT-compile on first use), "
                    "median over the warm reps",
            "batch": "batch is --N_sample, the number of backbones from one diffusion call; a cell "
                     "is comparable only against a cell with the same target length AND the same "
                     "N_sample",
            "presets": "extended = AF2-IG + Protenix base filter; preview = AF2-IG only. Recorded "
                       "separately, never pooled",
            "stage_split": "leaf stages partition the run; gen_total and eval_total are umbrellas "
                           "over leaves and are never summed into the split",
            "kernel_paths": "counted, not inferred: DS4Sci_EvoformerAttention call count per run",
            "exclusivity": "every compute app on the card is recorded before and after each cell",
        },
        "target_manifest": manifest,
        "cells": cells,
    }
    pathlib.Path(a.out).write_text(json.dumps(doc, indent=2, default=str))
    print("wrote %s: %d cells" % (a.out, len(cells)))
    for c in cells:
        print("  %-26s %4s aa %-8s N=%-2s total=%8s s/design=%8s util=%5s%% ok=%s"
              % (c["label"], c["target_residues"], c["preset"], c["batch_n_sample"],
                 c["total_s_median"], c["s_per_design"], c["gpu_util_pct_mean"],
                 c["validation_ok"]))


if __name__ == "__main__":
    main()
