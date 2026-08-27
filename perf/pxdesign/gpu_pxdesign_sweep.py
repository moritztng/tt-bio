"""PXDesign GPU reference sweep: one subprocess per (target, preset, N_sample) cell, power sampled,
results appended as JSONL.

    python perf/pxdesign/gpu_pxdesign_sweep.py --cells anchor --reps 3 \
        --results /work/results/pxd_gpu.jsonl

Forked from perf/rf3/gpu_rf3_sweep.py. One subprocess per rep on purpose: PXDesign JIT-compiles two
CUDA extensions on first use (DeepSpeed's evoformer_attn and protenix's fast LayerNorm) and caches
them under ~/.cache/torch_extensions, so rep 0 of the whole sweep pays several minutes that no
later rep pays. Rep 0 of every cell is discarded as cold and reported separately.

nvidia-smi is sampled at 200 ms for the whole subprocess, then reduced TWICE:
  - over the warm window, for the cell's headline power/utilisation,
  - over each stage's own [t0, t1] window from the runner's report, which is the only way to see
    device vs host inside the AF2-IG and ProteinMPNN subprocesses, since pxdbench spawns those and
    we do not instrument other people's processes.

Resumable: a (cell, rep) already in the JSONL is skipped, so a preempted instance costs one rep.
"""

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import threading
import time

ANCHOR = [
    # label, yaml, preset, n_sample
    ("pdl1_ext_n1", "examples/PDL1_quick_start.yaml", "extended", 1),
    ("pdl1_ext_n8", "examples/PDL1_quick_start.yaml", "extended", 8),
    ("pdl1_prev_n1", "examples/PDL1_quick_start.yaml", "preview", 1),
    ("pdl1_prev_n8", "examples/PDL1_quick_start.yaml", "preview", 8),
]


def sample_power(stop, sink):
    p = subprocess.Popen(["nvidia-smi",
                          "--query-gpu=power.draw,utilization.gpu,memory.used,clocks.sm",
                          "--format=csv,noheader,nounits", "-lms", "200"],
                         stdout=subprocess.PIPE, text=True)
    try:
        for line in p.stdout:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 4:
                try:
                    sink.append((time.time(), float(parts[0]), float(parts[1]),
                                 float(parts[2]), float(parts[3])))
                except ValueError:
                    pass
            if stop.is_set():
                break
    finally:
        p.kill()


def reduce_window(samples, t0, t1):
    win = [s for s in samples if t0 <= s[0] <= t1]
    if not win:
        return None
    return {"power_W_median": round(statistics.median(s[1] for s in win), 1),
            "power_W_max": round(max(s[1] for s in win), 1),
            "util_pct_median": round(statistics.median(s[2] for s in win), 1),
            "util_pct_mean": round(sum(s[2] for s in win) / len(win), 1),
            "mem_MiB_max": round(max(s[3] for s in win)),
            "clock_sm_median": round(statistics.median(s[4] for s in win)),
            "n_samples": len(win)}


def run_rep(a, cell, rep):
    label, yaml_rel, preset, n_sample = cell
    repo = pathlib.Path(a.repo)
    yaml_path = yaml_rel if yaml_rel.startswith("/") else str(repo / yaml_rel)
    tag = "%s_rep%d" % (label, rep)
    report = pathlib.Path(a.results).parent / ("run_%s.json" % tag)
    out_dir = "%s/%s" % (a.out_root, tag)
    subprocess.run(["rm", "-rf", out_dir], check=False)

    cmd = [a.python, "-u", a.runner, "--yaml", yaml_path, "--out-dir", out_dir,
           "--report", str(report), "--preset", preset, "--n-sample", str(n_sample),
           "--n-step", str(a.n_step), "--dtype", a.dtype, "--seed", str(a.seed),
           "--rounds", str(a.rounds), "--label", tag]
    if a.extra:
        cmd += ["--extra", a.extra]

    samples, stop = [], threading.Event()
    th = threading.Thread(target=sample_power, args=(stop, samples), daemon=True)
    th.start()
    t0 = time.time()
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          cwd=str(repo))
    tail = []
    for line in pr.stdout:
        tail.append(line)
        if len(tail) > 400:
            tail.pop(0)
        if line.startswith("[pxd]") or "Traceback" in line or "Error" in line:
            sys.stdout.write(line)
            sys.stdout.flush()
    rc = pr.wait()
    wall = time.time() - t0
    stop.set()
    th.join(timeout=5)

    if not report.exists():
        print("[sweep] %s NO REPORT rc=%d\n%s" % (tag, rc, "".join(tail[-40:])), flush=True)
        return None
    r = json.loads(report.read_text())

    rec = {"model": "pxdesign", "label": label, "rep": rep, "cold": rep == 0,
           "preset": preset, "n_sample": n_sample, "yaml": yaml_path,
           "yaml_sha256": r.get("yaml_sha256"), "n_step": a.n_step, "dtype": a.dtype,
           "seed": a.seed, "extra": a.extra, "total_s": r.get("total_s"), "s_per_design": r.get("s_per_design"),
           "stages": r.get("stages"), "split": r.get("split"), "split_pct": r.get("split_pct"),
           "unattributed_s": r.get("unattributed_s"), "counts": r.get("counts"),
           "module_census": r.get("module_census"), "counter_info": r.get("counter_info"),
           "kernel_env_at_end": r.get("kernel_env_at_end"),
           "subprocesses": r.get("subprocesses"),
           "peak_vram_alloc_GiB": round((r.get("peak_vram_alloc_B") or 0) / 2 ** 30, 3),
           "peak_vram_reserved_GiB": round((r.get("peak_vram_reserved_B") or 0) / 2 ** 30, 3),
           "validation": r.get("validation"), "sanity_ok": r.get("ok"), "why": r.get("why"),
           "rounds": r.get("rounds"), "warm_n": r.get("warm_n"),
           "warm_median_cell_s": r.get("warm_median_cell_s"),
           "warm_spread_pct": r.get("warm_spread_pct"),
           "warm_median_gen_device_s": r.get("warm_median_gen_device_s"),
           "warm_median_gen_feat_s": r.get("warm_median_gen_feat_s"),
           "warm_median_gen_write_s": r.get("warm_median_gen_write_s"),
           "digests": r.get("digests"), "digest_repeat_ok": r.get("digest_repeat_ok"),
           "jax_counter_selftest": r.get("jax_counter_selftest"),
           "subprocess_overlaps_gen": r.get("subprocess_overlaps_gen"),
           "gpu_exclusive": r.get("gpu_exclusive"),
           "compute_apps_before": r.get("compute_apps_before"),
           "compute_apps_after": r.get("compute_apps_after"),
           "proc_wall_s": round(wall, 1), "rc": rc, "env": r.get("env")}

    rec["gpu_whole_run"] = reduce_window(samples, r.get("t_start", t0), r.get("t_end", t0 + wall))
    per_stage = {}
    for w in r.get("windows", []):
        red = reduce_window(samples, w["t0"], w["t1"])
        if red is None:
            continue
        prev = per_stage.get(w["stage"])
        # a stage can fire more than once; keep the busiest window and the total sample count
        if prev is None or red["util_pct_mean"] > prev["util_pct_mean"]:
            per_stage[w["stage"]] = red
    rec["gpu_per_stage"] = per_stage
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/work/PXDesign")
    ap.add_argument("--runner", default="/work/gpu_pxdesign_run.py")
    ap.add_argument("--python", default="python3")
    ap.add_argument("--results", default="/work/results/pxd_gpu.jsonl")
    ap.add_argument("--out-root", default="/work/out")
    ap.add_argument("--cells", default="anchor",
                    help="'anchor', or label:yaml:preset:n_sample entries separated by commas")
    ap.add_argument("--reps", type=int, default=3,
                    help="subprocesses per cell, rep 0 discarded as cold")
    ap.add_argument("--rounds", type=int, default=1,
                    help="pipeline invocations INSIDE each subprocess, round 0 discarded as cold. "
                         "--reps 2 --rounds 5 is the perf-page protocol: two independent processes, "
                         "four warm rounds each, eight warm samples")
    ap.add_argument("--n-step", type=int, default=400)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra", default="",
                    help="extra argv forwarded to pxdesign pipeline, e.g. a hydra-style override")
    ap.add_argument("--label-suffix", default="",
                    help="appended to every cell label, so an --extra arm cannot collide with the "
                         "same cell measured without it")
    a = ap.parse_args()

    if a.cells == "anchor":
        cells = [(lb + a.label_suffix, y, p_, n) for lb, y, p_, n in ANCHOR]
    else:
        cells = []
        for spec in a.cells.split(","):
            label, yaml_rel, preset, n = spec.split(":")
            cells.append((label + a.label_suffix, yaml_rel, preset, int(n)))

    out = pathlib.Path(a.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["label"], r["rep"]))

    for cell in cells:
        for rep in range(a.reps):
            if (cell[0], rep) in done:
                print("[sweep] %s rep%d cached" % (cell[0], rep), flush=True)
                continue
            rec = run_rep(a, cell, rep)
            if rec is None:
                continue
            with out.open("a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            sp = rec["split"] or {}
            print("[sweep] == %s rep%d %s N=%d total=%.1fs s/design=%.1f | pxd-d=%.1f ptx=%.1f "
                  "af2=%.1f mpnn=%.1f host=%.1f | util=%s%% %sW vram=%.1fGiB ds4sci=%s ok=%s %s"
                  % (rec["label"], rep, rec["preset"], rec["n_sample"], rec["total_s"] or -1,
                     rec["s_per_design"] or -1, sp.get("pxdesign_d_s", -1),
                     sp.get("protenix_s", -1), sp.get("af2ig_s", -1),
                     sp.get("proteinmpnn_s", -1), sp.get("host_data_s", -1),
                     (rec["gpu_whole_run"] or {}).get("util_pct_mean"),
                     (rec["gpu_whole_run"] or {}).get("power_W_median"),
                     rec["peak_vram_alloc_GiB"],
                     (rec["counts"] or {}).get("ds4sci_evo_attention"),
                     rec["sanity_ok"], rec["why"]), flush=True)
    print("[sweep] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
