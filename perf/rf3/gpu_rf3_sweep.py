"""RF3 H200 size ladder: one subprocess per rung, power sampled, results appended as JSONL.

    python perf/rf3/gpu_rf3_sweep.py --repo /work --results /work/results/rf3_gpu.jsonl

Forked from perf/dsfix/gpu_rfd3_sweep.py. One subprocess per rung on purpose: a rung is a fixed
shape, so its reps recompile nothing, but two rungs in one process would reorder and recompile
(measured on RFD3: a 5-spec probe read 3.9-8.5 s for work that took 2.7 s in a dedicated process).
The checkpoint load is re-paid per rung and sits outside every reported number.

Power is sampled at 200 ms for the whole subprocess and reduced over the WARM WINDOW only: from
the end of rep 0 to the end of the last rep, located from the runner's own `[rf3] ... repN` lines.
Model load and the cold rep sit near idle and would drag the median down.

Resumable: a rung already in the JSONL is skipped, so a preempted instance costs one rung.
"""

import argparse
import json
import pathlib
import re
import statistics
import subprocess
import sys
import threading
import time

REP_LINE = re.compile(r"\[rf3\] (\S+) rep(\d+) ([\d.]+)s")


def sample_power(stop: threading.Event, sink: list) -> None:
    p = subprocess.Popen(["nvidia-smi",
                          "--query-gpu=power.draw,utilization.gpu,memory.used,clocks.sm",
                          "--format=csv,noheader,nounits", "-lms", "200"],
                         stdout=subprocess.PIPE, text=True)
    try:
        for line in p.stdout:                                 # type: ignore[union-attr]
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


def run_rung(a, n: int, batch: int) -> dict | None:
    label = "cdk2_%d_b%d" % (n, batch)
    repo = pathlib.Path(a.repo)
    report = pathlib.Path(a.results).parent / ("run_%s.json" % label)
    out_dir = "/work/out/%s" % label
    subprocess.run(["rm", "-rf", out_dir], check=False)
    cmd = [a.python, str(repo / "perf/rf3/gpu_rf3_run.py"),
           "--inputs", str(repo / ("perf/rf3/inputs/rf3_%d.json" % n)),
           "--out-dir", out_dir, "--report", str(report),
           "--reps", str(a.reps), "--n-recycles", str(a.n_recycles),
           "--num-steps", str(a.num_steps), "--diffusion-batch-size", str(batch),
           "--early-stop-plddt", "0", "--seed", str(a.seed), "--label", label]

    samples: list = []
    stop = threading.Event()
    th = threading.Thread(target=sample_power, args=(stop, samples), daemon=True)
    th.start()
    t0 = time.time()
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          cwd=str(repo))
    lines, rep_end = [], []
    for line in pr.stdout:                                    # type: ignore[union-attr]
        lines.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()
        if REP_LINE.search(line):
            rep_end.append(time.time())
    rc = pr.wait()
    wall = time.time() - t0
    stop.set()
    th.join(timeout=5)
    log = "".join(lines)

    if not report.exists():
        print("[sweep] %s NO REPORT rc=%d\n%s" % (label, rc, log[-3000:]), flush=True)
        return None
    rep = json.loads(report.read_text())
    reps = rep.get("rep_s") or []
    if len(reps) < 2:
        print("[sweep] %s only %d reps rc=%d\n%s" % (label, len(reps), rc, log[-3000:]),
              flush=True)
        return None
    warm = reps[1:]

    win = [s for s in samples if len(rep_end) >= 2 and rep_end[0] <= s[0] <= rep_end[-1]]
    phases = rep.get("phases", {})
    warm_phase_keys = [str(i) for i in range(1, len(reps))]
    phase_names = ("featinit", "trunk", "distogram", "diffusion", "confidence")
    phase_med = {}
    for name in phase_names:
        xs = [phases[k][name] for k in warm_phase_keys if k in phases and name in phases[k]]
        phase_med[name + "_s"] = round(statistics.median(xs), 4) if xs else None
    accounted = sum(v for v in phase_med.values() if v)
    med = statistics.median(warm)

    rec = {"model": "rf3", "rung_aa": n, "batch": batch, "label": label,
           "n_recycles": a.n_recycles, "num_steps": a.num_steps, "seed": a.seed,
           "reps_total": len(reps), "reps_warm": len(warm),
           "cold_rep_s": reps[0],
           "fold_s_median": round(med, 4), "fold_s_min": round(min(warm), 4),
           "fold_s_max": round(max(warm), 4), "fold_s_all_warm": warm,
           "load_s": rep.get("load_s"),
           "peak_vram_alloc_GiB": round((rep.get("peak_vram_alloc_B") or 0) / 2**30, 3),
           "peak_vram_reserved_GiB": round((rep.get("peak_vram_reserved_B") or 0) / 2**30, 3),
           "other_s": round(med - accounted, 4),
           "power_W_median": round(statistics.median([s[1] for s in win]), 1) if win else None,
           "power_W_max": round(max(s[1] for s in win), 1) if win else None,
           "util_pct_median": round(statistics.median([s[2] for s in win]), 1) if win else None,
           "mem_MiB_max": round(max(s[3] for s in win), 0) if win else None,
           "clock_sm_median": round(statistics.median([s[4] for s in win]), 0) if win else None,
           "n_power_samples": len(win),
           "counts": rep.get("counts"), "confidence": rep.get("confidence"),
           "sanity_ok": rep.get("ok"), "sanity_why": rep.get("why"),
           "proc_wall_s": round(wall, 1), "rc": rc, "env": rep.get("env")}
    rec.update(phase_med)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/work")
    ap.add_argument("--python", default="/work/v_rf3/bin/python")
    ap.add_argument("--results", default="/work/results/rf3_gpu.jsonl")
    ap.add_argument("--sizes", default="128,256,512,768,1024")
    ap.add_argument("--batches", default="1")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--n-recycles", type=int, default=10)
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = pathlib.Path(a.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["rung_aa"], r["batch"]))

    for batch in [int(x) for x in a.batches.split(",")]:
        for n in [int(x) for x in a.sizes.split(",")]:
            if (n, batch) in done:
                print("[sweep] %d aa b=%d cached" % (n, batch), flush=True)
                continue
            rec = run_rung(a, n, batch)
            if rec is None:
                continue
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print("[sweep] == %d aa b=%d  fold=%.3fs  trunk=%s diff=%s conf=%s  %sW  "
                  "vram=%.1fGiB  cueq_tri_att=%s cueq_tri_mul=%s  sanity=%s"
                  % (n, batch, rec["fold_s_median"], rec["trunk_s"], rec["diffusion_s"],
                     rec["confidence_s"], rec["power_W_median"],
                     rec["peak_vram_alloc_GiB"],
                     (rec["counts"] or {}).get("triangle_attention_cueq"),
                     (rec["counts"] or {}).get("triangle_multiply_cueq"),
                     rec["sanity_ok"]), flush=True)
    print("[sweep] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
