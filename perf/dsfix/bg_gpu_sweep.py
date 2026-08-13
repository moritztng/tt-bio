"""BoltzGen H200 ladder: per-step diffusion wall + power, per rung x per batch.

Same measurement as the TT ladder: one timestamp per denoising step, median over the warm steps
of the second design. Two designs per point, the first discarded whole (compile and autotune),
and the first 3 steps of the warm design dropped.

Power is sampled at 200 ms and reduced over the warm window only -- from the first warm step to
the last -- so model load and the cold design cannot drag the median down.

Per the closed finding in `boltzgen-batch-threshold-dead-end` diffusion_batch_size is an
idleness probe here, never a production recommendation.
"""
import json, os, pathlib, statistics, subprocess, sys, threading, time

OUT = pathlib.Path("/work/results/bg_gpu.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
STEPS = 60
PY = "/work/v_bg/bin/python"
POINTS = [(r, b) for r in ["R0", "R1", "R2", "R3", "R4"] for b in (1, 2, 4)]

done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            done.add((r["rung"], r["batch"]))


def sample_power(stop, sink):
    p = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used,clocks.sm",
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


def run(rung, b):
    out_dir = "/work/out/bg_%s_b%d" % (rung, b)
    subprocess.run(["rm", "-rf", out_dir], check=False)
    cmd = [PY, "/work/bg_gpu_run.py", "run",
           "/work/perf/dsfix/fixtures/bg_%s.yaml" % rung,
           "--output", out_dir, "--protocol", "protein-anything",
           # num_designs MUST be 2*b, not 2. diffusion_batch_size batches b designs into ONE
           # denoising loop, so num_designs=2 at b=2 gives a single 60-step loop with no cold
           # block to discard -- and on the TT side it made the batch arm a no-op that silently
           # re-measured b=1. 2*b gives exactly two loops of b: discard the first, measure the
           # second.
           "--steps", "design", "--num_designs", str(2 * b),
           "--diffusion_batch_size", str(b), "--no_subprocess",
           "--config", "design", "sampling_steps=%d" % STEPS]
    samples, stop = [], threading.Event()
    th = threading.Thread(target=sample_power, args=(stop, samples), daemon=True)
    th.start()
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, cwd="/work")
    stamps, tail = [], []
    for line in p.stdout:
        if line.startswith("STEP "):
            stamps.append(float(line.split()[1]))
        else:
            tail.append(line)
            if len(tail) > 60:
                tail.pop(0)
    p.wait()
    stop.set(); th.join(timeout=5)
    return stamps, samples, time.time() - t0, p.returncode, "".join(tail), out_dir


def sanity(out_dir):
    p = pathlib.Path(out_dir)
    files = list(p.rglob("*.cif")) + list(p.rglob("*.pdb"))
    if not files:
        return False, "no structure written", None
    n, bad = 0, 0
    for line in files[0].read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            n += 1
            for tok in line.split()[10:13]:
                try:
                    v = float(tok)
                except ValueError:
                    continue
                if v != v or abs(v) == float("inf"):
                    bad += 1
    if n == 0:
        return False, "0 atoms", None
    if bad:
        return False, "%d non-finite coords" % bad, n
    return True, "", n


for rung, b in POINTS:
    if (rung, b) in done:
        print("[bgg] %s b=%d cached" % (rung, b), flush=True)
        continue
    stamps, samples, wall, rc, tail, out_dir = run(rung, b)
    # Two designs of STEPS steps each; the warm design is the second block.
    if len(stamps) < 2 * STEPS - 2:
        print("[bgg] %s b=%d FAILED rc=%d (%d stamps)\n%s"
              % (rung, b, rc, len(stamps), tail[-2500:]), flush=True)
        continue
    warm = stamps[STEPS + 3:2 * STEPS]
    per = [warm[i + 1] - warm[i] for i in range(len(warm) - 1)]
    win = [s for s in samples if warm[0] <= s[0] <= warm[-1]]
    ok, why, natoms = sanity(out_dir)
    n_written = len(list(pathlib.Path(out_dir).rglob("*.cif")))
    rec = {"rung": rung, "batch": b, "sampling_steps": STEPS,
           "step_ms_median": round(statistics.median(per) * 1000, 3),
           "step_ms_min": round(min(per) * 1000, 3), "step_ms_max": round(max(per) * 1000, 3),
           "n_steps": len(per), "n_stamps_total": len(stamps),
           "num_designs_requested": 2 * b, "n_cif_written": n_written,
           "designs_per_s": round(b / (500 * statistics.median(per)), 5),
           "proc_wall_s": round(wall, 1),
           "power_W_median": round(statistics.median([s[1] for s in win]), 1) if win else None,
           "power_W_min": round(min(s[1] for s in win), 1) if win else None,
           "power_W_max": round(max(s[1] for s in win), 1) if win else None,
           "util_pct_median": round(statistics.median([s[2] for s in win]), 1) if win else None,
           "mem_MiB_max": round(max(s[3] for s in win), 0) if win else None,
           "n_power_samples": len(win),
           "sanity_ok": ok, "sanity_why": why, "atoms_written": natoms,
           "gpu": "H200", "power_limit_W": 700.0, "idle_W": 80.3}
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("[bgg] %s b=%d  %.2f ms/step (n=%d)  %sW  util %s%%  sanity=%s"
          % (rung, b, rec["step_ms_median"], len(per), rec["power_W_median"],
             rec["util_pct_median"], ok), flush=True)
print("[bgg] ALL DONE", flush=True)
