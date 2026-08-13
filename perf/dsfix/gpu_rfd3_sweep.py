"""RFD3 H200 ladder: per-step wall + power, per rung x per batch.

One subprocess per (rung, batch, N) on purpose. A single process fed several specs reorders them
and recompiles on every shape change -- a 5-spec N=2 probe read 3.9-8.5 s per batch against 2.7 s
for the same work in a dedicated process -- so batch times could not be attributed to a rung
without guessing. Model load is re-paid per point; the step-count differential cancels it exactly.

Per point: n_batches=4, the first discarded as cold, median of the warm 3. Then

    per_step = (median t(N2) - median t(N1)) / (N2 - N1)

"Finished inference batch in X seconds" (rfd3/engine.py:334) wraps the sampler loop only, so
featurisation and file IO are already outside the number.

Power is sampled at 200 ms for the whole subprocess and reduced over the WARM WINDOW only --
from the end of rep 0 to the end of the last rep, located by the engine's own HH:MM:SS log
stamps. Model load and the cold rep sit at near-idle draw and would drag the median down.
"""
import json, os, pathlib, re, statistics, subprocess, sys, threading, time

N1, N2, NREP = 8, 40, 4          # NREP includes the discarded cold rep
RFD3 = "/work/v_rfd3/bin/rfd3"
FIX = "/work/perf/dsfix/fixtures"
OUT = pathlib.Path("/work/results/rfd3_gpu.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
# The GPU has its own b_max. TT's per-rung clamp comes from `8*3359**2 // L**2` in
# tt_bio/rfd3/design.py; the H200 has 143 GB and used 9.1 GB at R1 b=8, so that clamp is
# meaningless here. Plan sec.4.5 pre-commits the GPU to b in {1,2,4,8} at every rung, with
# b_max = 8 because 8 is the batch RFD3 actually ships, so M = T(8)/T(4) throughout.
LADDER = [("R0", [1, 2, 4, 8]), ("R1", [1, 2, 4, 8]), ("R2", [1, 2, 4, 8]),
          ("R3", [1, 2, 4, 8]), ("R4", [1, 2, 4, 8])]
FIN = re.compile(r"(\d\d):(\d\d):(\d\d).*Finished inference batch in ([\d.]+) seconds")

done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            done.add((r["rung"], r["batch"], r["N"]))


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


def run_point(rung, b, N):
    out_dir = "/work/out/g_%s_b%d_n%d" % (rung, b, N)
    subprocess.run(["rm", "-rf", out_dir], check=False)
    cmd = [RFD3, "design", "out_dir=" + out_dir,
           "inputs=%s/rfd3_%s.json" % (FIX, rung),
           "inference_sampler.num_timesteps=%d" % N,
           "diffusion_batch_size=%d" % b, "n_batches=%d" % NREP,
           "skip_existing=False"]
    samples, stop = [], threading.Event()
    th = threading.Thread(target=sample_power, args=(stop, samples), daemon=True)
    th.start()
    t0 = time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    stop.set(); th.join(timeout=5)
    log = pr.stdout + pr.stderr

    reps, stamps = [], []
    day0 = time.localtime(t0)
    for m in FIN.finditer(log):
        hh, mm, ss, secs = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
        st = time.mktime((day0.tm_year, day0.tm_mon, day0.tm_mday, hh, mm, ss, 0, 0, -1))
        reps.append(secs); stamps.append(st)
    return reps, stamps, samples, wall, log, pr.returncode


def sanity(rung, b, N):
    """Every written design must exist, be non-empty, and carry finite coordinates."""
    import gzip
    out_dir = pathlib.Path("/work/out/g_%s_b%d_n%d" % (rung, b, N))
    cifs = sorted(out_dir.glob("*.cif.gz"))
    if not cifs:
        return False, "no cif written", None
    natoms = None
    for c in cifs[:2]:
        n, bad = 0, []
        with gzip.open(c, "rt") as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")):
                    n += 1
                    for tok in line.split()[10:13]:
                        try:
                            v = float(tok)
                        except ValueError:
                            continue
                        if v != v or abs(v) == float("inf"):
                            bad.append(tok)
        if n == 0:
            return False, "%s has 0 atoms" % c.name, None
        if bad:
            return False, "%s has non-finite coords" % c.name, n
        natoms = n
    return True, "", natoms


results = {}
for rung, batches in LADDER:
    for b in batches:
        legs = {}
        for N in (N1, N2):
            key = (rung, b, N)
            if key in done:
                for line in OUT.read_text().splitlines():
                    r = json.loads(line)
                    if (r["rung"], r["batch"], r["N"]) == key:
                        legs[N] = r
                print("[gpu] %s b=%d N=%d cached" % key, flush=True)
                continue
            reps, stamps, samples, wall, log, rc = run_point(rung, b, N)
            if len(reps) < 2:
                print("[gpu] %s b=%d N=%d FAILED rc=%d\n%s" % (rung, b, N, rc, log[-3000:]),
                      flush=True)
                continue
            warm = reps[1:]
            t_lo, t_hi = stamps[0], stamps[-1]
            win = [s for s in samples if t_lo <= s[0] <= t_hi]
            ok, why, natoms = sanity(rung, b, N)
            rec = {"rung": rung, "batch": b, "N": N, "n_reps_warm": len(warm),
                   "batch_s_median": round(statistics.median(warm), 4),
                   "batch_s_min": round(min(warm), 4), "batch_s_max": round(max(warm), 4),
                   "cold_rep_s": round(reps[0], 4), "reps": [round(x, 4) for x in reps],
                   "proc_wall_s": round(wall, 1),
                   "power_W_median": round(statistics.median([s[1] for s in win]), 1) if win else None,
                   "power_W_min": round(min(s[1] for s in win), 1) if win else None,
                   "power_W_max": round(max(s[1] for s in win), 1) if win else None,
                   "util_pct_median": round(statistics.median([s[2] for s in win]), 1) if win else None,
                   "mem_MiB_max": round(max(s[3] for s in win), 0) if win else None,
                   "clock_sm_median": round(statistics.median([s[4] for s in win]), 0) if win else None,
                   "n_power_samples": len(win),
                   "sanity_ok": ok, "sanity_why": why, "atoms_written": natoms,
                   "gpu": "H200", "power_limit_W": 700.0, "idle_W": 80.3,
                   "stack": "rc-foundry 0.2.0, torch 2.13.0+cu130"}
            with OUT.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            legs[N] = rec
            print("[gpu] %s b=%d N=%d  batch=%.3fs  %sW  util %s%%  sanity=%s"
                  % (rung, b, N, rec["batch_s_median"], rec["power_W_median"],
                     rec["util_pct_median"], ok), flush=True)
        if N1 in legs and N2 in legs:
            ps = (legs[N2]["batch_s_median"] - legs[N1]["batch_s_median"]) / (N2 - N1)
            print("[gpu] == %s b=%d per_step=%.2f ms  T=%.5f designs/s @200"
                  % (rung, b, ps * 1000, b / (200 * ps)), flush=True)
print("[gpu] ALL DONE", flush=True)
