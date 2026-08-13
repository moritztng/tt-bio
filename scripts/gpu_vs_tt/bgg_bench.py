"""Driver for the BoltzGen H200/B200 benchmark on the pinned bg_R3 fixture.

One arm = one process = one `boltzgen run --steps design` at the shipped settings, driven through
`bgg_run.py` so every design is timed inside a synchronised region and every cuEquivariance entry is
counted. An arm's timing is written only after its own output passes a guard: a plausible number
from a run that wrote nothing, or that wrote a target-only file with no binder in it, has happened
on this benchmark before and must not be able to count.

Usage, from /work on the rented box:
    /work/venv-bgg/bin/python scripts/gpu_vs_tt/bgg_bench.py H200 smoke
    /work/venv-bgg/bin/python scripts/gpu_vs_tt/bgg_bench.py H200 headline replicate sat2 sat4
    /work/venv-bgg/bin/python scripts/gpu_vs_tt/bgg_bench.py H200 nokern r4

Results append to /work/results/bgg_<gpu>.jsonl; one design per arm is kept under
/work/results/structures/.
"""

import json
import math
import pathlib
import shutil
import statistics
import subprocess
import sys
import threading
import time

WORK = pathlib.Path("/work")
PY = str(WORK / "venv-bgg" / "bin" / "python")
RUNNER = str(WORK / "scripts" / "gpu_vs_tt" / "bgg_run.py")
RESULTS = WORK / "results"
BINDER_RES = 100
TARGET_RES = {"R3": 414, "R4": 585}
TIMEOUT_S = 1200          # a clean R3 arm is minutes; anything at 20 min is the Blackwell hang
HANG_WINDOW_S = 90        # power/CPU are reduced over this tail when an arm times out

# Every arm runs the shipped design defaults: sampling_steps 500 (design.yaml), recycling_steps 3,
# precision bf16-mixed, matmul_precision high (TF32 on, deliberately not touched), protocol
# protein-anything, and both shipped design checkpoints. Nothing here overrides a model knob;
# `designs`/`batch` only choose how many designs the process makes and how many share a loop, and
# `steps` is 500 everywhere except the deliberate smoke arm.
#
# batch != 1 is a SATURATION PROBE ONLY. The production batch is 1 on both sides: BoltzGen's batch
# path drifts 0.5-2.8 A Kabsch RMSD per slot (boltzgen-batch-threshold-dead-end, 3 independent
# runs), so a batched design is not the same design. The sat arms answer "how much of the GPU does
# the production configuration leave idle" and are never a recommendation.
#
# The design counts are not round numbers by accident. The shipped default is two design
# checkpoints, and the switch lands on the loop at ceil(0.5 * n_loops); that loop is dropped along
# with the cold loop, so `designs` is chosen to leave at least two clean warm loops after both are
# gone. num_designs=3 at batch 1 would leave exactly one and the arm would refuse to report.
ARMS = {
    #  name          rung  batch  designs  steps  extra CLI
    "smoke":        ("R3", 1, 4, 20, []),
    "headline":     ("R3", 1, 6, 500, []),
    "replicate":    ("R3", 1, 6, 500, []),
    "sat2":         ("R3", 2, 12, 500, []),
    "sat4":         ("R3", 4, 24, 500, []),
    "nokern":       ("R3", 1, 5, 500, ["--use_kernels", "false"]),
    "r4":           ("R4", 1, 5, 500, []),
}


def nvsmi_static():
    q = ("name,driver_version,power.limit,power.max_limit,memory.total,"
         "clocks.max.sm,persistence_mode")
    out = subprocess.run(["nvidia-smi", "--query-gpu=" + q,
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()[0]
    keys = ["gpu_name", "driver_version", "power_limit_W", "power_max_limit_W",
            "memory_total_MiB", "clocks_max_sm_MHz", "persistence_mode"]
    d = dict(zip(keys, [v.strip() for v in out.split(",")]))
    for k in ("power_limit_W", "power_max_limit_W"):
        try:
            d[k] = float(d[k])
        except (ValueError, KeyError):
            pass
    return d


def sample_power(stop, sink):
    """nvidia-smi at 200 ms, the sampler the fixture-selection sweep used."""
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


def measure_idle(seconds=4.0):
    samples, stop = [], threading.Event()
    th = threading.Thread(target=sample_power, args=(stop, samples), daemon=True)
    th.start()
    time.sleep(seconds)
    stop.set()
    th.join(timeout=5)
    w = [s[1] for s in samples]
    u = [s[2] for s in samples]
    return {"idle_W_median": round(statistics.median(w), 1) if w else None,
            "idle_util_pct_median": round(statistics.median(u), 1) if u else None,
            "idle_n_samples": len(w)}


def proc_utime(pid):
    """Jiffies of user CPU. py-spy needs SYS_PTRACE, which vast.ai does not grant; a utime that
    advances at ~100 jiffies/s is the one-core spin of the Blackwell cuEquivariance hang."""
    try:
        f = pathlib.Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()
        return int(f[11]) + int(f[12])
    except Exception:                                             # noqa: BLE001
        return None


def read_chains(path):
    """(chain -> residue count, n_atoms, n_nonfinite) from a written mmCIF.

    gemmi is a boltzgen dependency, so it is in the venv; the field-index fallback is the parse the
    Tenstorrent-side re-verification used on these same files.
    """
    try:
        import gemmi
        blk = gemmi.cif.read(str(path)).sole_block()
        tab = blk.find("_atom_site.", ["label_asym_id", "label_seq_id",
                                       "Cartn_x", "Cartn_y", "Cartn_z"])
        res, atoms, bad = {}, 0, 0
        for row in tab:
            atoms += 1
            res.setdefault(row[0], set()).add(row[1])
            for j in (2, 3, 4):
                try:
                    v = float(row[j])
                except ValueError:
                    bad += 1
                    continue
                if not math.isfinite(v):
                    bad += 1
        return {k: len(v) for k, v in res.items()}, atoms, bad
    except Exception:                                             # noqa: BLE001
        res, atoms, bad = {}, 0, 0
        for line in path.read_text().splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            f = line.split()
            atoms += 1
            res.setdefault(f[6], set()).add(f[8])
            for tok in f[10:13]:
                try:
                    v = float(tok)
                except ValueError:
                    continue
                if not math.isfinite(v):
                    bad += 1
        return {k: len(v) for k, v in res.items()}, atoms, bad


def expected_names(rung, total):
    """The DesignWriter's own naming: <stem>_<global_idx zero-padded to len(str(total-1))>.cif."""
    if total <= 1:
        return ["bg_%s.cif" % rung]
    nd = len(str(total - 1))
    return ["bg_%s_%0*d.cif" % (rung, nd, i) for i in range(total)]


def guard(out_dir, rung, total):
    """Guard on the OUTPUT. Returns (ok, why, detail).

    Requires exactly the files the writer should have produced, and in EVERY one of them two chains
    sized {100 designed, target}, every coordinate finite. The pipeline's top-level target-only copy
    is deliberately not what gets checked: checking that file is how a run which produced no binder
    once passed a finiteness check.
    """
    d = pathlib.Path(out_dir) / "intermediate_designs"
    if not d.is_dir():
        return False, "no intermediate_designs dir under %s" % out_dir, {}
    want_names = expected_names(rung, total)
    have = sorted(p.name for p in d.glob("*.cif"))
    missing = [n for n in want_names if not (d / n).exists()]
    if missing:
        return False, "missing %d of %d design files (%s...)" % (
            len(missing), len(want_names), missing[:3]), {"have": have}
    want = sorted([BINDER_RES, TARGET_RES[rung]])
    detail = {}
    for n in want_names:
        chains, atoms, bad = read_chains(d / n)
        detail[n] = {"chains": chains, "atoms": atoms, "nonfinite": bad}
        if len(chains) != 2:
            return False, "%s: expected 2 chains, got %d" % (n, len(chains)), detail
        if sorted(chains.values()) != want:
            return False, "%s: chain sizes %s != %s" % (n, sorted(chains.values()), want), detail
        if bad:
            return False, "%s: %d non-finite coords" % (n, bad), detail
        if atoms == 0:
            return False, "%s: 0 atoms" % n, detail
    return True, "", detail


def run_arm(name, gpu):
    rung, batch, designs, steps, extra = ARMS[name]
    out_dir = WORK / "out" / ("bgg_%s_%s" % (gpu, name))
    shutil.rmtree(out_dir, ignore_errors=True)
    fixture = WORK / "perf" / "dsfix" / "fixtures" / ("bg_%s.yaml" % rung)
    cmd = [PY, RUNNER, "run", str(fixture),
           "--output", str(out_dir),
           "--protocol", "protein-anything",
           "--steps", "design",
           "--num_designs", str(designs),
           "--diffusion_batch_size", str(batch),
           "--no_subprocess"] + extra
    if steps != 500:
        cmd += ["--config", "design", "sampling_steps=%d" % steps]

    idle = measure_idle()
    samples, stop = [], threading.Event()
    th = threading.Thread(target=sample_power, args=(stop, samples), daemon=True)
    th.start()
    t_start = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, cwd=str(WORK))

    stamps, loops, switches, counters, env, tail = [], [], [], None, None, []
    timed_out = [False]

    def watchdog():
        t0 = time.time()
        u0 = proc_utime(p.pid)
        while p.poll() is None:
            if time.time() - t0 > TIMEOUT_S:
                timed_out[0] = True
                u1 = proc_utime(p.pid)
                if u0 is not None and u1 is not None:
                    hang["utime_jiffies_per_s"] = round((u1 - u0) / (time.time() - t0), 1)
                p.kill()
                return
            time.sleep(2.0)

    hang = {}
    wd = threading.Thread(target=watchdog, daemon=True)
    wd.start()
    for line in p.stdout:
        if line.startswith("STEP "):
            stamps.append(float(line.split()[1]))
        elif line.startswith("DESIGN "):
            f = line.split()
            loops.append((int(f[1]), float(f[2]), float(f[3])))
        elif line.startswith("CKPTSWITCH "):
            switches.append(float(line.split()[1]))
        elif line.startswith("COUNTERS "):
            counters = json.loads(line[len("COUNTERS "):])
        elif line.startswith("ENV "):
            env = json.loads(line[len("ENV "):])
        else:
            tail.append(line)
            if len(tail) > 200:
                tail.pop(0)
    p.wait()
    stop.set()
    th.join(timeout=5)
    wall = time.time() - t_start

    n_loops = math.ceil(designs / batch)
    rec = {"arm": name, "gpu": gpu, "rung": rung, "batch": batch,
           "num_designs": designs, "n_loops_expected": n_loops, "sampling_steps": steps,
           "extra_cli": " ".join(extra), "rc": p.returncode,
           "proc_wall_s": round(wall, 2), "n_loops_timed": len(loops),
           "n_steps_total": len(stamps), "n_ckpt_switches": len(switches),
           "counters": counters, "env": env, "cmd": " ".join(cmd),
           "timed_out": timed_out[0]}
    rec.update(idle)

    if timed_out[0]:
        # The Blackwell signature, measured rather than assumed: GPU pinned at "100 % util" while
        # drawing a fraction of the limit, one host core spinning, log frozen before the first step.
        tailwin = [s for s in samples if s[0] >= time.time() - HANG_WINDOW_S]
        hang.update({
            "power_W_median_tail": round(statistics.median([s[1] for s in tailwin]), 1)
            if tailwin else None,
            "util_pct_median_tail": round(statistics.median([s[2] for s in tailwin]), 1)
            if tailwin else None,
            "n_tail_samples": len(tailwin),
        })
        rec["hang"] = hang
        rec["ok"] = False
        rec["why"] = ("timed out after %d s with %d step stamps and %d loops"
                      % (TIMEOUT_S, len(stamps), len(loops)))
        rec["tail"] = "".join(tail)[-4000:]
        return rec, out_dir

    ok, why, detail = guard(out_dir, rung, designs)
    rec["output_ok"], rec["output_why"], rec["output_detail"] = ok, why, detail

    if len(loops) != n_loops:
        rec["ok"] = False
        rec["why"] = "timed %d denoising loops, expected %d" % (len(loops), n_loops)
        rec["tail"] = "".join(tail)[-4000:]
        return rec, out_dir

    # Warm set: drop loop 0 (cold: kernel autotune, allocator growth, cuDNN plans) and any loop
    # whose window contains a checkpoint switch. The shipped default is two design checkpoints, so
    # one mid-run loop pays a full torch.load + load_state_dict inside its predict_step. That is a
    # real production cost, reported separately, but it is not a per-design compute measurement.
    warm = [(i, t0, t1) for (i, t0, t1) in loops
            if i > 0 and not any(t0 <= s <= t1 for s in switches)]
    if len(warm) < 2:
        rec["ok"] = False
        rec["why"] = "only %d warm loops after dropping cold + checkpoint-switch" % len(warm)
        rec["tail"] = "".join(tail)[-4000:]
        return rec, out_dir

    per_loop = [t1 - t0 for (_, t0, t1) in warm]
    per_design = [w / batch for w in per_loop]
    by_idx = {i: (t0, t1) for (i, t0, t1) in loops}
    gaps = [by_idx[i][0] - by_idx[i - 1][1] for (i, _, _) in warm if i - 1 in by_idx]

    step_ms = []
    for (_, t0, t1) in warm:
        s = [x for x in stamps if t0 <= x <= t1]
        step_ms += [(s[k + 1] - s[k]) * 1000 for k in range(3, len(s) - 1)]

    win = [s for s in samples if any(t0 <= s[0] <= t1 for (_, t0, t1) in warm)]
    med_design = statistics.median(per_design)
    med_gap = statistics.median(gaps) if gaps else 0.0
    e2e = med_design + med_gap / batch

    rec.update({
        "n_warm_loops": len(warm),
        "warm_loop_idx": [i for (i, _, _) in warm],
        "s_per_loop_median": round(statistics.median(per_loop), 4),
        "s_per_design_median": round(med_design, 4),
        "s_per_design_min": round(min(per_design), 4),
        "s_per_design_max": round(max(per_design), 4),
        "s_per_design_spread_pct": round(100 * (max(per_design) - min(per_design)) / med_design, 3),
        "s_per_design_all": [round(x, 4) for x in per_design],
        "designs_per_hour": round(3600 / med_design, 1),
        "inter_loop_gap_s_median": round(med_gap, 4),
        "s_per_design_e2e": round(e2e, 4),
        "designs_per_hour_e2e": round(3600 / e2e, 1),
        "cold_loop_s": round(loops[0][2] - loops[0][1], 4),
        "load_s_to_first_design": round(loops[0][1] - t_start, 2),
        "ckpt_switch_loop_s": [round(t1 - t0, 4) for (i, t0, t1) in loops
                               if any(t0 <= s <= t1 for s in switches)],
        "step_ms_median": round(statistics.median(step_ms), 4) if step_ms else None,
        "step_ms_min": round(min(step_ms), 4) if step_ms else None,
        "step_ms_max": round(max(step_ms), 4) if step_ms else None,
        "n_step_intervals": len(step_ms),
        "power_W_median": round(statistics.median([s[1] for s in win]), 1) if win else None,
        "power_W_min": round(min(s[1] for s in win), 1) if win else None,
        "power_W_max": round(max(s[1] for s in win), 1) if win else None,
        "util_pct_median": round(statistics.median([s[2] for s in win]), 1) if win else None,
        "util_pct_min": round(min(s[2] for s in win), 1) if win else None,
        "mem_MiB_max": round(max(s[3] for s in win), 0) if win else None,
        "clocks_sm_MHz_median": round(statistics.median([s[4] for s in win]), 0) if win else None,
        "n_power_samples": len(win),
    })

    # Fast-path engagement, from counters rather than from the flag that was passed.
    # Verified statically before the run: the design step's cuEquivariance calls all live in the
    # trunk (PairformerModule + msa module, use_miniformer=false in the shipped checkpoint), and the
    # 500-step denoising loop is a 24-layer token transformer through torch SDPA with
    # diffusion_pairformer_args.num_blocks=0. So cueq counts are per-recycle, not per-step, and a
    # large torch_sdpa count is expected and correct rather than a fallback.
    c = counters or {}
    if name == "nokern":
        kern_ok = (c.get("cueq_trimul", 1) == 0 and c.get("cueq_triatt", 1) == 0
                   and c.get("triatt_torch_fallback", 0) > 0)
        kern_why = "" if kern_ok else "--use_kernels false did not take: %s" % json.dumps(c)
    else:
        kern_ok = (c.get("cueq_trimul", 0) > 0 and c.get("cueq_triatt", 0) > 0
                   and c.get("triatt_torch_fallback", 1) == 0)
        kern_why = "" if kern_ok else "cuEquivariance not engaged: %s" % json.dumps(c)
    rec["kernels_ok"], rec["kernels_why"] = kern_ok, kern_why
    # Not a gate: MiniformerModule/MiniformerNoSeqModule accept use_kernels and drop it before
    # their triangular update, so a nonzero difference here is an upstream plumbing fact about
    # blocks the shipped checkpoint does not use. Recorded so it is visible either way.
    rec["trimul_not_through_kernel"] = (c.get("trimul_forward_total", 0)
                                        - c.get("cueq_trimul", 0)) if c else None

    steps_ok = len(stamps) == n_loops * steps
    rec["steps_ok"] = steps_ok
    rec["ok"] = bool(ok and kern_ok and steps_ok and p.returncode == 0)
    rec["why"] = "; ".join(x for x in [
        why, kern_why,
        "" if steps_ok else "saw %d step stamps, expected %d" % (len(stamps), n_loops * steps),
        "" if p.returncode == 0 else "rc=%d" % p.returncode] if x)
    if not rec["ok"]:
        rec["tail"] = "".join(tail)[-4000:]
    return rec, out_dir


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    gpu, names = sys.argv[1], sys.argv[2:]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        print("unknown arms: %s (have %s)" % (unknown, list(ARMS)))
        sys.exit(2)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "structures").mkdir(exist_ok=True)
    out_jsonl = RESULTS / ("bgg_%s.jsonl" % gpu)
    static = nvsmi_static()
    print("[bgg] box: %s" % json.dumps(static), flush=True)

    bad = 0
    for name in names:
        print("[bgg] === arm %s : %s ===" % (name, time.strftime("%FT%TZ", time.gmtime())),
              flush=True)
        rec, out_dir = run_arm(name, gpu)
        rec["box"] = static
        rec["utc"] = time.strftime("%FT%TZ", time.gmtime())
        with out_jsonl.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        keep = sorted((pathlib.Path(out_dir) / "intermediate_designs").glob("bg_*_*.cif")) \
            if (pathlib.Path(out_dir) / "intermediate_designs").is_dir() else []
        if keep:
            shutil.copy(keep[-1], RESULTS / "structures" /
                        ("%s_%s_%s" % (gpu, name, keep[-1].name)))
        bad += 0 if rec.get("ok") else 1
        print("[bgg] %s ok=%s  %s s/design (n=%s, spread %s%%)  %s ms/step  %s W  util %s%%  "
              "designs/h %s  output_ok=%s  cueq=%s/%s  %s"
              % (name, rec.get("ok"), rec.get("s_per_design_median"), rec.get("n_warm_loops"),
                 rec.get("s_per_design_spread_pct"), rec.get("step_ms_median"),
                 rec.get("power_W_median"), rec.get("util_pct_median"),
                 rec.get("designs_per_hour"), rec.get("output_ok"),
                 (rec.get("counters") or {}).get("cueq_trimul"),
                 (rec.get("counters") or {}).get("cueq_triatt"),
                 rec.get("why", "")), flush=True)
    print("[bgg] DONE, %d of %d arms not ok" % (bad, len(names)), flush=True)


if __name__ == "__main__":
    main()
