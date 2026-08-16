"""BoltzGen TT ladder: per-step diffusion wall + dispatch share, per rung x per batch.

The `--debug --log` stream prints one `diff k/N` line per diffusion step. Timestamping those
lines as they arrive gives the per-step wall directly, with a real distribution from a single
run, so this needs no step-count differential and no repeated model load: one process per
(rung, batch) instead of six.

Two designs are run per point. The first is discarded whole -- it carries kernel compile for
every new shape. The steps of the second are the measurement, and the first 3 of those are
dropped as well so nothing straddles the design boundary.

Reported per point: median per-step wall, min-max spread, n, and the batch probe throughput.
Per the closed finding in `boltzgen-batch-threshold-dead-end` the device diffusion path is NOT
batch-invariant, so diffusion_batch_size is used here strictly as an idleness probe and never as
a production recommendation.
"""
import json, os, pathlib, re, statistics, subprocess, sys, time

# The machine facts every row must carry come from the environment, not from a literal: this
# ladder now runs on a Blackhole QuietBox and on the Wormhole Galaxy, and a row that does not name
# its host, card and wheel cannot carry a cross-architecture ratio.
OUT = pathlib.Path(os.environ.get("BG_OUT", "perf/dsfix/results/bg_tt.jsonl"))
OUT.parent.mkdir(parents=True, exist_ok=True)
STEPS = int(os.environ.get("BG_STEPS", "60"))
PY = os.environ.get("BG_PY", "/home/ttuser/tt-bio-dev/env/bin/python3")
DEV = os.environ.get("BG_DEV", "0")
HOST = os.environ.get("BG_HOST", "qb1")
CARD = os.environ.get("BG_CARD", "0")
TTNN = os.environ.get("BG_TTNN", "0.67.4")
LEASE = os.environ.get("BG_LEASE", "worker:design-representative-fixtures")
ONLY = os.environ.get("BG_POINTS", "")   # "R0:1:0,R1:1:0" -> rung:batch:trace
DIFF = re.compile(r"diff (\d+)/(\d+)")
BATCH = re.compile(r"batch (\d+)/(\d+)")

# (rung, batch, trace)
POINTS = []
for rung in ["R0", "R1", "R2", "R3", "R4"]:
    POINTS.append((rung, 1, False))
for rung in ["R0", "R1", "R2", "R3", "R4"]:
    for b in (2, 4):
        POINTS.append((rung, b, False))
for rung in ["R0", "R1", "R2", "R3", "R4"]:
    POINTS.append((rung, 1, True))
if ONLY:
    POINTS = [(r, int(bt), tr == "1") for r, bt, tr in (x.split(":") for x in ONLY.split(","))]

done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            done.add((r["rung"], r["batch"], r["trace"]))


def run(rung, b, trace):
    out_dir = "/tmp/bg_%s_b%d_%s" % (rung, b, "tr" if trace else "no")
    subprocess.run(["rm", "-rf", out_dir], check=False)
    cmd = [PY, "-u", "-m", "tt_bio.main", "design",
           "perf/dsfix/fixtures/bg_%s.yaml" % rung,
           "--model", "boltzgen", "--steps", "design",
           "--num_designs", "2", "--out_dir", out_dir,
           "--config", "design", "sampling_steps=%d" % STEPS,
           "--config", "design", "diffusion_batch_size=%d" % b,
           "--debug", "--log"]
    if trace:
        cmd.append("--diffusion_trace")
    env = dict(os.environ, TT_VISIBLE_DEVICES=DEV,
               TT_BIO_LEASE_HOLDER=LEASE,
               PYTHONPATH=os.getcwd(), PYTHONUNBUFFERED="1")
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env)
    # stamps[design_index] = [wall time of each `diff k/N` line]
    stamps, cur_batch, tail = {}, 0, []
    for line in p.stdout:
        tail.append(line)
        if len(tail) > 60:
            tail.pop(0)
        mb = BATCH.search(line)
        if mb:
            cur_batch = int(mb.group(1))
        md = DIFF.search(line)
        if md:
            stamps.setdefault(cur_batch, []).append(time.time())
    p.wait()
    return stamps, time.time() - t0, p.returncode, "".join(tail), out_dir


def sanity(out_dir, b):
    """A point counts only if it wrote the designs it claimed and their coords are finite."""
    cifs = list(pathlib.Path(out_dir).rglob("*.cif")) + list(pathlib.Path(out_dir).rglob("*.pdb"))
    if not cifs:
        return False, "no structure written", None
    n, bad = 0, 0
    for line in cifs[0].read_text().splitlines():
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
        return False, "0 atoms in %s" % cifs[0].name, None
    if bad:
        return False, "%d non-finite coords" % bad, n
    return True, "", n


for rung, b, trace in POINTS:
    if (rung, b, trace) in done:
        print("[bg] %s b=%d trace=%s cached" % (rung, b, trace), flush=True)
        continue
    stamps, wall, rc, tail, out_dir = run(rung, b, trace)
    warm_key = max(stamps) if stamps else None
    ts = stamps.get(warm_key, [])
    if len(ts) < 12:
        print("[bg] %s b=%d trace=%s FAILED rc=%d (%d stamps)\n%s"
              % (rung, b, trace, rc, len(ts), tail[-2000:]), flush=True)
        continue
    per = [ts[i + 1] - ts[i] for i in range(3, len(ts) - 1)]   # drop first 3 of the warm design
    ok, why, natoms = sanity(out_dir, b)
    rec = {"rung": rung, "batch": b, "trace": trace, "sampling_steps": STEPS,
           "step_ms_median": round(statistics.median(per) * 1000, 3),
           "step_ms_min": round(min(per) * 1000, 3), "step_ms_max": round(max(per) * 1000, 3),
           "n_steps": len(per), "designs_seen": sorted(stamps),
           "proc_wall_s": round(wall, 1),
           "designs_per_s": round(b / (500 * statistics.median(per)), 5),
           "sanity_ok": ok, "sanity_why": why, "atoms_written": natoms,
           "loadavg": float(open("/proc/loadavg").read().split()[0]),
           "host": HOST, "card": CARD, "ttnn": TTNN}
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("[bg] %s b=%d trace=%s  %.2f ms/step (n=%d, %.2f-%.2f)  sanity=%s"
          % (rung, b, trace, rec["step_ms_median"], len(per),
             rec["step_ms_min"], rec["step_ms_max"], ok), flush=True)
print("[bg] ALL DONE", flush=True)
