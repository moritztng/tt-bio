"""Corrected BoltzGen TT batch probe: is the card idle at batch 1?

Supersedes the batch arms of `bg_ladder.py`, which passed a fixed `--num_designs 2` while varying
`diffusion_batch_size`. That combination is a no-op: BoltzGen ran 2 designs in 2 separate denoising
loops at every b, so the arms re-measured b=1 three times and any T(b) = b/(500*step) computed from
them divided a b=1 wall by a batch that never happened.

Plan sec.4.4 specifies `--num_designs <b>` alongside `diffusion_batch_size=<b>`. This uses
`num_designs = 2*b` so there are exactly TWO loops of b designs each: the first is discarded whole
(compile), the second is the measurement. The number of loops and the number of written designs are
both recorded and asserted, so a repeat of the same silent no-op fails loudly instead of producing
a plausible number.

Per the closed finding in `boltzgen-batch-threshold-dead-end` the device diffusion path is not
batch-invariant, so this is an idleness probe only and never a production recommendation.
"""
import json, os, pathlib, re, statistics, subprocess, sys, time

OUT = pathlib.Path("perf/dsfix/results/bg_batchprobe.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
STEPS = 60
PY = "/home/ttuser/tt-bio-dev/env/bin/python3"
DIFF = re.compile(r"diff (\d+)/(\d+)")
BATCH = re.compile(r"batch (\d+)/(\d+)")
POINTS = [(r, b) for r in ["R0", "R1", "R2", "R3"] for b in (1, 2, 4)]

done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            done.add((r["rung"], r["batch"]))


def run(rung, b):
    out_dir = "/tmp/bgp_%s_b%d" % (rung, b)
    subprocess.run(["rm", "-rf", out_dir], check=False)
    cmd = [PY, "-u", "-m", "tt_bio.main", "design",
           "perf/dsfix/fixtures/bg_%s.yaml" % rung,
           "--model", "boltzgen", "--steps", "design",
           "--num_designs", str(2 * b), "--out_dir", out_dir,
           "--config", "design", "sampling_steps=%d" % STEPS,
           "--config", "design", "diffusion_batch_size=%d" % b,
           "--debug", "--log"]
    env = dict(os.environ, TT_VISIBLE_DEVICES="0",
               TT_BIO_LEASE_HOLDER="worker:design-representative-fixtures",
               PYTHONPATH=os.getcwd(), PYTHONUNBUFFERED="1")
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env)
    stamps, cur, tail = {}, 0, []
    for line in p.stdout:
        tail.append(line)
        if len(tail) > 80:
            tail.pop(0)
        mb = BATCH.search(line)
        if mb:
            cur = int(mb.group(1))
        if DIFF.search(line):
            stamps.setdefault(cur, []).append(time.time())
    p.wait()
    n_cif = len(list(pathlib.Path(out_dir).rglob("intermediate_designs/*.cif")))
    return stamps, time.time() - t0, p.returncode, "".join(tail), n_cif


for rung, b in POINTS:
    if (rung, b) in done:
        print("[bgp] %s b=%d cached" % (rung, b), flush=True)
        continue
    stamps, wall, rc, tail, n_cif = run(rung, b)
    blocks = sorted(k for k, v in stamps.items() if len(v) >= STEPS - 2)
    # The probe is only valid if the batch actually took effect: two loops, 2*b designs written.
    valid = len(blocks) >= 2 and n_cif == 2 * b
    if not valid:
        print("[bgp] %s b=%d INVALID rc=%d blocks=%s cifs=%d (want 2 blocks, %d cifs)\n%s"
              % (rung, b, rc, [len(stamps[k]) for k in sorted(stamps)], n_cif, 2 * b, tail[-1500:]),
              flush=True)
        continue
    ts = stamps[blocks[-1]]
    per = [ts[i + 1] - ts[i] for i in range(3, len(ts) - 1)]
    med = statistics.median(per)
    rec = {"rung": rung, "batch": b, "num_designs": 2 * b, "n_cif_written": n_cif,
           "loops_seen": len(blocks), "steps_per_loop": [len(stamps[k]) for k in sorted(stamps)],
           "step_ms_median": round(med * 1000, 3),
           "step_ms_min": round(min(per) * 1000, 3), "step_ms_max": round(max(per) * 1000, 3),
           "n_steps": len(per), "designs_per_s_at_500": round(b / (500 * med), 5),
           "proc_wall_s": round(wall, 1),
           "host": "qb1", "card": 0, "ttnn": "0.67.4"}
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("[bgp] %s b=%d  %.2f ms/step  T=%.5f  loops=%d cifs=%d"
          % (rung, b, rec["step_ms_median"], rec["designs_per_s_at_500"], len(blocks), n_cif),
          flush=True)
print("[bgp] ALL DONE", flush=True)
