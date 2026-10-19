"""Run bc4-quality's stock-GPU reference distribution on the rented box, from its own request file.

    python bc4_quality_ref.py --req quality-req.json --out DIR --deadline EPOCH

The request (state/bc4/QUALITY-REF-REQUEST.json, schema bc4-quality-ref-request/1) names every campaign and the exact
argv it wants. We rewrite three things and nothing else: the interpreter, the runner path (both live elsewhere on the
box than in the repo the request was written against) and the output directory. Tiers run P0 first, then P1, then P2,
three campaigns at a time, since a campaign's designs are a function of (sha, settings, seed, trajectory) and packing
them onto one card changes the wall clock only. This never overlaps a timing arm: box_bc4.sh runs it last.
"""
import argparse, json, os, subprocess, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--req", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--deadline", type=float, required=True, help="epoch seconds")
ap.add_argument("--runner", default="/root/kc/bc2_gpu.py")
ap.add_argument("--python", default="/opt/bc2-venv/bin/python")
a = ap.parse_args()

req = json.loads(a.req.read_text())
a.out.mkdir(parents=True, exist_ok=True)
conc = req.get("concurrency") or {}
width = int(conc.get("suggested") or 3)
env = dict(os.environ)
env.update({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30"})
env.update(conc.get("env") or {})
TIER = {"P0": 0, "P1": 1, "P2": 2}
runs = sorted(req.get("runs", []), key=lambda r: (TIER.get(r.get("tier", "P9"), 9), r.get("id", "")))


def say(*x):
    print(time.strftime("%FT%TZ", time.gmtime()), *x, flush=True)


def argv(r):
    cmd = [a.runner if c.endswith("bc2_gpu.py") else c for c in r["cmd"]]
    cmd[0] = a.python
    d = a.out / r["id"].replace("/", "_")
    if "--out" in cmd:
        cmd[cmd.index("--out") + 1] = str(d)
    return cmd, d


def prune(d):
    """The grade reads csv and jsonl. The structures are large and the session rsyncs this tree every 3 minutes."""
    for pat in ("*.cif", "*.pdb", "*.pkl", "*.npz"):
        for f in d.rglob(pat):
            f.unlink()


def reap(live, done, block):
    for proc, r, d, t0 in list(live):
        if proc.poll() is not None:
            live.remove((proc, r, d, t0))
            prune(d)
            done.append(dict(id=r["id"], tier=r.get("tier"), rc=proc.returncode, wall_s=round(time.time() - t0, 1)))
            say("finished", r["id"], "rc", proc.returncode, "in", round(time.time() - t0), "s")
    if block and live:
        time.sleep(10)


live, done, i = [], [], 0
while i < len(runs) or live:
    reap(live, done, block=len(live) >= width or i >= len(runs))
    if len(live) >= width or i >= len(runs):
        continue
    r = runs[i]; i += 1
    est = float(r.get("est_wall_s") or 900)
    if time.time() + est > a.deadline - 300:
        say("skipping", r["id"], f"({est:.0f}s estimate does not fit before the deadline)")
        done.append(dict(id=r["id"], tier=r.get("tier"), rc=None, skipped="no time"))
        continue
    cmd, d = argv(r)
    d.mkdir(parents=True, exist_ok=True)
    say("starting", r["id"], " ".join(cmd))
    live.append((subprocess.Popen(cmd, env=env, cwd=str(Path(a.runner).parent),
                                  stdout=open(d / "run.log", "w"), stderr=subprocess.STDOUT), r, d, time.time()))

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
inst = Path("/root/kc/instance")
(a.out / "MANIFEST.json").write_text(json.dumps(dict(
    schema="bc4-quality-ref-result/1", request=req.get("schema"), gpu=gpu,
    instance=inst.read_text().strip() if inst.exists() else None,
    ok=[d["id"] for d in done if d.get("rc") == 0],
    missing=[d["id"] for d in done if d.get("rc") != 0], runs=done), indent=1) + "\n")
say("quality reference finished:", sum(1 for d in done if d.get("rc") == 0), "of", len(runs), "runs")
