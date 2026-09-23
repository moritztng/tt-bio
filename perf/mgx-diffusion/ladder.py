"""Fold a plan of (model, tokens, samples, recycles, steps) points on one pinned chip.

The sample and recycle axes of `tt-bio predict`, which the size ladder holds at 1 sample and the
model's default recycles. Every point folds the size ladder's own fixture (cdk2x2_<tokens>,
single sequence, seed 0) through the shipped CLI, so what is measured is what a user runs.

One JSON line per point in perf/mgx-diffusion/runs.jsonl, carrying:
  runtime_s   the fold's own results.json time (model load and startup excluded)
  aiclk       sampled DURING the fold on the pinned chip (release_gate._clock_during)
  load        1-min host load sampled during the fold, min/max
  dram        with probe=1 only: the peak DRAM used and the tag it was reached at, plus the
              peak inside the diffusion + confidence phase and inside the trunk. The probe
              drains the pipeline at every tag, so a probe=1 runtime is NOT a timing
  progress    trunk cycles actually run (the progress stream's trunk total), and seconds per
              phase (trunk, diffusion, confidence...) from the fold log's stage lines
  n_struct    structures written, which must equal the samples asked for
  cold        the kernel cache grew during the fold, so it compiled and its runtime is not a
              timing; a cold timing point (probe off) is folded again at once and both kept

Plan lines: `<model> <tokens> <samples> [steps=N] [recycles=N] [probe=1] [mps=N] [guard=0]`.
steps defaults to the model's own default (production), `guard=0` turns the size guard off.
A point already recorded on this engine tree, chip and host is skipped, so a chain resumes.

    TT_VISIBLE_DEVICES=<c> TT_BIO_LEASE_CARDS=<c> TT_BIO_LEASE_HOLDER=worker:mgx-diffusion \
        python perf/mgx-diffusion/ladder.py perf/mgx-diffusion/plan.txt
"""
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import release_gate as rg  # noqa: E402

HERE = ROOT / "perf" / "mgx-diffusion"
RUNS = HERE / "runs.jsonl"
WORK = Path(os.environ.get("MGX_DIFFUSION_WORK", HERE / "work"))
TIMEOUT = float(os.environ.get("MGX_DIFFUSION_TIMEOUT", 7200))
HOST_THREADS = int(os.environ.get("MGX_DIFFUSION_HOST_THREADS", 2))
KEYS = ("steps", "recycles", "probe", "mps", "guard")

git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def parse(line):
    f = line.split("#")[0].split()
    if not f:
        return None
    p = {"model": f[0], "tokens": int(f[1]), "samples": int(f[2])}
    for kv in f[3:]:
        k, v = kv.split("=")
        assert k in KEYS, kv
        p[k] = int(v)
    return p


def point_id(p):
    return (p["model"], p["tokens"], p["samples"]) + tuple(p.get(k) for k in KEYS)


class Load:
    """1-min loadavg on a thread; the speed bar voids a rung whose load passed 1.5x nproc."""

    def __init__(self):
        self.xs, self.stop = [], threading.Event()
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            self.xs.append(os.getloadavg()[0])
            self.stop.wait(15)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join()

    def cell(self):
        return {"min": round(min(self.xs), 1), "max": round(max(self.xs), 1),
                "nproc": os.cpu_count()} if self.xs else None


DRAM = re.compile(r"\[DRAM\] (.+?): ([\d.]+) GiB used \(of ([\d.]+) GiB\) maxfree=(\d+)MiB")
SAMPLE_TAGS = re.compile(r"(?i)diffus|edm|sampl|denois|rollout|confid|dit\b|atom")


def dram_cell(path):
    """Peak over all tags, and split into the sample-axis phase and the rest."""
    if not path.exists():
        return None
    rows = [(m[1], float(m[2]), float(m[3]), int(m[4]))
            for m in DRAM.finditer(path.read_text(errors="replace"))]
    if not rows:
        return None
    top = max(rows, key=lambda r: r[1])
    samp = [r for r in rows if SAMPLE_TAGS.search(r[0])]
    rest = [r for r in rows if not SAMPLE_TAGS.search(r[0])]
    pk = lambda rs: ({"gib": max(r[1] for r in rs), "tag": max(rs, key=lambda r: r[1])[0]}
                     if rs else None)
    return {"peak_gib": top[1], "peak_tag": top[0], "total_gib": top[2],
            "min_maxfree_mib": min(r[3] for r in rows), "sample_phase": pk(samp),
            "other": pk(rest), "n_tags": len(rows)}


STAGE = re.compile(r"^(\d\d):(\d\d):(\d\d)\s+\[[^\]]+\]\s+([a-z][a-z ]*?)(?: (\d+)/(\d+))?\s*$", re.M)


def progress_cell(cap, log_text):
    """Trunk cycles run, and seconds per phase, off the live progress stream.

    The captured events carry the counts but no clock, so the phase seconds come from the
    fold log's own timestamped stage lines (the same stream, as the display printed it).
    """
    evs = []
    if cap.exists():
        for ln in cap.read_text(errors="replace").splitlines():
            try:
                evs.append(json.loads(ln))
            except ValueError:
                pass
    trunk = [e for e in evs if e.get("stage") == "trunk"]
    diff = [e for e in evs if e.get("stage") == "diffusion"]
    first = {}
    for m in STAGE.finditer(log_text):
        t = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
        first.setdefault(m[4].strip(), t)
    order = sorted(first.items(), key=lambda kv: kv[1])
    phase = {}
    for (name, t), (_n, t_next) in zip(order, order[1:]):
        phase[name] = (t_next - t) % 86400
    return {"trunk_total": max((int(e.get("total") or 0) for e in trunk), default=None),
            "trunk_steps_seen": len({e.get("step") for e in trunk}),
            "diffusion_total": max((int(e.get("total") or 0) for e in diff), default=None),
            "phase_s": phase}


def cache_files():
    root = os.environ.get("TT_METAL_CACHE")
    return sum(len(f) for _d, _s, f in os.walk(root)) if root else 0


def fold(p, ident):
    label = "-".join(str(x) for x in point_id(p) if x is not None)
    out, log = WORK / f"out_{label}", WORK / f"{label}.log"
    cap, probe = WORK / f"progress_{label}.jsonl", WORK / f"dram_{label}.txt"
    subprocess.run(["rm", "-rf", str(out), str(cap), str(probe)])
    WORK.mkdir(parents=True, exist_ok=True)
    fixture = rg._size_ladder_fixture(p["model"], p["tokens"])
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(fixture), "--model", p["model"],
           "--single_sequence", "--diffusion_samples", str(p["samples"]),
           "--seed", str(rg.SEED), "--host_threads", str(HOST_THREADS), "--out_dir", str(out)]
    if p.get("steps") is not None:
        cmd += ["--sampling_steps", str(p["steps"])]
    if p.get("recycles") is not None:
        cmd += ["--recycling_steps", str(p["recycles"])]
    if p.get("mps") is not None:
        cmd += ["--max_parallel_samples", str(p["mps"])]
    env = dict(os.environ, TT_BIO_PROGRESS_CAPTURE=str(cap))
    if p.get("probe"):
        env["TT_BIO_DRAM_PEAK"] = str(probe)
    if p.get("guard") == 0:
        env["TT_BIO_SIZE_LIMIT"] = "0"
    n_cache, t0 = cache_files(), time.time()
    with open(log, "w") as fp, rg._clock_during() as clk, Load() as load:
        rc, timed_out = rg._run_fold(cmd, TIMEOUT, cwd=ROOT, stdout=fp, stderr=subprocess.STDOUT,
                                     env=env)
    cell = {**ident, **p, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "wall_s": round(time.time() - t0, 1), "aiclk": rg._aiclk_cell(clk),
            "load": load.cell(), "dram": dram_cell(probe), "cold": cache_files() > n_cache, "log": str(log)}
    text = rg._fold_log_text(log)
    cell["progress"] = progress_cell(cap, text)
    if timed_out:
        cell["error"] = f"timed out after {TIMEOUT:.0f}s"
    elif rc != 0:
        cell["refused" if rg._size_limit_refusal(text) else "error"] = (
            rg._size_limit_refusal(text) or f"exit {rc}: {rg._fold_error(text)}")
    from tt_bio.main import predict_results_dir_name
    res = out / predict_results_dir_name(p["model"], fixture.stem) / "results.json"
    if res.exists():
        rows = json.loads(res.read_text())
        ts = [r["runtime_s"] for r in rows if r.get("status") == "ok" and r.get("runtime_s")]
        cell["runtime_s"] = max(ts) if ts else None
        bad = [r.get("error") for r in rows if r.get("status") != "ok"]
        if bad and "error" not in cell:
            cell["error"] = str(bad[0])[:600]
    cell["n_struct"] = len(list(out.rglob("*.cif"))) if out.exists() else 0
    if cell.get("error") and "timed out" not in cell["error"]:
        cell["error_tail"] = "\n".join(text.strip().splitlines()[-6:])[-1200:]
    if not cell.get("error") and not cell.get("refused"):
        subprocess.run(["rm", "-rf", str(out)])     # keep failures, drop 25-CIF successes
    return cell


def main():
    plan = [q for q in map(parse, Path(sys.argv[1]).read_text().splitlines()) if q]
    commit = git("rev-parse", "HEAD")
    ident = {"commit": commit, "engine": git("rev-parse", f"{commit}:tt_bio"),
             "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
             "host_threads": HOST_THREADS}
    if git("status", "--porcelain", "--", "tt_bio", "scripts"):
        sys.exit(f"engine tree is dirty at {commit}: a point must name a commit")
    seen = set()
    if RUNS.exists():
        for c in map(json.loads, RUNS.read_text().splitlines()):
            same = (c["engine"], c["host"], c["card"]) == (ident["engine"], ident["host"], ident["card"])
            warm_or_final = not c.get("cold") or c.get("probe") or c.get("error") or c.get("refused")
            if same and warm_or_final:
                seen.add(point_id(c))
    for p in plan:
        if point_id(p) in seen:
            continue
        for _ in range(2):
            cell = fold(p, ident)
            with open(RUNS, "a") as fp:
                fp.write(json.dumps(cell, default=str) + "\n")
            print(json.dumps({k: cell.get(k) for k in (
                "model", "tokens", "samples", "recycles", "steps", "probe", "runtime_s",
                "wall_s", "n_struct", "cold", "error", "refused")}), flush=True)
            if not cell["cold"] or p.get("probe") or cell.get("error") or cell.get("refused"):
                break
        seen.add(point_id(p))


if __name__ == "__main__":
    main()
