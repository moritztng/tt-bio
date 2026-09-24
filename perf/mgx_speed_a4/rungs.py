#!/usr/bin/env python3
"""Time one designer, embedding model or affinity surface at its speed-bar rungs, on one pinned chip.

The speed bar (docs/speed-bar.md, scripts/speed_bar.py) was only ever fed the predict models. This
feeds it the other eleven surfaces with the same identity rules: every rung of one model on one
chip, one host, one engine tree, one host thread cap, AICLK sampled DURING each timed unit and the
1-min load sampled beside it. Each rung gets one untimed warm-up (its kernels compile there), then
one timed unit, three at the sigma rung (512).

What a timed unit is, per family:

  design     one `tt-bio design` job at a fixed design count and the upstream step count, via
             perf/mgxscale/job.py so the fixture and the design count are the ones mgx-design-scale
             measured. runtime_s = wall / designs counted on disk (boltzgen writes no device
             runtime, so wall is the only unit all three share; it carries ~10 s of bring-up that
             does not grow with N, the lenient direction the bar already names). The warm-up runs
             the same shapes with fewer steps (pxdesign, rfd3) or one design (boltzgen, which
             diffuses one design per batch), because a warm-up is not a rung.
  affinity   one shipped-CLI job (`tt-bio affinity` for nesso1, `tt-bio predict --model boltz2` with
             an affinity property for boltz2-affinity) on the nesso1 ladder fixture: CDK2 tiled to N
             plus the ladder's 19-heavy-atom drug. runtime_s is the CLI's own seconds (nesso1's
             `seconds`, boltz2's `runtime_s` = structure + affinity), no model load.
  embed      in one process, the model loaded once, then per rung two untimed calls and timed
             blocks of back-to-back single-sequence calls long enough (>= 20 s) that tt-smi samples
             the clock several times. runtime_s is the block's mean seconds per call. Sequence:
             human mTOR (AF-P42345-F1) truncated to N, as mgx-embed-scale used.

    TT_VISIBLE_DEVICES=<c> TT_BIO_LEASE_CARDS=<c> TT_BIO_LEASE_HOLDER=worker:mgx-speed-a4 \\
        python perf/mgx_speed_a4/rungs.py <model> 512,640,768,896,1024,1280,1536

Resumes: a rung with its warm-up and enough timed units under the load ceiling is skipped. A rung
above 1024 that fails ends the walk (the rungs above it allocate more).
"""
import argparse
import json
import math
import os
import pathlib
import socket
import statistics
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from gate_guard import DEFAULT_LOAD_CEILING  # noqa: E402
from perf.clocksample import during  # noqa: E402

DESIGN = {  # fixed count, upstream steps (mgx-design-scale), warm-up overrides
    "pxdesign": {"designs": 8, "steps": 400, "warm": {"steps": 20}},
    "rfd3": {"designs": 2, "steps": 100, "batch_size": 2, "warm": {"steps": 10}},
    "boltzgen": {"designs": 2, "bg_steps": "design", "warm": {"designs": 1}},
}
AFFINITY = ("nesso1", "boltz2-affinity")
EMBED = ("esmc-300m", "esmc-600m", "esmc-6b", "saprot-35m", "saprot-650m", "saprot-1.3b")
BINDER = 80                      # designed residues; the rung is target + binder
# rfd3's default fixture crops chain A only (1008 residues on big_1831.cif), so past 1088 total
# the target spills into chain B explicitly, keeping the binder at 80.
RFD3_CONTIG = {1280: "A1-1008,B1-192,80", 1536: "A1-1008,B1-448,80"}
SIGMA_RUNG, SIGMA_REPS, EXTRA = 512, 3, 2
EMBED_BLOCK_S = 20.0
HOLDER = os.environ.get("TT_BIO_LEASE_HOLDER", "worker:mgx-speed-a4")


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


class loadwatch:
    """1-min loadavg / nproc every 10 s for the duration of one timed unit."""

    def __init__(self):
        self.s, self._stop = [], threading.Event()

    def _run(self):
        while not self._stop.is_set():
            self.s.append(float(open("/proc/loadavg").read().split()[0]) / os.cpu_count())
            self._stop.wait(10)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=5)

    def summary(self):
        s = sorted(self.s) or [float("nan")]
        return {"min": round(s[0], 3), "median": round(s[len(s) // 2], 3), "max": round(s[-1], 3),
                "n": len(self.s), "nproc": os.cpu_count()}


def design_unit(model, rung, warm, card, work):
    from perf.mgxscale import job
    cfg = {**DESIGN[model], **(DESIGN[model]["warm"] if warm else {})}
    a = argparse.Namespace(
        model=model, target=str(ROOT / "perf/bhdesign/targets/big_1831.cif"),
        target_res=rung if model == "rfd3" else rung - BINDER, designs=cfg["designs"],
        steps=cfg.get("steps", 400), binder=BINDER, batch_size=cfg.get("batch_size", 8),
        contig=RFD3_CONTIG.get(rung, "") if model == "rfd3" else "", bg_steps=cfg.get("bg_steps", ""),
        budget=0, card=card, holder=HOLDER, work=str(work), timeout=6 * 3600, host_threads=2,
        tag=f"_a4{'w' if warm else ''}")
    try:
        rec = job.run_job(a)
    except SystemExit as e:           # 75: the lease refused, another row opened the chip
        return {"contended": True, "error": f"design job exited {e.code}"}
    out = {"wall_s": rec["wall_s"], "n_designs": rec["n_designs"], "asked": rec["asked"],
           "steps": rec["steps"], "aiclk": rec.get("aiclk"), "dev_runtime_s": rec.get("runtime_s"),
           "load": {k: (round(v / rec["load"]["nproc"], 3) if k in ("min", "median", "max") else v)
                    for k, v in rec["load"].items()} if rec.get("load") else None}
    if rec["verdict"] != "PASS":
        out["error"] = f"{rec['verdict']} {rec.get('mechanism')}: {str(rec.get('diag'))[:300]}"
    elif rec["n_designs"]:
        out["runtime_s"] = round(rec["wall_s"] / rec["n_designs"], 2)
    return out


def affinity_unit(model, rung, warm, card, work):
    fx = ROOT / f"perf/nesso1/inputs/ladder/aa{rung}/cdk2_{rung}.yaml"
    out_dir = work / f"out_{model}_{rung}{'_w' if warm else ''}"
    subprocess.run(["rm", "-rf", str(out_dir)])
    py = [sys.executable, "-m", "tt_bio.main"]
    if model == "nesso1":
        # `affinity` has no --host_threads on main; the OMP/MKL caps below hold it at 2
        cmd = py + ["affinity", str(fx), "--out_dir", str(out_dir)]
    else:
        cmd = py + ["predict", str(fx), "--model", "boltz2", "--out_dir", str(out_dir),
                    "--single_sequence", "--host_threads", "2", "--accelerator", "tenstorrent",
                    "--seed", "0"]
        if warm:  # same shapes, fewer steps: compiles every kernel without paying 200 steps twice
            cmd += ["--sampling_steps", "10", "--sampling_steps_affinity", "10"]
    env = dict(os.environ, PYTHONPATH=str(ROOT), TT_BIO_SIZE_LIMIT="0",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    log = work / f"log_{model}_{rung}{'_w' if warm else ''}.txt"
    t0 = time.time()
    with open(log, "w") as fh, during() as clk, loadwatch() as ld:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    rec = {"wall_s": round(time.time() - t0, 1), "aiclk": clk.summary().get(0), "load": ld.summary(),
           "rc": rc, "cmd": " ".join(cmd[3:])}
    text = log.read_text(errors="replace")
    if rc == 75 or "is in use by worker:" in text:
        return {**rec, "contended": True, "error": "lease refused"}
    try:
        if model == "nesso1":
            import csv
            row = next(csv.DictReader(open(out_dir / "affinity.csv")))
            rec.update(runtime_s=round(float(row["seconds"]), 2), n_tokens=int(row["n_tokens"]),
                       affinity_pred_value=float(row["affinity_pred_value"]))
            if row.get("error"):
                raise ValueError(row["error"])
        else:
            d = json.loads(next(out_dir.rglob("results.json")).read_text())
            row = (d if isinstance(d, list) else d.get("results", [d]))[0]
            rec.update(runtime_s=row["runtime_s"], structure_runtime_s=row.get("structure_runtime_s"),
                       affinity_runtime_s=row.get("affinity_runtime_s"),
                       affinity_pred_value=row.get("affinity_pred_value"))
    except Exception as e:  # noqa: BLE001 -- anything unreadable is a failed rung, with its log tail
        rec["error"] = f"rc={rc} {type(e).__name__}: {e}; " + " | ".join(text.splitlines()[-6:])[:400]
    return rec


class Embedder:
    def __init__(self, model):
        import torch
        torch.set_grad_enabled(False)
        torch.set_num_threads(2)  # the record says host_threads 2; the CLI paths cap via OMP env
        sys.path.insert(0, str(ROOT / "perf" / "mgx_embed"))
        from accuracy import pdb_sequence
        pdb = os.environ.get("MGX_EMBED_PDB", str(pathlib.Path.home() / "scratch/mgxembed/mtor.pdb"))
        self.full = pdb_sequence(pdb)
        if model.startswith("saprot"):
            from tt_bio import saprot as mod
            self.m = mod.load_saprot(model)
        else:
            from tt_bio import esmc as mod
            self.m = mod.load_esmc(model)
        self.mod = mod

    def call(self, L):
        t = time.perf_counter()
        self.mod.embed_sequences(self.m, {"s": self.full[:L]}, batch_size=1)
        return time.perf_counter() - t

    def unit(self, rung, warm):
        if rung > len(self.full):
            return {"error": f"fixture carries {len(self.full)} residues"}
        try:
            if warm:
                t0 = time.time()
                for _ in range(2):  # compile, then trace capture on the second sighting
                    self.call(rung)
                return {"wall_s": round(time.time() - t0, 1)}
            k = max(5, math.ceil(EMBED_BLOCK_S / self.call(rung)))
            with during() as clk, loadwatch() as ld:
                walls = [self.call(rung) for _ in range(k)]
        except Exception as e:  # noqa: BLE001 -- an OOM at a rung is its coverage result
            return {"error": f"{type(e).__name__}: {str(e)[:400]}"}
        return {"runtime_s": round(statistics.mean(walls), 5), "calls": k,
                "call_min_s": round(min(walls), 5), "call_max_s": round(max(walls), 5),
                "wall_s": round(sum(walls), 2), "aiclk": clk.summary().get(0), "load": ld.summary()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=[*DESIGN, *AFFINITY, *EMBED])
    ap.add_argument("rungs")
    ap.add_argument("--out", default=str(HERE / "runs"))
    ap.add_argument("--work", default=str(pathlib.Path.home() / "mgx-speed-a4-work"))
    a = ap.parse_args()
    model, rungs = a.model, [int(x) for x in a.rungs.split(",")]
    card = os.environ["TT_VISIBLE_DEVICES"]
    if "," in card:
        sys.exit("one chip per model: TT_VISIBLE_DEVICES must name one card")
    if git("status", "--porcelain", "--", "tt_bio", "scripts"):
        sys.exit("engine tree is dirty: a timed rung must name a commit")
    family = "design" if model in DESIGN else "affinity" if model in AFFINITY else "embed"
    ident = {"model": model, "family": family, "commit": git("rev-parse", "HEAD"),
             "engine": git("rev-parse", "HEAD:tt_bio", "HEAD:scripts").replace("\n", " "),
             "host": socket.gethostname(), "card": card, "host_threads": 2}
    work = pathlib.Path(a.work).expanduser() / model
    work.mkdir(parents=True, exist_ok=True)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{model}.jsonl"
    prior = [json.loads(x) for x in log.read_text().splitlines() if x.strip()] if log.exists() else []
    done = [c for c in prior if (c["host"], c["card"], c["engine"]) ==
            (ident["host"], ident["card"], ident["engine"])]

    emb = Embedder(model) if family == "embed" else None

    def unit(rung, tag):
        t0 = time.time()
        warm = tag == "warmup"
        if family == "design":
            r = design_unit(model, rung, warm, card, work)
        elif family == "affinity":
            r = affinity_unit(model, rung, warm, card, work)
        else:
            r = emb.unit(rung, warm)
        cell = {**ident, "rung": rung, "tag": tag,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), **r}
        with open(log, "a") as fh:
            fh.write(json.dumps(cell, default=str) + "\n")
        print(json.dumps(cell, default=str), flush=True)
        return cell

    contended = lambda c: bool(c.get("contended"))
    failed = lambda c: bool(c.get("error")) and not contended(c)
    quiet = lambda c: (c.get("load") or {}).get("max", DEFAULT_LOAD_CEILING + 1) <= DEFAULT_LOAD_CEILING
    for rung in rungs:
        mine = [c for c in done if c["rung"] == rung and not contended(c)]
        reps = SIGMA_REPS if rung == SIGMA_RUNG else 1
        refused = 0
        while not any(c["tag"] == "warmup" for c in mine) and not any(map(failed, mine)) and refused < 6:
            w = unit(rung, "warmup")
            refused += contended(w)
            if contended(w):
                time.sleep(60)
            else:
                mine.append(w)
        timed = [c for c in mine if c["tag"] != "warmup"]
        budget = len(timed) + reps + EXTRA
        while (not any(map(failed, mine)) and sum(map(quiet, timed)) < reps and len(timed) < budget
               and refused < 6):
            c = unit(rung, f"rep{len(timed)}")
            if contended(c):
                refused += 1
                time.sleep(60)
                continue
            timed.append(c)
            mine.append(c)
        if any(map(failed, mine)) and rung > 1024:
            break


if __name__ == "__main__":
    main()
