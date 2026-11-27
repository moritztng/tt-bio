#!/usr/bin/env python3
"""Grade tt-bio's DiffDock-L against upstream on PoseBusters: dock, score, compare.

Three subcommands, in the order they run:

  pose_eval.py dock    --ids all --out <dir> --workers 4   # tt-bio poses, resumable
  pose_eval.py grade   --out <dir> --acc <dir>             # symmetry-corrected RMSD + validity
  pose_eval.py compare --tt <per_complex.json> --ref <per_complex.json> --out compare.json

The recipe is upstream's default: 20-step schedule, 19 actual steps, 10 samples, confidence
model on (``state/dfd/PROTOCOL.md`` section 1). Nothing here may reduce it.

**The grading half is the GPU side's code.** ``grade()`` below is the body of
``state/dfd/gpu/accuracy.py::grade`` (dfd-gpu-fair, the row that measured the upstream
reference on an RTX 3090) with the pose and reference directories passed in instead of
hard-coded. One protocol, both sides: the TT poses are scored by the same spyrmsd call,
against the same multi-conformer reference set, with the same PoseBusters config, so a
difference in the numbers is a difference in the poses.

Docking runs one worker process per chunk of complexes with ``isolate=True``, and the
driver counts the ``rank1.sdf`` files on disk rather than the worker's exit code: a
per-complex failure that returns rc=0 must not quietly leave the denominator.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

PB = Path(os.environ.get("DFD_PB", Path.home() / "scratch/dfd/pb/posebusters_benchmark_set"))
REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- statistics


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 4), round((c + h) / d, 4))


def boot_median_ci(xs, reps=2000, seed=42):
    if not xs:
        return (None, None)
    r = random.Random(seed)
    n = len(xs)
    med = sorted(statistics.median(r.choice(xs) for _ in range(n)) for _ in range(reps))
    return (round(med[int(0.025 * reps)], 3), round(med[int(0.975 * reps)], 3))


def paired_boot_ci(pairs, reps=10000, seed=42):
    """95 % CI for the paired difference mean(a) - mean(b), resampling complexes.

    ``pairs`` is a list of (a, b) over the SAME complexes, so the resample carries both
    halves of a complex together and the CI is paired, not two independent intervals.
    """
    if not pairs:
        return (None, None)
    r = random.Random(seed)
    n = len(pairs)
    deltas = []
    for _ in range(reps):
        sample = [pairs[r.randrange(n)] for _ in range(n)]
        deltas.append(sum(a - b for a, b in sample) / n)
    deltas.sort()
    return (round(deltas[int(0.025 * reps)], 4), round(deltas[int(0.975 * reps)], 4))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p for discordant counts b (tt only) and c (ref only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------- docking


def docked(out_dir: Path, i: str) -> bool:
    """A complex counts as docked only if rank1.sdf holds a READABLE molecule.

    A dock that raises while RDKit is already writing leaves a 0-byte rank1.sdf behind
    (seen on 5S8I_2LY: "Can't kekulize mol"). Treating that file as a result would both
    skip the complex on resume and feed an empty molecule to the grader.

    Size alone is not enough once the pool can be stopped and restarted: killing a worker
    mid-SDWriter leaves a non-empty but truncated file, which a size check reads as a result
    and the resume then never redocks. So parse it. 428 parses cost about a second in all,
    paid once per resume, and the alternative is a silently missing complex in the grade.
    """
    f = out_dir / i / "rank1.sdf"
    if not f.exists() or f.stat().st_size == 0:
        return False
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        return next(Chem.SDMolSupplier(str(f), sanitize=False, removeHs=False), None) is not None
    except Exception:
        return False


def ids_all():
    return sorted(d.name for d in PB.iterdir() if d.is_dir())


def smiles_of(i):
    from rdkit import Chem
    m = next(Chem.SDMolSupplier(f"{PB}/{i}/{i}_ligand.sdf", sanitize=True, removeHs=True), None)
    return None if m is None else Chem.MolToSmiles(Chem.RemoveHs(m))


def dock_chunk(ids, out_dir: Path, seed: int, device: str = "cpu", edge_tp: str = "gemm"):
    """Dock `ids` in THIS process with the default recipe. Used by the worker subprocess.

    *device* and *edge_tp* select the arm: ``cpu``/``gemm`` is the host-torch network that the
    428-complex set grade uses, ``tt``/``gemm`` the card's default form, ``tt``/``tile`` the
    fused edge-tile form. The recipe is identical in all three, which is the point: the arms
    differ in the network, not in the sampling.
    """
    sys.path.insert(0, str(REPO))
    from tt_bio.diffdock.job import DockJob, DockOptions
    from tt_bio.diffdock import run

    jobs = []
    for i in ids:
        smi = smiles_of(i)
        if smi is None:
            print(f"skip {i}: unreadable ligand", flush=True)
            continue
        jobs.append(DockJob(name=i, receptor_path=Path(f"{PB}/{i}/{i}_protein.pdb"), ligand=smi))
    kw = {}
    if device != "cpu":
        kw["device"] = device
        if "edge_tp" in {f.name for f in dataclasses.fields(DockOptions)}:
            kw["edge_tp"] = edge_tp
        elif edge_tp != "gemm":
            raise SystemExit("this checkout has no DockOptions.edge_tp: merge wk/dfd-perf-tile "
                             "before running the tile arm, do not grade a form you did not run")
    options = DockOptions(seed=seed, **kw)
    # isolate=False on a card: run.dock opens the device and loads the 30 M-parameter network
    # once for the whole list, and a per-complex subprocess would re-open the card per complex.
    summary = run.dock(jobs, options, out_dir, isolate=(device == "cpu"))
    for name, why in summary["failed"]:
        print(f"FAILED {name}: {why}", flush=True)
    return summary


def dock(ids, out_dir: Path, workers: int, chunk: int, seed: int, threads: int,
         device: str = "cpu", edge_tp: str = "gemm"):
    out_dir.mkdir(parents=True, exist_ok=True)
    if device != "cpu":
        # One chip, one process: the card is the serial resource, so the pool is a pool of one
        # and the whole list is one chunk. A clock sample runs for the length of the arm, because
        # a device timing without the AICLK it ran at is not a measurement.
        workers, chunk = 1, max(1, len(ids))
    todo = [i for i in ids if not docked(out_dir, i)]
    print(f"{len(ids)} complexes, {len(ids) - len(todo)} already docked, {len(todo)} to go",
          flush=True)
    chunks = [todo[s:s + chunk] for s in range(0, len(todo), chunk)]
    logs = out_dir / "logs"
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    running = []
    clk_samples, clk_stop, clk_thread = [], None, None
    if device != "cpu":
        import threading
        sys.path.insert(0, str(REPO))
        from tt_bio.diffdock.capture import sample_aiclk
        clk_stop = threading.Event()
        clk_thread = threading.Thread(
            target=sample_aiclk, daemon=True,
            args=(clk_stop, clk_samples, os.environ.get("TT_VISIBLE_DEVICES", "0")))
        clk_thread.start()
    t0 = time.time()
    while chunks or running:
        while chunks and len(running) < workers:
            part = chunks.pop(0)
            log = open(logs / f"dock-{part[0]}.log", "w")
            p = subprocess.Popen(
                [sys.executable, __file__, "worker", "--out", str(out_dir),
                 "--seed", str(seed), "--device", device, "--edge-tp", edge_tp,
                 "--complexes", ",".join(part)],
                stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((p, part, log))
        time.sleep(5)
        for entry in list(running):
            p, part, log = entry
            if p.poll() is not None:
                log.close()
                running.remove(entry)
                done = sum(1 for i in part if docked(out_dir, i))
                print(f"chunk {part[0]}..{part[-1]}: rc={p.returncode} {done}/{len(part)} "
                      f"({time.time() - t0:.0f}s elapsed)", flush=True)
    # The done-marker must assert the work this run did, not the artifacts it happens to find.
    # On 2026-11-27 Galaxy .112 was launched without the checkpoints: all 20 workers raised
    # CheckpointMissing, every chunk returned rc=1 0/n, and the run still printed
    # "DOCK_DONE 108/214" after 10 s, because 108 poses were sitting in the output directory
    # from the rsync that seeded it. A marker counted off the directory cannot tell a finished
    # run from a run that did nothing, which is the whole question the grade gate asks it.
    if clk_stop is not None:
        clk_stop.set()
        clk_thread.join(timeout=30)
        from tt_bio.diffdock.capture import clock_summary
        clk = dict(clock_summary(clk_samples), device=device, edge_tp=edge_tp,
                   chip=os.environ.get("TT_VISIBLE_DEVICES"), seconds=round(time.time() - t0, 1),
                   n_requested=len(ids))
        json.dump(clk, open(out_dir / "clock.json", "w"), indent=1)
        print("CLOCK " + json.dumps(clk), flush=True)
    n = sum(1 for i in ids if docked(out_dir, i))
    refused = set()
    for lg in sorted(logs.glob("dock-*.log")):
        for ln in lg.read_text(errors="replace").splitlines():
            if ln.startswith("FAILED ") or ln.startswith("skip "):
                refused.add(ln.split(None, 2)[1].rstrip(":"))
    unexplained = [i for i in ids if not docked(out_dir, i) and i not in refused]
    el = time.time() - t0
    if unexplained:
        print(f"DOCK_INCOMPLETE {n}/{len(ids)} docked, {len(refused)} refused with a reason, "
              f"{len(unexplained)} unaccounted after {el:.0f}s. First unaccounted: "
              f"{unexplained[:8]}. This run is NOT a finished set and must not gate a grade.",
              flush=True)
    else:
        print(f"DOCK_DONE {n}/{len(ids)} complexes have rank1.sdf and {len(refused)} are "
              f"refused with a recorded reason, after {el:.0f}s", flush=True)


# ---------------------------------------------------------------- grading


def grade(ids, out_dir: Path, acc_dir: Path, label: str):
    """dfd-gpu-fair's grader, pose/reference paths parameterised. See the module docstring."""
    from rdkit import Chem
    from spyrmsd import rmsd as sprmsd
    from spyrmsd.molecule import Molecule

    OUTD = str(out_dir)
    recs = []
    for i in ids:
        d = f"{OUTD}/{i}"
        if not docked(out_dir, i):
            recs.append({"id": i, "docked": False})
            continue
        # Upstream's own evaluator takes the MINIMUM RMSD over EVERY conformer in
        # `_ligands.sdf` for dataset == 'posebusters' (datasets/pdbbind.py:392-402).
        refs = [m for m in Chem.SDMolSupplier(f"{PB}/{i}/{i}_ligands.sdf", sanitize=True,
                                              removeHs=True) if m is not None]
        if not refs:
            one = next(Chem.SDMolSupplier(f"{PB}/{i}/{i}_ligand.sdf", sanitize=True,
                                          removeHs=True), None)
            refs = [one] if one is not None else []
        if not refs:
            recs.append({"id": i, "docked": True, "error": "ref unreadable"})
            continue
        mrefs = []
        for r0 in refs:
            mr = Molecule.from_rdkit(Chem.RemoveHs(r0))
            mr.strip()
            mrefs.append(mr)
        rms, single, nonfinite = [], [], False
        for rank in range(1, 6):
            cand = [f"{d}/rank{rank}.sdf"] + [f"{d}/{f}" for f in sorted(os.listdir(d))
                                              if f.startswith(f"rank{rank}_confidence")]
            path = next((c for c in cand if os.path.exists(c)), None)
            if path is None:
                break
            m = next(Chem.SDMolSupplier(path, sanitize=True, removeHs=True), None)
            if m is None:
                rms.append(None)
                continue
            m = Chem.RemoveHs(m)
            mm = Molecule.from_rdkit(m)
            mm.strip()
            if not all(all(map(math.isfinite, xyz)) for xyz in mm.coordinates):
                nonfinite = True
            vs = []
            for mref in mrefs:
                try:
                    vs.append(float(sprmsd.symmrmsd(mref.coordinates, mm.coordinates,
                                                    mref.atomicnums, mm.atomicnums,
                                                    mref.adjacency_matrix, mm.adjacency_matrix,
                                                    minimize=False)))
                except Exception as e:
                    print(f"{i} rank{rank}: rmsd failed {e}", flush=True)
            rms.append(min(vs) if vs else None)
            single.append(vs[0] if vs else None)
        rec = {"id": i, "docked": True, "rmsd": rms, "rmsd_single": single,
               "n_refs": len(mrefs)}
        if nonfinite:
            rec["nonfinite_coords"] = True
        recs.append(rec)

    try:
        from posebusters import PoseBusters
        import pandas as pd
        pbt = PoseBusters(config="redock")
        scored = [r for r in recs if r.get("docked") and r.get("rmsd")]
        rows = [{"mol_pred": f"{OUTD}/{r['id']}/rank1.sdf",
                 "mol_true": f"{PB}/{r['id']}/{r['id']}_ligand.sdf",
                 "mol_cond": f"{PB}/{r['id']}/{r['id']}_protein.pdb"} for r in scored]
        df = pbt.bust_table(pd.DataFrame(rows))
        cols = [c for c in df.columns if df[c].dtype == bool]
        phys = [c for c in cols if "rmsd" not in c.lower()]
        valid = df[cols].all(axis=1).tolist()
        pvalid = df[phys].all(axis=1).tolist()
        for r, v, pv in zip(scored, valid, pvalid):
            r["pb_valid"] = bool(v)
            r["pb_physics_valid"] = bool(pv)
        df.to_csv(acc_dir / "posebusters_table.csv")
    except Exception as e:
        print(f"PoseBusters validity failed: {e}", flush=True)

    acc_dir.mkdir(parents=True, exist_ok=True)
    json.dump(recs, open(acc_dir / "per_complex.json", "w"), indent=1)

    graded = [r for r in recs if r.get("rmsd") and r["rmsd"] and r["rmsd"][0] is not None]
    n = len(graded)
    top1 = [r["rmsd"][0] for r in graded]
    k1 = sum(1 for v in top1 if v < 2.0)
    top5 = [min(v for v in r["rmsd"][:5] if v is not None) for r in graded]
    k5 = sum(1 for v in top5 if v < 2.0)
    pbv = [r["pb_valid"] for r in recs if "pb_valid" in r]
    pbp = [r["pb_physics_valid"] for r in recs if "pb_physics_valid" in r]
    summary = {
        "label": label,
        "n_requested": len(ids), "n_graded": n, "n_failed": len(ids) - n,
        "recipe": "upstream default_inference_args.yaml: 20 steps / 19 actual / 10 samples / "
                  "confidence on",
        "rmsd_metric": "spyrmsd symmetry-corrected, heavy atoms, no minimisation, minimum over "
                       "every conformer in <id>_ligands.sdf (upstream evaluate.py)",
        "top1_success_lt2A": round(k1 / n, 4) if n else None,
        "top1_success_ci95": wilson(k1, n),
        "top5_success_lt2A": round(k5 / n, 4) if n else None,
        "top5_success_ci95": wilson(k5, n),
        "median_rmsd_top1": round(statistics.median(top1), 3) if n else None,
        "median_rmsd_top1_ci95": boot_median_ci(top1),
        "percentiles_top1": percentiles(top1),
        "posebusters_validity_rate": round(sum(pbv) / len(pbv), 4) if pbv else None,
        "posebusters_validity_ci95": wilson(sum(pbv), len(pbv)) if pbv else None,
        "posebusters_physics_validity_rate": round(sum(pbp) / len(pbp), 4) if pbp else None,
        "nonfinite_or_collapse": sum(1 for r in recs if r.get("nonfinite_coords")),
        "per_complex": str(acc_dir / "per_complex.json"),
    }
    json.dump(summary, open(acc_dir / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def percentiles(xs):
    if not xs:
        return {"p25": None, "p50": None, "p75": None}
    s = sorted(xs)
    def q(f):
        return round(s[min(len(s) - 1, int(f * len(s)))], 3)
    return {"p25": q(0.25), "p50": q(0.50), "p75": q(0.75)}


# ---------------------------------------------------------------- comparison


def load_per_complex(path):
    return {r["id"]: r for r in json.load(open(path))}


def compare(tt_path, ref_path, out_path):
    from scipy import stats

    tt, ref = load_per_complex(tt_path), load_per_complex(ref_path)

    def ok(r):
        return bool(r.get("rmsd")) and r["rmsd"] and r["rmsd"][0] is not None

    common = sorted(i for i in tt if i in ref and ok(tt[i]) and ok(ref[i]))
    tt_only = sorted(i for i in tt if ok(tt.get(i, {})) and i in ref and not ok(ref[i]))
    ref_only = sorted(i for i in ref if ok(ref[i]) and (i not in tt or not ok(tt[i])))

    a = [tt[i]["rmsd"][0] for i in common]
    b = [ref[i]["rmsd"][0] for i in common]
    sa = [1.0 if v < 2.0 else 0.0 for v in a]
    sb = [1.0 if v < 2.0 else 0.0 for v in b]
    disc_tt = sum(1 for x, y in zip(sa, sb) if x and not y)
    disc_ref = sum(1 for x, y in zip(sa, sb) if y and not x)

    def top5(r):
        vs = [v for v in r["rmsd"][:5] if v is not None]
        return min(vs) if vs else None

    a5 = [top5(tt[i]) for i in common]
    b5 = [top5(ref[i]) for i in common]

    va = [(i, tt[i]["pb_valid"]) for i in common if "pb_valid" in tt[i] and "pb_valid" in ref[i]]
    vb = [ref[i]["pb_valid"] for i, _ in va]
    vpairs = [(1.0 if v else 0.0, 1.0 if w else 0.0) for (_, v), w in zip(va, vb)]

    res = {
        "n_common_graded": len(common),
        "tt_only_graded": tt_only,
        "upstream_only_graded": ref_only,
        "top1": {
            "tt": round(sum(sa) / len(sa), 4) if sa else None,
            "upstream": round(sum(sb) / len(sb), 4) if sb else None,
            "delta": round((sum(sa) - sum(sb)) / len(sa), 4) if sa else None,
            "ci95_paired_bootstrap": paired_boot_ci(list(zip(sa, sb))),
            "mcnemar_discordant": {"tt_only": disc_tt, "upstream_only": disc_ref},
            "mcnemar_p": round(mcnemar_exact(disc_tt, disc_ref), 4),
        },
        "top5": {
            "tt": round(sum(1 for v in a5 if v < 2.0) / len(a5), 4) if a5 else None,
            "upstream": round(sum(1 for v in b5 if v < 2.0) / len(b5), 4) if b5 else None,
        },
        "rmsd_distribution": {
            "ks_stat": None, "ks_p": None,
            "median_tt": round(statistics.median(a), 3) if a else None,
            "median_upstream": round(statistics.median(b), 3) if b else None,
            "median_delta_ci95_paired": paired_boot_ci(list(zip(a, b))),
            "percentiles_tt": percentiles(a),
            "percentiles_upstream": percentiles(b),
        },
        "validity": {
            "tt": round(sum(x for x, _ in vpairs) / len(vpairs), 4) if vpairs else None,
            "upstream": round(sum(y for _, y in vpairs) / len(vpairs), 4) if vpairs else None,
            "delta_ci95_paired": paired_boot_ci(vpairs),
            "n": len(vpairs),
        },
        "nan_or_collapse_tt": sum(1 for i in tt if tt[i].get("nonfinite_coords")),
    }
    if a:
        ks = stats.ks_2samp(a, b)
        res["rmsd_distribution"]["ks_stat"] = round(float(ks.statistic), 4)
        res["rmsd_distribution"]["ks_p"] = round(float(ks.pvalue), 4)
    for cid in ("7F8T_FAD", "8AEU_M0L"):
        if cid in tt or cid in ref:
            res.setdefault("outliers", {})[cid] = {
                "tt_rmsd": tt.get(cid, {}).get("rmsd", [None])[0] if ok(tt.get(cid, {})) else None,
                "upstream_rmsd": ref.get(cid, {}).get("rmsd", [None])[0]
                if ok(ref.get(cid, {})) else None,
            }
    if out_path:
        json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------- cli


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dock")
    d.add_argument("--ids", default="all")
    d.add_argument("--shard", default="0/1",
                   help="i/n: dock only the complexes whose position in the sorted set is "
                        "congruent to i mod n. One rented box per shard; the shards are "
                        "interleaved, not contiguous, so a slow block does not land on one box.")
    d.add_argument("--out", required=True)
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--chunk", type=int, default=4)
    d.add_argument("--threads", type=int, default=2)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--device", default="cpu",
                   help="DockOptions.device: cpu (host torch, the set grade) or tt (the card). "
                        "On a card the pool is one process on one chip.")
    d.add_argument("--edge-tp", default="gemm", choices=("gemm", "tile"),
                   help="device only: the edge tensor-product form of the score network.")

    w = sub.add_parser("worker")
    w.add_argument("--out", required=True)
    w.add_argument("--complexes", required=True)
    w.add_argument("--seed", type=int, default=42)
    w.add_argument("--device", default="cpu")
    w.add_argument("--edge-tp", default="gemm", choices=("gemm", "tile"))

    g = sub.add_parser("grade")
    g.add_argument("--ids", default="all")
    g.add_argument("--out", required=True)
    g.add_argument("--acc", required=True)
    g.add_argument("--label", default="tt-bio diffdock-l host torch")

    c = sub.add_parser("compare")
    c.add_argument("--tt", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--out", default=None)

    args = ap.parse_args()
    if args.cmd == "worker":
        dock_chunk(args.complexes.split(","), Path(args.out), args.seed,
                   args.device, args.edge_tp)
        return 0
    if args.cmd == "compare":
        compare(args.tt, args.ref, args.out)
        return 0
    ids = ids_all() if args.ids == "all" else json.load(open(args.ids))
    if args.cmd == "dock":
        i, n = (int(x) for x in args.shard.split("/"))
        if n > 1:
            ids = [c for k, c in enumerate(ids) if k % n == i]
            print(f"shard {i}/{n}: {len(ids)} complexes", flush=True)
        dock(ids, Path(args.out), args.workers, args.chunk, args.seed, args.threads,
             args.device, args.edge_tp)
    else:
        grade(ids, Path(args.out), Path(args.acc), args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
