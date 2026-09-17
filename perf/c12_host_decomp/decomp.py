"""Decompose the Boltz-2 fold into named line items at two pinned clocks, one process per size.

Protocol is `c10-bare-baseline`'s, unchanged: the committed `cdk2x2_<size>` fixture and its
35-row a3m, 200 sampling steps, 3 recycles, 1 diffusion sample, seed 0, no templates, the
production ttnn wheel, warm (the cold fold is discarded). The timed span is the c10 one --
`synchronize_device`, `monotonic_ns`, the complete `state.predict_one` including CIF writing,
`synchronize_device`, `monotonic_ns` -- so the wall here is directly comparable to the 14.881 s
of record and to the F fitted from it.

What this adds: every label runs at a clock requested through ARC FORCE_AICLK and sampled at
about 1 kHz DURING the fold, and the labels come in ARMS, so the same fold is measured with four
different instruments attached and the instrument's own cost is a measured column rather than an
assumption:

  bare      no instrument. The wall of record and the input to the whole-fold two-clock fit.
  regions   `perf/b2x_host_residual/host_residual.py`'s region tree, imported not copied, so
            this row and the `b2z2-host-residual-round2` census share one instrument. Gives
            inclusive/exclusive wall per named item plus a MEASURED leftover row.
  cprofile  function-level attribution, deliberately the heaviest instrument, as the
            cross-check on whether the region tree named the right callee.
  sample    the same region tree plus a 1 ms main-thread stack sampler and gc timing, for
            within-item shares.
  pyspy     an out-of-process sampler, no GIL, no instrumentation bias, as the independent
            confirmation of the top items.

Arms are interleaved inside the clock and the clock order reverses on odd repetitions, so a
drifting box cannot masquerade as a clock effect or as an instrument effect.
"""
from __future__ import annotations
import argparse, cProfile, gzip, importlib.metadata, io, json, math, os, pstats, shutil, signal
import socket, statistics as st, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "scripts/gpu_vs_tt"),
                str(ROOT / "perf/b2x_host_residual"), str(ROOT / "perf/other512")]
from evidence import (coverage, digest, holders, own_nodes, snapshot, validate_snapshot,
                      write_json)
from force_aiclk import FORCE_AICLK, smc

# `c10-fixed-cost`'s fitted clock-immune term, the number this row has to close against.
class Fatal(RuntimeError):
    """An integrity failure: the capture cannot continue and no row from it is trustworthy.

    Everything else that can go wrong in one label -- an instrument breaking, a device
    exception, a clock arm the governor did not hold -- costs that label only. It is recorded
    with valid=False and an error, the arm loop moves on, and `fit.py` reads only valid rows.
    Losing 30 folds because arm 5 of 36 threw is not a stricter protocol, just a lost pass.
    """


F_REFERENCE = {512: 3.9830, 298: 1.9500}
WORK_REFERENCE = {512: 14665.0, 298: 10403.4}          # Mcycles
ARMS = ("bare", "regions", "cprofile", "sample", "pyspy", "cacheclear")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def holder_coverage(rows, interval, pid, node):
    dev = f"/dev/tenstorrent/{node}"
    start, end = interval["start_monotonic_ns"], interval["end_monotonic_ns"]
    during = [r for r in rows if start <= r.get("monotonic_ns", -1) <= end]
    foreign = [h for r in during for h in r.get("holders", [])
               if dev in h["nodes"] and h["pid"] != pid]
    other = sorted({n for r in during for n in r.get("owner_nodes", []) if n != dev})
    points = [start] + [r["monotonic_ns"] for r in during] + [end]
    out = {"observations": len(during), "foreign": foreign[:4], "other_nodes_opened": other,
           "max_gap_ns": max(b - a for a, b in zip(points, points[1:]))}
    out["passed"] = (len(during) >= 3 and not foreign and not other
                     and out["max_gap_ns"] <= 1_000_000_000)
    return out


def lines(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def cif_sha(struct_dir):
    out = {}
    for p in sorted(Path(struct_dir).rglob("*")):
        if p.is_file() and p.suffix in (".cif", ".pdb"):
            out[p.name] = digest(p)["sha256"]
    return out


def profile_table(pr, top=70):
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    rows = []
    for (fn, ln, name), (cc, nc, tt, ct, _callers) in ps.stats.items():
        rows.append({"at": f"{Path(fn).name}:{ln}:{name}", "calls": nc, "tottime_s": tt,
                     "cumtime_s": ct})
    rows.sort(key=lambda r: -r["tottime_s"])
    return {"total_tottime_s": sum(r["tottime_s"] for r in rows), "rows": rows[:top]}


def pyspy_fold(out, seconds, rate=200):
    raw = out / "pyspy.raw"
    cmd = ["sudo", "-n", "py-spy", "record", "--pid", str(os.getpid()), "--duration",
           str(int(seconds)), "--rate", str(rate), "--format", "raw", "--nonblocking",
           "-o", str(raw)]
    try:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT), raw
    except OSError as e:
        return None, repr(e)


def pyspy_reduce(raw, top=40):
    if not Path(raw).exists():
        return {"error": "no py-spy output"}
    leaf, total = Counter(), 0
    for line in Path(raw).read_text().splitlines():
        if not line.strip():
            continue
        stack, _, n = line.rpartition(" ")
        try:
            n = int(n)
        except ValueError:
            continue
        frames = stack.split(";")
        if frames:
            leaf[frames[-1]] += n
        total += n
    return {"samples": total,
            "by_leaf": [{"at": k, "n": v, "share": round(v / total, 5) if total else None}
                        for k, v in leaf.most_common(top)]}


def install_extra(reg, state, T, boltz2):
    """Spans the parent instrument does not carry, added because the ports it was built around
    have since landed default-on.

    `b2x-host-residual`'s tree was taken with `TT_BIO_DEVICE_CONDITIONING`/`_ZINIT`/`_CONFIDENCE`
    off, so its host rows (`diffusion_cond` 1.364 s, `predict_step` exclusive 0.351 s,
    `confidence` exclusive 0.571 s on qb2 at 512 aa) are work that now runs on the card. All
    three gates default to True on this tree, so the host path is thinner and the clock-immune
    seconds have to be somewhere else. These spans put the candidates on the table: the trunk's
    static build and upload, the device pair-assembly stages, and `Boltz2.forward`'s own body,
    which is where the unconditional program-cache clear and the `reset_static_cache` walk over
    every module live.
    """
    ok = {}
    ok["forward"] = reg.patch(boltz2.Boltz2, "forward", "forward")
    for cls, attr, label in (("TrunkModule", "_build_static", "trunk_build_static"),
                             ("TrunkModule", "pop_device_z", "trunk_pop_device_z"),
                             ("PairAssemblyDevice", "__call__", "pair_assembly_device"),
                             ("PairAssemblyDevice", "forward", "pair_assembly_device"),
                             ("DiffusionConditioningDevice", "__call__", "cond_device")):
        obj = getattr(T, cls, None)
        if obj is None or label in ok or not hasattr(obj, attr):
            continue
        ok[label] = reg.patch(obj, attr, label)
    return ok


def cacheclear(dev, n=2):
    """What every `Boltz2.forward` pays to clear and re-enable the device program cache.

    `tt_bio/boltz2.py:5731` runs this on EVERY forward unless `TT_BIO_BOLTZ2_KEEP_PROGRAM_CACHE`
    is set, so a warm fold rebuilds every program it uses. Timed here rather than inferred from a
    stack share, at whatever clock is currently pinned, because the question the row asks about it
    is precisely whether its seconds move with the clock.
    """
    rows = []
    for i in range(n):
        t0 = time.perf_counter()
        dev.disable_and_clear_program_cache()
        t1 = time.perf_counter()
        dev.enable_program_cache()
        t2 = time.perf_counter()
        rows.append({"i": i, "clear_s": t1 - t0, "enable_s": t2 - t1})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, choices=[298, 512], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--node", type=int, required=True)
    ap.add_argument("--clocks", default="1350,800")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--arms", default="bare,regions,cprofile,sample,pyspy")
    ap.add_argument("--settle-s", type=float, default=0.6)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    clocks = [int(x) for x in a.clocks.split(",") if x]
    arms = [x for x in a.arms.split(",") if x]
    for x in arms:
        if x not in ARMS:
            raise SystemExit(f"unknown arm {x}")
    criterion = json.loads((HERE / "criterion.json").read_text())
    dev_path = f"/dev/tenstorrent/{a.node}"

    result = dict(pid=os.getpid(), host=socket.gethostname(), size=a.size, node=a.node,
                  clocks=clocks, arms=arms, reps=a.reps, rows=[], errors=[],
                  F_reference_s=F_REFERENCE[a.size], work_reference_Mcycles=WORK_REFERENCE[a.size],
                  criterion=digest(HERE / "criterion.json"),
                  prediction=digest(HERE / "prediction.json"),
                  source_commit=git("rev-parse", "HEAD").strip(),
                  source_base=criterion["source_base"], dirty_status=git("status", "--short"),
                  production_diff=git("diff", criterion["source_base"], "--", "tt_bio",
                                      "scripts/gpu_vs_tt/tt_baseline.py", "perf/other512"),
                  timer=("synchronize_device; monotonic_ns; state.predict_one incl. CIF write; "
                         "synchronize_device; monotonic_ns"),
                  started_utc_ns=time.time_ns(), completed=False)
    save = lambda: write_json(out / "result.json", result)
    sampler = fd = T = monitor_log = None

    def interrupted(sig, frame):
        raise RuntimeError(f"signal {sig}; release clock and device")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        result["before"] = snapshot()
        validate_snapshot(result["before"], a.node)
        save()
        if socket.gethostname() != "tt-quietbox2":
            raise RuntimeError("wrong host")
        if os.getloadavg()[0] > 2.0:
            raise RuntimeError("load prerequisite failed after the benchlock wait")
        if result["production_diff"]:
            raise RuntimeError("production source differs from the requested base")
        result["dirty_diff"] = git("diff", "HEAD")
        (out / "dirty.patch").write_text(result["dirty_diff"])
        want = {"TT_VISIBLE_DEVICES": str(a.node), "TT_BIO_LEASE_CARDS": str(a.node),
                "TT_BIO_LEASE_HOLDER": "worker:c12-host-decomp"}
        for k, v in want.items():
            if os.environ.get(k) != v:
                raise RuntimeError(f"wrong {k}: {os.environ.get(k)!r} != {v!r}")
        forbidden = {k: v for k, v in os.environ.items()
                     if (k.startswith("TT_BIO_") and k not in want)
                     or k.startswith("TT_METAL_PROFILER") or k.startswith("TRACY")}
        if forbidden:
            raise RuntimeError(f"non-default environment: {forbidden}")
        result["environment"] = {k: v for k, v in os.environ.items()
                                 if k.startswith(("TT_", "OMP_", "MKL_", "PYTHONPATH",
                                                  "LD_LIBRARY_PATH", "BENCHLOCK_"))}

        import torch, ttnn
        import tt_bio.tenstorrent as T
        import tt_bio.boltz2 as boltz2
        import tt_baseline as B
        import host_residual as HR
        from fold_ab_multi import patch_boltz2_cfg
        result["instrument"] = {"host_residual": digest(
            ROOT / "perf/b2x_host_residual/host_residual.py")}
        result["ttnn_version"] = importlib.metadata.version("ttnn")
        result["ttnn_path"] = ttnn.__file__
        result["ttnn_config"] = str(ttnn.CONFIG)
        if Path(T.__file__).resolve() != ROOT / "tt_bio/tenstorrent.py":
            raise RuntimeError("wrong production source loaded")
        if "/site-packages/ttnn/" not in ttnn.__file__:
            raise RuntimeError("expected the production wheel")
        for attr in ("enable_logging", "enable_graph_report", "enable_detailed_buffer_report",
                     "enable_detailed_tensor_report"):
            if getattr(ttnn.CONFIG, attr):
                raise RuntimeError(f"profiling enabled: {attr}")
        root = Path(f"/sys/class/tenstorrent/tenstorrent!{a.node}")
        result["assigned_node_sysfs"] = {"resolved": str(root.resolve()),
                                         "device": str((root / "device").resolve())}
        for name in ("tt_asic_id", "tt_card_type", "tt_fw_bundle_ver", "tt_serial", "tt_aiclk"):
            if (root / name).exists():
                try:
                    result["assigned_node_sysfs"][name] = (root / name).read_text().strip()
                except OSError as e:
                    raise RuntimeError(f"assigned node {a.node} sysfs unreadable ({name}): {e!r}")
        B.RECYCLING_STEPS = 3
        B.SAMPLING_STEPS = 200
        B.DIFFUSION_SAMPLES = 1
        B.SEED = 0
        patch_boltz2_cfg()
        B._card_info = lambda: {"assigned_node_sysfs": result["assigned_node_sysfs"]}
        fixture = ROOT / f"perf/size512/fixtures/cdk2x2_{a.size}"
        target, msa = fixture.with_suffix(".yaml"), fixture.with_suffix(".a3m")
        if "templates:" in target.read_text():
            raise RuntimeError("templates in input")
        result["inputs"] = [digest(target), digest(msa)]
        for p in (target, msa):
            committed = subprocess.check_output(
                ["git", "show", criterion["source_base"] + ":" + str(p.relative_to(ROOT))],
                cwd=ROOT)
            if committed != p.read_bytes():
                raise RuntimeError("fixture differs from the committed input")
        _fold, meta, state = B.build_fold("boltz2", out / "msa", target, msa, instrument=False,
                                          hoist=False, fast=False, trace=False, recycling_steps=3)
        dev = T.get_device()
        result["opened"] = snapshot()
        validate_snapshot(result["opened"], a.node, True)
        if own_nodes() != [dev_path]:
            raise RuntimeError(f"wrong actual opened device: {own_nodes()}")
        result["model_meta"] = {k: meta[k] for k in
                                ("hardware", "grid", "load_s", "n_msa", "card_type",
                                 "recycling_steps", "timed_region") if k in meta}
        result["model_predict_args"] = dict(state.model.predict_args)
        expected = {"recycling_steps": 3, "sampling_steps": 200, "diffusion_samples": 1,
                    "max_parallel_samples": None}
        if dict(state.model.predict_args) != expected:
            raise RuntimeError("actual model config mismatch")
        if meta["n_msa"] != 35:
            raise RuntimeError("MSA rows not 35")
        result["sdpa_cap"] = T._triatt_sdpa._Q_SPLIT_MAX_S
        result["acquisition_sources"] = [digest(p) for p in sorted(HERE.glob("*.py"))]
        struct_dir = Path(meta["struct_dir"])
        job_cfg = meta["job_cfg"]

        fd = os.open(dev_path, os.O_RDWR | os.O_APPEND)
        monitor_log = (out / "sampler.log").open("w")
        sampler = subprocess.Popen(
            [sys.executable, str(HERE / "evidence.py"), str(out / "clock.jsonl"),
             str(os.getpid()), str(a.node)],
            stdin=subprocess.PIPE, stdout=monitor_log, stderr=subprocess.STDOUT)
        time.sleep(0.3)
        save()

        def one(label, arm, clock, rep):
            """One timed fold under one instrument at one requested clock."""
            before = snapshot()
            validate_snapshot(before, a.node, True)
            if dict(state.model.predict_args) != expected:
                raise Fatal("model config changed between labels")
            if job_cfg["seed"] != 0:
                raise Fatal("seed changed between labels")
            if before["boot_id"] != result["before"]["boot_id"]:
                raise Fatal("boot changed")
            force = list(smc(fd, FORCE_AICLK, clock))
            if force[0] != 0:
                raise RuntimeError(f"FORCE_AICLK({clock}) failed: {force}")
            time.sleep(a.settle_s)
            for p in struct_dir.glob("*"):
                p.unlink()
            T.SDPA_FUSED_LARGE_S_STATS[:] = [0, 0]
            row = dict(label=label, arm=arm, clock_MHz=clock, rep=rep, force_response=force,
                       before=before, valid=False)
            reg = pr = smp = gt = spy = raw = None
            if arm in ("regions", "sample"):
                reg = HR.Regions()
                row["patched"] = HR.install_regions(reg, state, T, boltz2)
                row["patched_extra"] = install_extra(reg, state, T, boltz2)
            if arm == "sample":
                gt = HR.GCTimer()
                gt.on()
                smp = HR.StackSampler(period=0.001)
                smp.start()
            if arm == "cprofile":
                pr = cProfile.Profile()
            if arm == "cacheclear":
                force = list(smc(fd, FORCE_AICLK, clock))
                time.sleep(a.settle_s)
                start = time.monotonic_ns()
                rows = cacheclear(dev)
                end = time.monotonic_ns()
                row.update(start_monotonic_ns=start, end_monotonic_ns=end,
                           elapsed_s=(end - start) / 1e9, cacheclear=rows,
                           plddt=None, cif={}, above_cap_sdpa_counts=[0, 0])
                row["clock"] = coverage(lines(out / "clock.jsonl"), row, clock)
                row["holders"] = holder_coverage(lines(out / "holders.jsonl"), row,
                                                 os.getpid(), a.node)
                if not row["clock"]["pass"]:
                    raise RuntimeError(f"clock artifact on cacheclear: {row['clock']}")
                row["valid"] = True
                result["rows"].append(row)
                save()
                print(json.dumps({"label": label, "arm": arm, "clock": clock,
                                  "elapsed_s": round(row["elapsed_s"], 4),
                                  "MHz": row["clock"]["max_MHz"]}), flush=True)
                return row
            if arm == "pyspy":
                est = 16.0 if a.size == 512 else 11.0
                spy, raw = pyspy_fold(out, est * (1350.0 / clock) + 4.0)
                time.sleep(1.0)
            ttnn.synchronize_device(dev)
            start = time.monotonic_ns()
            if pr is not None:
                pr.enable()
            metrics, best, feats = state.predict_one(target, job_cfg)
            if pr is not None:
                pr.disable()
            ttnn.synchronize_device(dev)
            end = time.monotonic_ns()
            row.update(start_monotonic_ns=start, end_monotonic_ns=end,
                       elapsed_s=(end - start) / 1e9, plddt=metrics.get("plddt"),
                       above_cap_sdpa_counts=list(T.SDPA_FUSED_LARGE_S_STATS))
            if smp is not None:
                smp.stop.set()
                smp.join(timeout=5)
                gt.off()
            if reg is not None:
                reg.remove()
                tbl = reg.table()
                top = sum(v["incl_s"] for k, v in tbl.items() if "/" not in k)
                row["tree"] = tbl
                row["top_level_incl_s"] = top
                row["unattributed_s"] = row["elapsed_s"] - top
                row["region_notes"] = reg.notes
            if smp is not None:
                row["stack_sampler"] = smp.report()
                row["gc"] = gt.report()
            if pr is not None:
                row["cprofile"] = profile_table(pr)
            if spy is not None:
                try:
                    spy.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    spy.terminate()
                row["pyspy_stdout"] = (spy.stdout.read() or b"").decode()[-2000:]
                keep = out / f"pyspy_{label}.raw"
                if Path(raw).exists():
                    shutil.move(str(raw), keep)
                row["pyspy"] = pyspy_reduce(keep)
            keep = out / "cifs" / label
            keep.mkdir(parents=True)
            for p in struct_dir.glob("*"):
                if p.is_file():
                    shutil.copyfile(p, keep / p.name)
            row["cif"] = cif_sha(keep)
            if len(row["cif"]) != 1:
                raise RuntimeError(f"expected one CIF, got {list(row['cif'])}")
            if row["plddt"] is None or not math.isfinite(float(row["plddt"])):
                raise RuntimeError("nonfinite confidence")
            row["after"] = snapshot()
            validate_snapshot(row["after"], a.node, True)
            if row["after"]["boot_id"] != result["before"]["boot_id"]:
                raise Fatal("boot changed")
            if row["above_cap_sdpa_counts"] != [0, 0]:
                raise RuntimeError("above-cap SDPA route unexpectedly reached")
            time.sleep(0.15)
            row["clock"] = coverage(lines(out / "clock.jsonl"), row, clock)
            row["holders"] = holder_coverage(lines(out / "holders.jsonl"), row, os.getpid(), a.node)
            if not row["clock"]["pass"]:
                raise RuntimeError(f"clock artifact or thin coverage: {row['clock']}")
            if not row["holders"]["passed"]:
                raise RuntimeError(f"co-tenancy or thin holder coverage: {row['holders']}")
            row["valid"] = True
            result["rows"].append(row)
            save()
            print(json.dumps({"label": label, "arm": arm, "clock": clock,
                              "elapsed_s": round(row["elapsed_s"], 4),
                              "unattributed_s": row.get("unattributed_s"),
                              "MHz": row["clock"]["max_MHz"], "W": row["clock"]["max_W"],
                              "cif": list(row["cif"].values())[0][:16],
                              "plddt": row["plddt"]}), flush=True)
            del metrics, best, feats
            return row

        # one discarded cold fold per process, at the highest clock, no instrument
        one("cold", "bare", max(clocks), -1)
        result["rows"][-1]["valid"] = False
        result["rows"][-1]["note"] = "cold fold, discarded from every number"
        for rep in range(a.reps):
            order = clocks if rep % 2 == 0 else list(reversed(clocks))
            for clock in order:
                for arm in arms:
                    label = f"{arm}_c{clock}_r{rep}"
                    try:
                        one(label, arm, clock, rep)
                    except Fatal:
                        raise
                    except Exception as e:
                        result["errors"].append(f"{label}: {e!r}")
                        result["rows"].append(dict(label=label, arm=arm, clock_MHz=clock,
                                                   rep=rep, valid=False, error=repr(e)))
                        save()
                        traceback.print_exc()
                        print(json.dumps({"label": label, "arm": arm, "clock": clock,
                                          "SKIPPED": type(e).__name__}), flush=True)
        result["completed"] = True
        result["labels_skipped"] = [r["label"] for r in result["rows"] if r.get("error")]
    except BaseException as e:
        result["errors"].append(repr(e))
        traceback.print_exc()
    finally:
        if fd is not None:
            try:
                result["release_response"] = list(smc(fd, FORCE_AICLK, 0))
            except BaseException as e:
                result["errors"].append("release: " + repr(e))
                result["completed"] = False
            finally:
                os.close(fd)
        if sampler is not None:
            try:
                sampler.communicate(b"stop\n", timeout=15)
            except subprocess.TimeoutExpired:
                sampler.terminate()
                sampler.wait(timeout=5)
            result["sampler_returncode"] = sampler.returncode
            if sampler.returncode:
                result["completed"] = False
        if monitor_log:
            monitor_log.close()
        if T is not None:
            try:
                T.cleanup()
            except BaseException as e:
                result["errors"].append("cleanup: " + repr(e))
                result["completed"] = False
        try:
            result["after"] = snapshot()
            validate_snapshot(result["after"], a.node)
            if result["after"]["own_nodes"]:
                raise RuntimeError("device still open after cleanup")
            if result.get("release_response", [None])[0] != 0:
                raise RuntimeError("clock release not confirmed")
            if result["after"]["boot_id"] != result["before"]["boot_id"]:
                raise Fatal("boot changed")
        except BaseException as e:
            result["errors"].append("snapshot: " + repr(e))
            result["completed"] = False
        for name in ("clock.jsonl", "holders.jsonl"):
            p = out / name
            if p.exists():
                (out / (name + ".gz")).write_bytes(gzip.compress(p.read_bytes(), mtime=0))
                p.unlink()
        valid = [r for r in result["rows"] if r.get("valid")]
        for arm in arms:
            for c in clocks:
                ws = [r["elapsed_s"] for r in valid if r["arm"] == arm and r["clock_MHz"] == c]
                if ws:
                    result.setdefault("medians", {})[f"{arm}_{c}"] = round(st.median(ws), 4)
        result["finished_utc_ns"] = time.time_ns()
        save()
    return 0 if result["completed"] else 2


if __name__ == "__main__":
    sys.exit(main())
