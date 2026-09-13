#!/usr/bin/env python3
"""One paired, interleaved Boltz-2 512 aa fold A/B of a single shipped lever, arch-agnostic.

This is the transfer-function instrument. The same file, the same fixture and the same protocol
run on Wormhole and on Blackhole, so a WH ratio and a BH ratio for one lever are comparable by
construction rather than by reconciliation of two campaigns' bookkeeping.

Arms are set IN PROCESS. Every lever on this page is read either into a module constant at import
(`_QKVG_ENABLED`, `GATE_GRANULARITY`, ...) or out of `os.environ` per call, so an arm is a set of
module-attribute writes plus a set of environment writes, applied before each fold. An env pin from
the outside would make both arms the same arm, which is why the driver refuses to start with one set.

Order is ABBA inside a rep (off, on, on, off), so a box that drifts during a rep cancels instead of
landing on whichever arm runs second. The two `off` folds of a rep bracket the two `on` folds, and
their own ratio is this session's A/A floor: a lever ratio inside that floor is not a measurement.

ENGAGEMENT IS THE KNOWN-ANSWER CONTROL. A declined optimization and an engaged one write the same
structure, so an identical digest is equally consistent with the lever never having run. Every lever
here exposes a (served, declined) counter; the run asserts the `on` arm served calls and the `off`
arm served none. A lever whose two arms engage identically is reported void, not fast.

    lever_ab.py --lever trunkfuse --out out/wh_trunkfuse_c0.json --reps 3
"""
from __future__ import annotations

import argparse, hashlib, importlib, importlib.util, json, os, shutil, socket
import statistics as st, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "_b2x_flaglev", REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py")
LEV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LEV)

FIX = REPO / "perf" / "size512" / "fixtures"
ORDER = ["off", "on", "on", "off"]


def _attr(dotted: str, value):
    mod, _, name = dotted.rpartition(".")
    return ("attr", mod, name, value)


def _env(name: str, value):
    return ("env", name, value)


# kind: what the lever is made of, which is the axis the transfer function is tested along.
#   arithmetic  -- same shapes, different amount or order of arithmetic / DRAM reads
#   layout      -- same arithmetic, different tensor movement or op class
#   eligibility -- a guard sized on the part (grid, L1, core count) decides per call
#   placement   -- work moves between host and device
LEVERS: dict[str, dict] = {
    "trunkfuse": dict(
        kind="arithmetic",
        what="triangle qkv+gate, +bias and trimul g_out folded into one weight read",
        off=[_attr("tt_bio.triatt_qkv._QKVG_ENABLED", False),
             _attr("tt_bio.triatt_qkv._QKVGB_ENABLED", False),
             _attr("tt_bio.tenstorrent._TRIMUL_FUSED_GOUT", False)],
        on=[_attr("tt_bio.triatt_qkv._QKVG_ENABLED", True),
            _attr("tt_bio.triatt_qkv._QKVGB_ENABLED", True),
            _attr("tt_bio.tenstorrent._TRIMUL_FUSED_GOUT", True)],
        stats=["tt_bio.triatt_qkv.QKVG_STATS", "tt_bio.triatt_qkv.QKVGB_STATS",
               "tt_bio.tenstorrent.TRIMUL_GOUT_STATS"]),
    "qchunk": dict(
        kind="eligibility",
        what="SDPA query chunk sized to fill one pass of the compute grid",
        off=[_attr("tt_bio.tenstorrent._SDPA_GRID_Q_CHUNK", False)],
        on=[_attr("tt_bio.tenstorrent._SDPA_GRID_Q_CHUNK", True)],
        stats=[]),
    "shiftgather": dict(
        kind="layout",
        what="atom key window as a shift+slice instead of a one-hot matmul",
        off=[_attr("tt_bio.tenstorrent._ATOM_SHIFT_GATHER_OFF", True)],
        on=[_attr("tt_bio.tenstorrent._ATOM_SHIFT_GATHER_OFF", False)],
        stats=["tt_bio.tenstorrent.ATOM_SHIFT_GATHER_STATS"]),
    "devcond": dict(
        kind="placement",
        what="diffusion conditioning on the card instead of the host",
        off=[_env("TT_BIO_DEVICE_CONDITIONING", "0")],
        on=[_env("TT_BIO_DEVICE_CONDITIONING", "1")],
        stats=[]),
    "pwabatch": dict(
        kind="layout",
        what="MSA row weights projected through the whole matrix once, heads sliced out",
        off=[_attr("tt_bio.tenstorrent._PWA_BATCH_HEAD_WEIGHTS", False)],
        on=[_attr("tt_bio.tenstorrent._PWA_BATCH_HEAD_WEIGHTS", True)],
        stats=[]),
    "gategran": dict(
        kind="arithmetic",
        what="tiles per DST acquire in the gated reblock-permute kernel, 1 against the shipped 2",
        off=[_attr("tt_bio.reblock_permute.GATE_GRANULARITY", 1)],
        on=[_attr("tt_bio.reblock_permute.GATE_GRANULARITY", 2)],
        stats=[]),
    "sdpaaddgran": dict(
        kind="arithmetic",
        what="tiles per pass in the fused SDPA's three adds, 1 against the auto value",
        off=[_env("TT_BIO_SDPA_ADD_GRANULARITY", "1")],
        on=[_env("TT_BIO_SDPA_ADD_GRANULARITY", None)],
        stats=[]),
    # The known-answer control for the whole rig, and the instrument for USABLE-WIDTH. Both arms
    # are the shipped default, so the true ratio is 1.0 by construction and whatever the run reads
    # instead is this width's noise floor. A lever ratio smaller than the null read at the same
    # width is not a measurement of the lever.
    "null": dict(
        kind="control",
        what="both arms are the shipped default; the answer is 1.0",
        off=[_attr("tt_bio.tenstorrent._SDPA_GRID_Q_CHUNK", True)],
        on=[_attr("tt_bio.tenstorrent._SDPA_GRID_Q_CHUNK", True)],
        stats=[]),
    "fusebias": dict(
        kind="arithmetic",
        what="all diffusion bias stacks built in one pass instead of one call per layer",
        off=[_env("TT_BIO_FUSE_BIAS_STACKS", "0")],
        on=[_env("TT_BIO_FUSE_BIAS_STACKS", "1")],
        stats=[]),
}

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def apply_arm(spec: list) -> None:
    for item in spec:
        if item[0] == "attr":
            _, mod, name, value = item
            m = importlib.import_module(mod)
            assert hasattr(m, name), f"{mod}.{name} does not exist on this tree"
            setattr(m, name, value)
        else:
            _, name, value = item
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _grid(dev) -> str:
    try:
        g = dev.compute_with_storage_grid_size()
        return f"{g.x}x{g.y}"
    except Exception as e:
        return f"unavailable: {e}"


def _arch(dev) -> str:
    try:
        return str(dev.arch())
    except Exception as e:
        return f"unavailable: {e}"


def read_stats(paths: list) -> dict:
    out = {}
    for p in paths:
        mod, _, name = p.rpartition(".")
        out[name] = list(getattr(importlib.import_module(mod), name))
    return out


def zero_stats(paths: list) -> None:
    for p in paths:
        mod, _, name = p.rpartition(".")
        v = getattr(importlib.import_module(mod), name)
        for i in range(len(v)):
            v[i] = 0


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", required=True, choices=sorted(LEVERS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--fixture", default="cdk2x2_512")
    ap.add_argument("--width", type=int, default=1,
                    help="how many sibling folds share the host; sets this process's thread cap")
    ap.add_argument("--host-threads", dest="host_threads", type=int, default=0)
    ap.add_argument("--no-thread-cap", dest="no_thread_cap", action="store_true",
                    help="leave the host threads uncapped, which is what every campaign before this "
                         "one was unknowingly measuring")
    ap.add_argument("--open-lock", dest="open_lock", default="",
                    help="flock this path around the device open. On whglx the library's own "
                         "host-wide /tmp/tt-bio-device-open.lock is owned by another account and "
                         "`open(..., 'w')` raises PermissionError, which tt_bio treats as 'no lock "
                         "file' and proceeds UNSERIALIZED -- the exact UMD bring-up race the lock "
                         "exists to prevent. This restores serialization within one fanout.")
    args = ap.parse_args()
    OUT_PATH = args.out
    lev = LEVERS[args.lever]
    LEV.SAMPLING_STEPS, LEV.RECYCLING_STEPS = args.steps, args.recycles

    for item in lev["off"] + lev["on"]:
        if item[0] == "env":
            assert item[1] not in os.environ, (
                f"{item[1]} is pinned in the environment; the arms are set in process and an "
                f"outside pin would make both arms the same arm")

    # A fanout of independent processes gets no thread cap from anybody: tt-bio caps the per-card
    # workers IT spawns, not siblings launched side by side. Eight uncapped folds took this 32-core
    # box to loadavg 133 and stretched the warm fold from 41.7 s to 96.9 s across chips, which is
    # noise landing in the arms rather than concurrency being measured. Cap to the share tt-bio's
    # own arithmetic hands this width, before torch sizes its pools.
    capped = args.width > 1 and not args.no_thread_cap
    if capped:
        from tt_bio import runtime as _RT
        os.environ.update(_RT.host_thread_cap_env(args.width, args.host_threads or None))
    import torch
    torch.set_grad_enabled(False)
    if capped:
        _RT.bind_host_threads()
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this tree")

    if args.open_lock:
        import fcntl
        _lk = open(args.open_lock, "a+")
        fcntl.flock(_lk, fcntl.LOCK_EX)
        print(f"  open-lock held {args.open_lock}", flush=True)
        try:
            dev = get_device()
        finally:
            fcntl.flock(_lk, fcntl.LOCK_UN)
    else:
        dev = get_device()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "lever": args.lever, "kind": lev["kind"], "what": lev["what"],
        "fixture": args.fixture, "steps": args.steps, "recycles": args.recycles,
        "tt_bio_file": _TB.__file__,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg_at_start": os.getloadavg(),
        "lease_dir": os.environ.get("TT_BIO_LEASE_DIR", "/tmp/tt-bio-device-leases"),
        "open_lock": args.open_lock or "NONE (library lock only)",
        "width": args.width,
        "thread_capped": capped,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        # The grid is the variable the eligibility levers key on, and a firmware move changes it
        # under a retest without anything else looking different. Record it, never assume it.
        "compute_grid": _grid(dev),
        "arch": _arch(dev),
    }
    dump()

    work = Path(tempfile.mkdtemp(prefix=f"k10-{args.lever}-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{args.fixture}.yaml", (FIX / f"{args.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run(f"k10-transfer-{args.lever}", cfg)
    diff_mod = getattr(state.model, "structure_module", None)
    diff_mod = getattr(diff_mod, "score_model", None)

    def fold(arm: str) -> dict:
        apply_arm(lev[arm])
        zero_stats(lev["stats"])
        if diff_mod is not None:
            try:
                diff_mod.reset_static_cache()
            except Exception:
                pass
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(FIX / f"{args.fixture}.yaml", cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        return {"arm": arm, "fold_s": round(wall, 3),
                "stats": read_stats(lev["stats"]),
                "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
                "loadavg1": round(os.getloadavg()[0], 2),
                "t_end": round(time.time(), 1)}

    runs = []
    for arm in ("off", "on"):
        r = fold(arm); r["warmup"] = True; r["rep"] = -1; runs.append(r)
        print(f"  warm {arm:3s} {r['fold_s']:7.3f}s {r['stats']} cif {r['cif_sha256'][:16]}",
              flush=True)
        OUT["runs"] = runs; dump()
    for i in range(args.reps):
        for arm in ORDER:
            r = fold(arm); r["warmup"] = False; r["rep"] = i; runs.append(r)
            print(f"  rep{i} {arm:3s} {r['fold_s']:7.3f}s load={r['loadavg1']} "
                  f"{r['stats']} plddt={r['plddt']} cif {r['cif_sha256'][:16]}", flush=True)
            OUT["runs"] = runs; dump()

    timed = [r for r in runs if not r["warmup"]]
    med = {a: st.median([r["fold_s"] for r in timed if r["arm"] == a]) for a in ("off", "on")}
    OUT["median_fold_s"] = med
    OUT["ratio"] = round(med["off"] / med["on"], 5)

    # ABBA: the two `off` folds of a rep bracket the two `on` folds. Their ratio is the floor.
    floors = []
    for i in range(args.reps):
        offs = [r["fold_s"] for r in timed if r["rep"] == i and r["arm"] == "off"]
        ons = [r["fold_s"] for r in timed if r["rep"] == i and r["arm"] == "on"]
        if len(offs) == 2:
            floors.append(max(offs) / min(offs))
        if len(ons) == 2:
            floors.append(max(ons) / min(ons))
    OUT["aa_floors"] = [round(f, 5) for f in floors]
    OUT["aa_floor_worst"] = round(max(floors), 5) if floors else None
    OUT["paired_ratios"] = [round(o / n, 5) for o, n in
                            zip([r["fold_s"] for r in timed if r["arm"] == "off"],
                                [r["fold_s"] for r in timed if r["arm"] == "on"])]
    OUT["separates_from_floor"] = bool(
        floors and abs(OUT["ratio"] - 1.0) > (max(floors) - 1.0))

    shas = {a: sorted({r["cif_sha256"] for r in timed if r["arm"] == a}) for a in ("off", "on")}
    OUT["cif_sha256"] = shas
    OUT["bit_exact"] = len(shas["off"]) == 1 and shas["off"] == shas["on"]
    OUT["plddt"] = {a: sorted({r["plddt"] for r in timed if r["arm"] == a}) for a in ("off", "on")}

    # The engagement control. Served counters are the even indices of each (served, declined) pair.
    def served(rs):
        tot = 0
        for r in rs:
            for v in r["stats"].values():
                tot += v[0]
        return tot
    on_served, off_served = served([r for r in timed if r["arm"] == "on"]), \
        served([r for r in timed if r["arm"] == "off"])
    OUT["engagement"] = {"on_served": on_served, "off_served": off_served,
                         "instrumented": bool(lev["stats"])}
    OUT["engagement_control"] = (
        "pass" if (on_served > 0 and off_served == 0) else
        "void" if lev["stats"] else "uninstrumented")
    dump()
    keys = ("median_fold_s", "ratio", "aa_floors", "aa_floor_worst", "paired_ratios",
            "separates_from_floor", "bit_exact", "plddt", "engagement", "engagement_control")
    print(json.dumps({k: OUT[k] for k in keys}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
