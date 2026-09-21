#!/usr/bin/env python3
"""Do the two triangle-attention levers fire on a DESIGN model? Counted in the spawn child.

`triatt_lever_ab.py` drives `tt_baseline.build_fold`, which only reaches PREDICT models -- the CLI
accepts boltz2, esmfold2, esmfold2-fast, protenix-v1, protenix-v2, openfold3, openbind, opendde,
opendde-abag, rf3 and **not** boltzgen or rfd3. So the two design models this row must cover were
unreachable by the instrument that covered the other six. This reaches them the same way
`mm_key_census.py` does: `tt_bio.main design` in a child process, with the counters read there.

A design spawns its workers, so counters read in the launcher are always zero
(`in-process-patch-never-reaches-a-spawn-child`). The hook runs in EVERY process via a generated
`sitecustomize.py`, dumps every 3 s so a run cut short still leaves its counts, and the parent sums.

The arms are set by ENV, not in-process: both flags are read at module import
(`triatt_sdpa.py:302` and `:467`), so one process per arm is the only honest A/B here. `off` is the
shipped default, so an `off` arm is main by construction.

  gate_served / gate_rejected   triatt_sdpa.GATE_STATS, plus reject reasons
  fuse_calls / fuse_rejected    a wrapper on sdpa_fused_qkv, plus FUSE_REJECTS by reason
  fuse_served = calls - rejects raised INSIDE that function (the three listed below are raised in
                tenstorrent.py before it is entered, so subtracting the total reads -1208)
"""
import argparse, json, os, subprocess, sys
from collections import Counter
from pathlib import Path

STATE = Counter()
WRAPPED = [False]

#: raised in tenstorrent.py BEFORE `sdpa_fused_qkv` is entered; everything else is raised inside it
FUSE_REJECT_OUTSIDE = ("qkv_already_fused_with_gate", "site", "no_full_S_chunk")


def _install():
    if WRAPPED[0]:
        return
    TS = sys.modules.get("tt_bio.triatt_sdpa")
    if TS is None or getattr(TS, "_allm_lever_wrapped", False):
        return
    orig = getattr(TS, "sdpa_fused_qkv", None)
    if orig is None:
        return

    def counting(*a, **k):
        STATE["fuse_calls"] += 1
        out = orig(*a, **k)
        # SERVED is the RETURN VALUE, not calls-minus-rejects. `sdpa_fused_qkv` opens with
        #   if not (_FUSE_QKV or force) or bias is None: return None
        # which returns WITHOUT calling `_fuse_reject`, so that exit is invisible to the reject
        # counters and the subtraction counted it as a serve: an `off` arm read 656 served with
        # the flag at 0. Observing the return makes the count true on both arms.
        STATE["fuse_served" if out is not None else "fuse_returned_none"] += 1
        return out

    TS.sdpa_fused_qkv = counting
    TS._allm_lever_wrapped = True
    WRAPPED[0] = True


def _snapshot():
    TS = sys.modules.get("tt_bio.triatt_sdpa")
    if TS is None:
        return {}
    out = dict(STATE)
    gs = getattr(TS, "GATE_STATS", None)
    if gs:
        out["gate_served"], out["gate_rejected"] = int(gs[0]), int(gs[1])
    for key, n in (getattr(TS, "GATE_REJECTS", {}) or {}).items():
        reason = key[0] if isinstance(key, tuple) else key
        k = f"gate_reject|{reason}"
        out[k] = out.get(k, 0) + n
    for reason, n in (getattr(TS, "FUSE_REJECTS", {}) or {}).items():
        out[f"fuse_reject|{reason}"] = out.get(f"fuse_reject|{reason}", 0) + n
    # fuse_served comes from the wrapper's observed return value (see `counting`), so nothing is
    # derived here. The reject tallies stay, because they say WHY a call did not serve.
    out.setdefault("fuse_served", 0)
    out.setdefault("fuse_returned_none", 0)
    out["gate_epilogue_flag"] = int(bool(getattr(TS, "_GATE_EPILOGUE", False)))
    out["fuse_qkv_flag"] = int(bool(getattr(TS, "_FUSE_QKV", False)))
    return out


def install_child_hook():
    outdir = os.environ.get("ALLM_LEVER_CENSUS_DIR")
    if not outdir:
        return
    import atexit, threading, time
    path = os.path.join(outdir, f"pid{os.getpid()}.json")

    def dump():
        snap = _snapshot()
        if not snap:
            return
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            json.dump({"pid": os.getpid(), "argv": sys.argv[:4], "state": snap}, fh)
        os.replace(tmp, path)

    def tick():
        while True:
            time.sleep(3)
            try:
                _install(); dump()
            except Exception:                                            # noqa: BLE001
                pass

    threading.Thread(target=tick, daemon=True).start()
    atexit.register(lambda: (_install(), dump()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--pythonpath")
    ap.add_argument("--label", required=True)
    ap.add_argument("--arm", required=True, choices=("off", "on"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=0, help="seconds; 0 = run to completion")
    ap.add_argument("cli", nargs="*")
    a = ap.parse_args()

    work = Path(a.out).resolve().parent / f".levercensus-{a.label}"
    hook, dumps = work / "hook", work / "dumps"
    hook.mkdir(parents=True, exist_ok=True)
    dumps.mkdir(parents=True, exist_ok=True)
    for stale in dumps.glob("pid*.json"):
        stale.unlink()
    (hook / "sitecustomize.py").write_text(
        "import sys\n"
        f"sys.path.append({str(Path(__file__).resolve().parent)!r})\n"
        "try:\n"
        "    from lever_design_census import install_child_hook\n"
        "    install_child_hook()\n"
        "except Exception:\n"
        "    pass\n")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(hook)] + ([a.pythonpath] if a.pythonpath else []))
    env["ALLM_LEVER_CENSUS_DIR"] = str(dumps)
    # The arms, set where the flags are actually read: at import, in the child.
    env["TT_BIO_TRIATT_GATE_EPILOGUE"] = "1" if a.arm == "on" else "0"
    env["TT_BIO_TRIATT_FUSE_QKV"] = "1" if a.arm == "on" else "0"

    import time as _t
    t0 = _t.time()
    p = subprocess.Popen([a.python, *a.cli], env=env)
    try:
        rc = p.wait(timeout=a.timeout or None)
        cut = False
    except subprocess.TimeoutExpired:
        # Counts are already on disk; a design is long and the firing question does not need it
        # to finish. SIGINT first so the engine can unwind, then SIGKILL the survivor.
        p.send_signal(2)
        try:
            rc = p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill(); rc = p.wait()
        cut = True
    wall = round(_t.time() - t0, 3)

    agg, procs = Counter(), 0
    for f in sorted(dumps.glob("pid*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:                                                # noqa: BLE001
            continue
        procs += 1
        for k, v in d["state"].items():
            if k.endswith("_flag"):
                agg[k] = max(agg[k], v)
            else:
                agg[k] += v
    snap = {"label": a.label, "arm": a.arm, "cli": a.cli, "rc": rc, "cut_short": cut,
            "wall_s": wall, "processes": procs, "state": dict(sorted(agg.items()))}
    json.dump(snap, open(a.out, "w"), indent=2)
    print(f"--- {a.label} arm={a.arm}: rc={rc} cut_short={cut} {procs} processes, {wall}s")
    for k, n in sorted(agg.items()):
        print(f"  {k:36s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
