#!/usr/bin/env python3
"""Which (kt, nt) keys does each model present to `_MM_BLOCK`, and does the table serve them?

`_MM_BLOCK` (tenstorrent.py:7093) is one shared lookup table read by five levers. It grew seven
entries in the pvx window -- (8, 32), (8, 33), (2, 6), (2, 8), (2, 9) for protenix-v2 and
(12, 36), (12, 12) for opendde -- and the last two are documented NOT bit-exact. The leak check
that shipped with them (`perf/odde4x/ab_px_leak.json`) covered protenix-v2 and nothing else, so
for every other model in the repo "does a key the table newly answers reach my weights" is an
open question. This answers it by COUNTING at run time rather than reading the table.

THE READER SET IS ENUMERATED FROM THE CODE, not from a list: `grep -rn _MM_BLOCK tt_bio/` finds
exactly three readers, and this hook wraps all three.

  tenstorrent._mm_block_for   the whole table, feeding `_qkv_mm_config` -> TriangleAttention
  swiglu_fused._block         allow-list {(4, 16)}
  trimul_tail._block          allow-list {(8, 8)}, plus (4, 4) under an opt-in env flag

A fold spawns its workers, so counters read in the launcher are always zero (the mistake
`scripts/lever_census.py` documents at its head). The hook therefore runs in EVERY process via a
generated `sitecustomize.py`, dumps to its own file, and the parent sums.
"""
import argparse, json, os, subprocess, sys
from collections import Counter
from pathlib import Path

KEYS = Counter()        # "reader|kt,nt|served" -> calls
WRAPPED = [False]


def _install():
    """Wrap the three readers. Idempotent, and retried on a timer until the modules exist."""
    if WRAPPED[0]:
        return
    T = sys.modules.get("tt_bio.tenstorrent")
    if T is None or getattr(T, "_allm_mm_wrapped", False):
        return

    def wrap(owner, fname, reader):
        orig = getattr(owner, fname)

        def probe(w, _orig=orig, _r=reader):
            out = _orig(w)
            # An instrument that can raise is worse than no instrument: a probe bug reads as a
            # model defect at the CALLER's line, which is how the first run of this file died
            # inside `trimul_tail.eligible`.
            try:
                kt = (int(w.shape[-2]) + 31) // 32
                nt = (int(w.shape[-1]) + 31) // 32
                KEYS[_r + "|%d,%d|" % (kt, nt) + ("served" if out is not None else "declined")] += 1
            except Exception:                                            # noqa: BLE001
                KEYS[_r + "|probe-error"] += 1
            return out

        setattr(owner, fname, probe)

    wrap(T, "_mm_block_for", "tenstorrent")
    for mod, fname, reader in (("tt_bio.swiglu_fused", "_block", "swiglu"),
                               ("tt_bio.trimul_tail", "_block", "trimul_tail")):
        m = sys.modules.get(mod)
        if m is not None:
            wrap(m, fname, reader)
    T._allm_mm_wrapped = True
    WRAPPED[0] = True


def install_child_hook():
    outdir = os.environ.get("ALLM_MM_CENSUS_DIR")
    if not outdir:
        return
    import atexit, threading, time
    path = os.path.join(outdir, f"pid{os.getpid()}.json")

    def dump():
        if not KEYS:
            return
        T = sys.modules.get("tt_bio.tenstorrent")
        g = getattr(T, "COMPUTE_GRID_MAIN", None) if getattr(T, "COMPUTE_GRID_MEASURED", False) else None
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            json.dump({"pid": os.getpid(), "argv": sys.argv[:4], "keys": dict(KEYS),
                       "grid": f"{int(g[0])}x{int(g[1])}" if g else None}, fh)
        os.replace(tmp, path)

    def tick():
        # Polling, not an import hook: a worker killed by a signal never runs atexit, so the
        # counts have to already be on disk.
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
    ap.add_argument("--python", required=True, help="interpreter of the venv that owns ttnn")
    ap.add_argument("--pythonpath", help="prepended after the hook dir, to census a worktree")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("cli", nargs="*")
    a = ap.parse_args()

    work = Path(a.out).resolve().parent / f".mmcensus-{a.label}"
    hook, dumps = work / "hook", work / "dumps"
    hook.mkdir(parents=True, exist_ok=True)
    dumps.mkdir(parents=True, exist_ok=True)
    for stale in dumps.glob("pid*.json"):
        stale.unlink()
    (hook / "sitecustomize.py").write_text(
        "import sys\n"
        f"sys.path.append({str(Path(__file__).resolve().parent)!r})\n"
        "try:\n"
        "    from mm_key_census import install_child_hook\n"
        "    install_child_hook()\n"
        "except Exception:\n"
        "    pass\n")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(hook)] + ([a.pythonpath] if a.pythonpath else []))
    env["ALLM_MM_CENSUS_DIR"] = str(dumps)
    t0 = __import__("time").time()
    rc = subprocess.call([a.python, *a.cli], env=env)
    wall = round(__import__("time").time() - t0, 3)

    agg, grids, procs = Counter(), set(), 0
    for p in sorted(dumps.glob("pid*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:                                                # noqa: BLE001
            continue
        procs += 1
        agg.update(d["keys"])
        if d.get("grid"):
            grids.add(d["grid"])
    snap = {"label": a.label, "cli": a.cli, "rc": rc, "wall_s": wall, "processes": procs,
            "grid": "/".join(sorted(grids)) or None, "keys": dict(sorted(agg.items()))}
    json.dump(snap, open(a.out, "w"), indent=2)
    grid = snap["grid"]
    print(f"--- {a.label}: rc={rc} {procs} processes, grid={grid}, {wall}s")
    for k, n in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {k:40s} x{n}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
