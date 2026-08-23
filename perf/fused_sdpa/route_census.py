"""Which SDPA route a fold's triangle attention actually took, summed over every process.

`tt-bio predict` folds in spawned worker processes, so a counter read in the launcher is always
zero (`scripts/lever_census.py` says the same thing at more length). This is the same trick for
the three counters that decide whether the fused-SDPA precision lever is reachable at all:

    tt_bio.tenstorrent.SDPA_ROUTE_COUNTS      calls served, per route
    tt_bio.tenstorrent.SDPA_CHUNK_PICKS       the (q_chunk, k_chunk, route) each shape settled on
    tt_bio.triatt_sdpa.STATS                  [served, declined] inside the fused kernel's guard
    tt_bio.triatt_sdpa.REJECTS                why it declined, per (reason, shape)

Usage:

    python3 perf/fused_sdpa/route_census.py --out c.json --pythonpath $WT -- \
        <python> -m tt_bio.main predict f.yaml --model boltz2 ...
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

COUNTERS = [
    ("tt_bio.tenstorrent", "SDPA_ROUTE_COUNTS"),
    ("tt_bio.tenstorrent", "SDPA_HIFI_CALLS"),
    ("tt_bio.tenstorrent", "SDPA_CHUNK_PICKS"),
    ("tt_bio.tenstorrent", "SDPA_K_CHUNK_STATS"),
    ("tt_bio.tenstorrent", "TRIATT_FUSED_HIFI_STATS"),
    ("tt_bio.triatt_sdpa", "STATS"),
    ("tt_bio.triatt_sdpa", "REJECTS"),
]


def install_hook():
    """Runs in every process of the fold, via the generated sitecustomize."""
    import atexit

    outdir = os.environ.get("ROUTE_CENSUS_DIR")
    if not outdir:
        return

    def dump():
        rows = {}
        for mod, attr in COUNTERS:
            m = sys.modules.get(mod)
            if m is None:
                continue
            v = getattr(m, attr, None)
            if v is None:
                continue
            if isinstance(v, dict):
                rows[f"{mod}.{attr}"] = {str(k): val for k, val in v.items()}
            else:
                rows[f"{mod}.{attr}"] = list(v)
        if not rows:
            return
        try:
            p = Path(outdir) / f"pid{os.getpid()}.json"
            p.write_text(json.dumps(rows))
        except Exception:                                                # noqa: BLE001
            pass

    atexit.register(dump)


def _merge(agg, rows):
    for key, v in rows.items():
        if isinstance(v, list):
            cur = agg.setdefault(key, [0] * len(v))
            for i, n in enumerate(v):
                cur[i] += n
        else:
            cur = agg.setdefault(key, {})
            for k, n in v.items():
                if isinstance(n, (int, float)):
                    cur[k] = cur.get(k, 0) + n
                else:
                    cur[k] = n           # SDPA_CHUNK_PICKS: a route, not a count
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pythonpath", help="prepended after the hook dir, to census a worktree")
    ap.add_argument("cli", nargs="*")
    args = ap.parse_args()
    if not args.cli:
        ap.error("give the CLI after `--`")

    work = Path(args.out).resolve().parent / (".route-" + Path(args.out).stem)
    hookdir, dumpdir = work / "hook", work / "dumps"
    hookdir.mkdir(parents=True, exist_ok=True)
    dumpdir.mkdir(parents=True, exist_ok=True)
    for stale in dumpdir.glob("pid*.json"):
        stale.unlink()
    # Appended to sys.path, not prepended: nothing here may shadow a stdlib module for the
    # process under test.
    (hookdir / "sitecustomize.py").write_text(
        "import sys\n"
        f"sys.path.append({str(Path(__file__).resolve().parent)!r})\n"
        "try:\n"
        "    from route_census import install_hook\n"
        "    install_hook()\n"
        "except Exception:\n"
        "    pass\n")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(hookdir)] + ([args.pythonpath] if args.pythonpath else []))
    env["ROUTE_CENSUS_DIR"] = str(dumpdir)
    rc = subprocess.call(args.cli, env=env)

    agg = {}
    dumps = sorted(dumpdir.glob("pid*.json"))
    for p in dumps:
        _merge(agg, json.loads(p.read_text()))
    snap = {"cli": args.cli, "rc": rc, "processes": len(dumps), "counters": agg}
    json.dump(snap, open(args.out, "w"), indent=2)
    print(f"--- {len(dumps)} processes, rc={rc}")
    for k, v in agg.items():
        print(f"{k}: {json.dumps(v)[:600]}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
