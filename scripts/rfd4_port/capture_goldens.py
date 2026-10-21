"""Capture upstream feature goldens for every shipped RFD4 benchmark example.

One .npz per input under state/rfd4/goldens-io/<family>/<stem>.npz, plus an INDEX.json
naming the upstream sha, the inputs and any failures. Parity for this row is scored on every
family, not on one example, so the corpus is the whole benchmark tree.

Runs in the upstream reference env only. Resumable: an existing .npz is skipped.

Work is handed out in chunks, one worker process per chunk, because importing rfproteina and
torch costs ~30 s and featurizing one example costs ~15 s. A process per file spent four
fifths of the wall clock on imports.

    rfd4-ref-env/venv/bin/python -I capture_goldens.py [--jobs N] [--chunk N]
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BENCH = Path("/home/moritz/rfd4-upstream/RFD4-Proteina/data/benchmarks")
OUT = Path("/home/moritz/.coworker/state/rfd4/goldens-io")
PY = "/home/moritz/rfd4-ref-env/venv/bin/python"
SELF = Path(__file__).resolve()


def discover():
    """benchmark inputs as (family, src, dst) triples, newest-style suffixes only."""
    jobs = []
    for p in sorted(BENCH.rglob("*.cif*")):
        if p.suffix not in (".zst", ".gz", ".cif"):
            continue
        family = p.relative_to(BENCH).parts[0]
        stem = p.name.split(".cif")[0]
        jobs.append((family, str(p), str(OUT / family / f"{stem}.npz")))
    return jobs


def run_chunk(chunk):
    """Worker entry: featurize a list of (family, src, dst) in ONE interpreter."""
    from dump_upstream_features import dump  # noqa: PLC0415 — after sys.path is set

    out = []
    for family, src, dst in chunk:
        if Path(dst).exists():
            out.append((src, "skip", 0.0, ""))
            continue
        t0 = time.time()
        try:
            dump(Path(src), Path(dst))
            out.append((src, "ok", time.time() - t0, ""))
        except Exception as e:  # one bad input must not lose the chunk
            out.append((src, "fail", time.time() - t0, f"{type(e).__name__}: {e}"[:300]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args()

    sys.path.insert(0, str(SELF.parent))
    jobs = discover()
    todo = [j for j in jobs if not Path(j[2]).exists()]
    fams = {}
    for f, _, _ in jobs:
        fams[f] = fams.get(f, 0) + 1
    print(f"{len(jobs)} inputs across {len(fams)} families ({len(todo)} to do): "
          + ", ".join(f"{k}={v}" for k, v in sorted(fams.items())), flush=True)

    chunks = [todo[i:i + a.chunk] for i in range(0, len(todo), a.chunk)]
    results, done = [], 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for res in ex.map(run_chunk, chunks):
            results.extend(res)
            done += len(res)
            for r in res:
                if r[1] == "fail":
                    print(f"  FAIL {Path(r[0]).name}: {r[3]}", flush=True)
            rate = sum(r[2] for r in results if r[1] == "ok") / max(1, sum(r[1] == "ok" for r in results))
            print(f"[{done}/{len(todo)}] {rate:.1f}s/example", flush=True)

    ok = sum(r[1] == "ok" for r in results)
    fail = [r for r in results if r[1] == "fail"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "INDEX.json").write_text(json.dumps({
        "upstream_sha": "2db07bb444d91695f8c10e6acb7648e9297f9221",
        "benchmarks": str(BENCH),
        "inputs": len(jobs),
        "captured": len(list(OUT.rglob("*.npz"))),
        "failed": len(fail),
        "families": dict(sorted(fams.items())),
        "failures": [{"input": r[0], "error": r[3]} for r in fail],
    }, indent=2) + "\n")
    print(f"captured {ok} new, {len(fail)} failed, "
          f"{len(list(OUT.rglob('*.npz')))} npz on disk", flush=True)


if __name__ == "__main__":
    main()
