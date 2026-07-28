#!/usr/bin/env python3
"""Record folds whose artifacts are complete but whose driver died before logging them.

`progress.jsonl` is written by the generate.py driver when a fold returns. If the driver
dies while a fold is running -- the fold keeps going, it is a separate process -- the fold
completes, writes its CIFs and results.json, and is never recorded. `done_pairs()` then
treats it as outstanding and it gets folded a second time.

This scans the output directories and appends an `ok` record for any (target, model) whose
artifacts are complete and which has no `ok` record already.

Everything written is derived from the artifacts on disk, never inferred:
  * `wall_s`, `device`, `host_threads` are recorded as null -- the driver held them and it
    is gone. Do not guess them; a fabricated timing would poison the rate tables that the
    stall scan and the adaptive timeout are built on.
  * `mps` comes from generate.py's constant, and is honest only because the campaign has
    run one value throughout. It is flagged along with everything else by `recovered: true`,
    so any record from this path can be excluded from an analysis that cares.

    python3 scripts/abag_xm_reconcile_orphans.py            # report only
    python3 scripts/abag_xm_reconcile_orphans.py --write    # append the records
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _gen():
    spec = importlib.util.spec_from_file_location(
        "abag_xm_generate", ROOT / "scripts" / "abag_xm_generate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="append the records (default: report)")
    a = ap.parse_args()
    g = _gen()

    recorded_ok = set()
    if g.PROGRESS.exists():
        for line in g.PROGRESS.open():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                recorded_ok.add((r["target"], r["model"]))

    found = []
    for model in g.MODELS:
        out_dir = g.OUT_BASE / model.replace("-", "_")
        if not out_dir.is_dir():
            continue
        prefix = g.RESULT_PREFIX[model]
        for rd in sorted(out_dir.glob(f"{prefix}_results_*")):
            target = rd.name[len(f"{prefix}_results_"):]
            if (target, model) in recorded_ok:
                continue
            rjson = rd / "results.json"
            structures = rd / "structures"
            if not rjson.exists() or not structures.is_dir():
                continue
            cifs = sorted(structures.glob("*.cif"))
            paes = sorted(structures.glob("*_pae.npz"))
            if len(cifs) != g.N_SAMPLES:
                found.append((target, model, len(cifs), len(paes), "INCOMPLETE - not recorded"))
                continue
            rec = {
                "target": target, "model": model,
                "wall_s": None, "device": None, "host_threads": None,
                "n_samples": g.N_SAMPLES, "mps": g.MPS,
                "host": g._HOST, "tt_bio_commit": None,
                "paired_msa": False, "status": "ok",
                "n_cifs": len(cifs), "n_paes": len(paes),
                "result_dir": str(rd),
                "recovered": True,
                "recovered_note": "driver died before logging; fields it held are null",
            }
            found.append((target, model, len(cifs), len(paes), "recoverable"))
            if a.write:
                with g.PROGRESS.open("a") as fp:
                    fp.write(json.dumps(rec) + "\n")

    if not found:
        print("nothing to reconcile: every complete fold already has an ok record")
        return 0
    for t, m, nc, npae, note in found:
        print(f"  {t:6s} {m:14s} cifs={nc:3d} paes={npae:3d}  {note}")
    n = sum(1 for f in found if f[4] == "recoverable")
    print(f"\n{n} recoverable, {len(found) - n} incomplete"
          f"{' -- WRITTEN' if a.write else ' -- report only, pass --write to append'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
