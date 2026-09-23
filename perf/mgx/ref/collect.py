#!/usr/bin/env python3
"""Turn the box's raw reference output into the committed set: gzipped CIFs plus manifest.json.

    rsync -a box:/root/refs/ /tmp/mgx_refs/          # or scp -r
    python3 perf/mgx/ref/collect.py /tmp/mgx_refs

For every model in plan.json (PREDICT_MODELS) x every fixture, it copies each ok seed's CIF to
perf/mgx/ref/refs/<model>/<fixture>/s<seed>.cif.gz and writes one manifest cell carrying the
per-seed record (upstream packages, checkpoint, GPU, dtype, device time, peak memory). A cell
with no ok seed gets `missing_reason`: the upstream's own error, or "not run" if the box never
reached it. Then it scores each cell's seed floor and the reference against the crystal, so the
manifest carries the numbers the scorer quotes.
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from score import compare  # noqa: E402

KEEP = ("status", "error", "gpu", "pkgs", "dtype", "checkpoint", "repo", "revision", "argv",
        "fold_kwargs", "recycles", "sampling_steps", "samples", "tokens", "device_s", "wall_s",
        "peak_mem_GiB_torch", "started")


def main() -> int:
    raw = Path(sys.argv[1])
    plan = json.loads((HERE / "plan.json").read_text())
    fx = json.loads((HERE / "fixtures" / "fixtures.json").read_text())["fixtures"]
    old = json.loads((HERE / "manifest.json").read_text()) if (HERE / "manifest.json").exists() else {}
    cells = old.get("cells", {})
    for m in plan["models"]:
        for f, meta in fx.items():
            key = f"{m}/{f}"
            cell = cells.get(key, {"seeds": {}})
            for s in plan["seeds"]:
                rp = raw / m / f / f"s{s}.json"
                if not rp.is_file():
                    continue
                r = json.loads(rp.read_text())
                ent = {k: r[k] for k in KEEP if k in r}
                if r.get("status") == "ok":
                    dst = HERE / "refs" / m / f / f"s{s}.cif.gz"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with open(raw / m / f / f"s{s}.cif", "rb") as src, gzip.open(dst, "wb") as out:
                        shutil.copyfileobj(src, out)
                    ent["cif"] = str(dst.relative_to(ROOT))
                cell["seeds"][str(s)] = ent
            ok = {s: e for s, e in cell["seeds"].items() if e.get("status") == "ok"}
            cell.pop("missing_reason", None)
            cell.pop("floor", None)
            if not ok:
                errs = [f"s{s} {e.get('status')}: {e.get('error', '')[:300]}"
                        for s, e in cell["seeds"].items()]
                cell["missing_reason"] = "; ".join(errs) if errs else "not run"
            elif len(ok) >= 2:
                a, b = sorted(ok)[:2]
                cell["floor"] = compare(str(ROOT / ok[b]["cif"]), str(ROOT / ok[a]["cif"]))
            if ok and meta.get("ground_truth"):
                cell["vs_crystal"] = {f"s{s}": compare(str(ROOT / e["cif"]), str(ROOT / meta["ground_truth"]))
                                      for s, e in sorted(ok.items())}
            cells[key] = cell
    manifest = {
        "about": "Upstream reference structures for the MGX 1024/1280/1536 rungs. Score a TT fold "
                 "with perf/mgx/ref/score.py; the floor is reference seed 0 vs seed 1.",
        "models": list(plan["models"]), "fixtures": list(fx), "seeds": plan["seeds"],
        "settings_source": plan["source"], "cells": cells}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    n_ok = sum(1 for c in cells.values() if "missing_reason" not in c)
    n_floor = sum(1 for c in cells.values() if "floor" in c)
    print(f"cells with a reference: {n_ok}/{len(plan['models']) * len(fx)}, with a floor: {n_floor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
