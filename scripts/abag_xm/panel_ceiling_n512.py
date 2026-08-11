#!/usr/bin/env python3
"""Pre-register the final panel size of the 512 rung, from disk, before labelling finishes.

Folding is complete, so the set of targets that CAN reach a rung is already fixed: rung 512
needs all 8 chunk dirs on disk, rung 256 needs 4, rung 64 needs its unchunked dir. Labels are
the only thing still arriving. This walks the dirs and reports, per model:

    CEIL   targets whose chunk dirs are all present (the upper bound labelling is working toward)
    LIVE   of those, the ones whose chunk dirs all carry a labels.json (where labelling is now)

and the 3-rung sets (64 AND 256 AND 512), which is the panel `--panel3` reads on.

Why this exists: without it, the only way to know whether the final panel is complete is to
watch `--panel3`'s denominator stop moving, and a denominator that stops moving looks identical
to labelling that stalled. With it, the final read has a number to hit.

CEIL is an upper bound on the admitted panel, not equal to it: `pool_fold` also drops a target
whose per-sample DockQ is null for the whole fold. On this panel that is exactly three targets,
9ly2 / 9ly3 / 9lz2, in all four models at every rung (undefined native interface, a target-level
property). So the expected admitted panel is CEIL - 3 per model, measured that way rather than
assumed: ceiling minus the script's own admitted count came out 3 in all four models at once.

Exclusions mirror `abag_xm_deepn_analysis.py`: opendde 9sbb (documented mis-fold), and the
large targets held out for device DRAM capacity, an engineering boundary and never a scoring or
biology decision. That set is per model, not a global four -- boltz2 folded all four through
p27/p28 so they are already in its published rungs, protenix-v2 and esmfold2 lack only 9j4c, and
opendde-abag never folded any of them on Wormhole below 512. Window p32 folded the excluded cells
on a separate OOM-fixed tree (commit a6d5b6fda), so those cells alone carry a different engine
than the rest of the panel; DEEPN_P32_EXT=1 admits them as the separately-reported extension
cohort. It cannot change the 3-rung panel, since every cell it admits exists at 512 only and so
fails the intersection anyway (verified both ways, pass 51).

Read-only. Writes a JSON cache of the per-model target lists beside the tree.
"""
import json
import os
from pathlib import Path

BASE = Path.home() / "abag_xm" / "deepn"
GALAXY = BASE / "galaxy"
PREFIX = {"opendde-abag": "opendde", "protenix-v2": "protenix",
          "boltz2": "boltz2", "esmfold2": "esmfold2"}
GALAXY_EXCLUDE = {"opendde-abag": {"9sbb"}}
P32_EXTENSION = {"opendde-abag": {"9i3p", "9j4c", "9ivj", "9q7y"},
                 "protenix-v2": {"9j4c"}, "esmfold2": {"9j4c"}}
P32_EXT_ON = os.environ.get("DEEPN_P32_EXT") == "1"
DOCKQ_NULL = {"9ly2", "9ly3", "9lz2"}
RUNGS = (64, 256, 512)


def scan(model):
    """(ceiling, live) target sets per rung, keyed by rung."""
    root = GALAXY / PREFIX[model]
    dirs = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or "_n" not in d.name:
            continue
        target, rest = d.name.split("_n", 1)
        try:
            rung = int(rest.split("_c")[0])
        except ValueError:
            continue
        if target in GALAXY_EXCLUDE.get(model, ()):
            continue
        if not P32_EXT_ON and target in P32_EXTENSION.get(model, ()):
            continue
        chunk = None
        if "_c" in rest:
            try:
                chunk = int(rest.split("_c")[1])
            except ValueError:
                chunk = None
        m = dirs.setdefault((target, rung),
                            {"chunks": set(), "labelled": set(), "plain": 0, "plain_lab": 0})
        ok = (d / "labels.json").exists()
        if chunk is None:
            m["plain"] += 1
            m["plain_lab"] += 1 if ok else 0
        else:
            m["chunks"].add(chunk)
            if ok:
                m["labelled"].add(chunk)
    ceil = {r: set() for r in RUNGS}
    live = {r: set() for r in RUNGS}
    for (target, rung), m in dirs.items():
        if rung not in RUNGS:
            continue
        if m["plain"] and not m["chunks"]:      # unchunked rung, complete by construction
            ceil[rung].add(target)
            if m["plain_lab"]:
                live[rung].add(target)
            continue
        need = max(1, rung // 64)               # same completeness gate as galaxy64_pools
        if len(m["chunks"]) >= need:
            ceil[rung].add(target)
        if len(m["labelled"]) >= need:
            live[rung].add(target)
    return ceil, live


def main():
    out = {}
    for model in sorted(PREFIX):
        ceil, live = scan(model)
        c3 = set.intersection(*(ceil[r] for r in RUNGS))
        l3 = set.intersection(*(live[r] for r in RUNGS))
        out[model] = {"ceil": {str(r): sorted(ceil[r]) for r in RUNGS},
                      "live": {str(r): sorted(live[r]) for r in RUNGS},
                      "ceil3": sorted(c3), "live3": sorted(l3),
                      "expected_panel": len(c3 - DOCKQ_NULL)}
        print(f"{model:<14} CEIL 64={len(ceil[64]):3d} 256={len(ceil[256]):3d} "
              f"512={len(ceil[512]):3d} | LIVE 512={len(live[512]):3d} | 3-rung CEIL {len(c3):3d} "
              f"LIVE {len(l3):3d} owed {len(c3) - len(l3):3d} | expected panel "
              f"{len(c3 - DOCKQ_NULL):3d}")
    inter = set.intersection(*(set(out[m]["ceil3"]) for m in out))
    live_inter = set.intersection(*(set(out[m]["live3"]) for m in out))
    print(f"\nfour-model 3-rung panel: expected {len(inter - DOCKQ_NULL)}  "
          f"(ceiling {len(inter)}, live now {len(live_inter - DOCKQ_NULL)})")
    out["four_model"] = {"ceil3": sorted(inter), "expected_panel": len(inter - DOCKQ_NULL)}
    dest = BASE / "panel_ceiling.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
