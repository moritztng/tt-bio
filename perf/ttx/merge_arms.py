#!/usr/bin/env python3
"""Lay the two stack arms out under one tag each so `roof_shared/assemble.py` and
`b2z2_fusebias/score.py` can score them both against the committed upstream fp32 reference.

`fold_shared.py` labels its shared arm `ttshared-s0` whichever stack ran it, so two arms collide.
This renames them by ttnn version -- `ttshared68`, `ttshared78` -- and nothing else.

    merge_arms.py --old <old/folds.json> --new <new/folds.json> --out <merged/folds.json>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="merged folds.json")
    a = ap.parse_args()

    cifdir = a.out.parent / "cif"
    shutil.rmtree(cifdir, ignore_errors=True)
    cifdir.mkdir(parents=True)

    merged = None
    for src, label in ((a.old, "ttshared68"), (a.new, "ttshared78")):
        d = json.loads(src.read_text())
        if merged is None:
            merged = {"doc": __doc__, "env": {}, "runs": []}
        merged["env"][label] = d["env"]
        for r in d["runs"]:
            size = r["target"].split("_")[-1]
            old_tag = r["tag"]
            new_tag = f"{label}-s{r['seed']}"
            r = dict(r, tag=new_tag, arm=label, stack=label)
            dst = cifdir / f"{size}_{new_tag}"
            dst.mkdir(parents=True)
            cif = next((src.parent / "cif" / f"{size}_{old_tag}").glob("*.cif"))
            shutil.copy(cif, dst / cif.name)
            merged["runs"].append(r)

    a.out.write_text(json.dumps(merged, indent=1))
    print(f"{len(merged['runs'])} runs -> {a.out}")
    for r in merged["runs"]:
        print(" ", r["target"], r["tag"], "plddt", r["plddt"], "sha", r["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
