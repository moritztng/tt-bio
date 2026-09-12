#!/usr/bin/env python3
"""Lay the already-folded killed arms out the way `perf/b2z2_fusebias/score.py` reads them.

The three levers this row re-scores were killed on ONE whole-molecule Kabsch RMSD against a
0.60 A bar, on CIFs that still exist. Re-folding them to re-read them would be waste, so this
script does the only thing that stands between those CIFs and the per-domain instrument: it
renames the arm directories into the scorer's `{size}_{arm}-s{seed}` layout and synthesises the
runs JSON the scorer wants (`tag`, `seed`, `target`, `sha256`, `plddt`, `fold_s`, `env`) out of
the run JSON the original harness already wrote.

`off` is the killed arm's own base, folded in the same process on the same device open, so the
pairing is the original A/B and not a cross-run comparison. Every arm here was folded at seed 0,
so `seed_floor` comes out empty by construction and the floor this row compares against is the
one `b2z2-fusebias-512-parity` measured. The repeats of each arm are byte-identical, which the
scorer records as `aa_floor_bitexact`.

    prepare_existing.py --run <compose json> --src <cifdir> --arm HOST --out <dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="the harness JSON that wrote the CIFs")
    ap.add_argument("--src", type=Path, required=True, help="its cifdir, holding {size}_{arm}_{rep}")
    ap.add_argument("--arm", required=True, help="the killed arm, e.g. HOST or MSA")
    ap.add_argument("--base", default="base")
    ap.add_argument("--size", default="512")
    ap.add_argument("--out", type=Path, required=True, help="scorer cifdir to build")
    ap.add_argument("--runs-out", type=Path, required=True)
    a = ap.parse_args()

    d = json.loads(a.run.read_text())
    timed = [r for r in d["phase1"] if not r["warmup"] and r["target"].endswith(a.size)]

    def first(arm: str) -> dict:
        rs = [r for r in timed if r["arm"] == arm]
        assert rs, f"no timed {arm} fold at {a.size} aa in {a.run}"
        shas = {r["cif_sha256"] for r in rs}
        assert len(shas) == 1, f"{arm} is not self-consistent across reps: {shas}"
        return rs[0]

    # Which rep directory holds which fold: the harness wrote `{size}_{arm}_{i}` in timed order
    # after the discarded warm-up, so rep 0 of the timed set is the first directory that exists.
    def srcdir(arm: str, i: int = 0) -> Path:
        cands = sorted(a.src.glob(f"{a.size}_{arm}_*"), key=lambda p: int(p.name.rsplit("_", 1)[1]))
        assert len(cands) > i, f"no CIF dir for {arm} rep {i} under {a.src}"
        return cands[i]

    plan = [("off-s0", a.base, 0), ("on-s0", a.arm, 0), ("off-s0_r1", a.base, 1)]
    runs = []
    for tag, arm, rep in plan:
        src = srcdir(arm, rep)
        dst = a.out / f"{a.size}_{tag}"
        dst.mkdir(parents=True, exist_ok=True)
        cif = next(src.glob("*.cif"))
        shutil.copy2(cif, dst / cif.name)
        r = first(arm)
        runs.append({"arm": "on" if arm == a.arm else "off", "seed": 0, "tag": tag,
                     "target": r["target"], "fold_s": r["fold_s"],
                     "sha256": r["cif_sha256"][:16], "plddt": r["plddt"],
                     "src_dir": str(src.relative_to(a.src))})

    env = dict(d["env"])
    env["rescored_from"] = str(a.run)
    env["arm_under_test"] = a.arm
    a.runs_out.parent.mkdir(parents=True, exist_ok=True)
    a.runs_out.write_text(json.dumps({"env": env, "runs": runs}, indent=1))
    print(f"{a.arm}: {len(runs)} CIFs -> {a.out}, runs -> {a.runs_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
