#!/usr/bin/env python3
"""Lay already-folded killed arms out the way `perf/b2z2_fusebias/score.py` reads them.

The levers this row re-scores were killed on ONE whole-molecule Kabsch RMSD against a 0.60 A bar,
on CIFs that still exist. Re-folding them to re-read them would be waste, so this script does the
only thing that stands between those CIFs and the per-domain instrument: it renames the arm
directories into the scorer's `{size}_{arm}-s{seed}` layout and synthesises the runs JSON the
scorer wants (`tag`, `seed`, `target`, `sha256`, `plddt`, `fold_s`, `env`) out of the run JSON the
original harness already wrote.

`off` is the killed arm's own base, folded in the same process on the same device open, so the
pairing is the original A/B and not a cross-run comparison. Every arm here was folded at one seed,
so `seed_floor` comes out empty by construction and the floor this row compares against is the one
`b2z2-fusebias-512-parity` measured. The base's own repeat becomes the A/A pair, which the scorer
records as `aa_floor_bitexact`.

Two harnesses wrote these CIFs and they disagree about their JSON, so there are two readers and one
output. Nothing else differs between them.

    prepare_existing.py --run <json> --src <cifdir> --arm HOST --out <dir> --runs-out <json>
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def read_compose(d: dict, size: str) -> tuple[list[dict], dict]:
    """`perf/b2z2_compose/ab_compose.py`: phase1 records, warm-ups flagged, one CIF sha each."""
    rows = [{"arm": r["arm"], "fold_s": r["fold_s"], "plddt": r["plddt"],
             "sha256": r["cif_sha256"][:16], "target": r["target"]}
            for r in d["phase1"] if not r["warmup"] and r["target"].endswith(size)]
    return rows, dict(d["env"])


def read_fusion(d: dict, size: str) -> tuple[list[dict], dict]:
    """`perf/b2z2_fusion/accuracy.py`: a flat `runs` list keyed by size, sha in a per-file dict."""
    rows = [{"arm": r["arm"], "fold_s": r["fold_s"], "plddt": r["plddt"],
             "sha256": next(iter(r["cif_sha256"].values())), "target": f"cdk2x2_{size}"}
            for r in d["runs"] if str(r["size"]) == size]
    env = {k: d[k] for k in ("host", "card", "ttnn", "model", "recycling_steps", "sampling_steps",
                             "fused_grid", "block_config", "mul_mode") if k in d}
    return rows, env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="the harness JSON that wrote the CIFs")
    ap.add_argument("--src", type=Path, required=True, help="its cifdir, holding {size}_{arm}_{rep}")
    ap.add_argument("--arm", required=True, help="the killed arm, e.g. HOST or swiglu_bf16")
    ap.add_argument("--base", default="base")
    ap.add_argument("--sizes", default="512")
    ap.add_argument("--out", type=Path, required=True, help="scorer cifdir to build")
    ap.add_argument("--runs-out", type=Path, required=True)
    a = ap.parse_args()

    d = json.loads(a.run.read_text())
    read = read_compose if "phase1" in d else read_fusion
    runs, env = [], {}
    for size in a.sizes.split(","):
        rows, env = read(d, size)

        def one(arm: str) -> dict:
            rs = [r for r in rows if r["arm"] == arm]
            assert rs, f"no timed {arm} fold at {size} aa in {a.run}"
            shas = {r["sha256"] for r in rs}
            assert len(shas) == 1, f"{arm} is not self-consistent across reps at {size} aa: {shas}"
            return rs[0]

        def srcdir(arm: str, i: int) -> Path:
            # Anchored on the rep digits, because a bare `{size}_{arm}_*` glob also matches every
            # arm whose name EXTENDS this one -- `512_swiglu_*` swallows `512_swiglu_appx_0` and
            # hands the plain arm the approximated arm's structure.
            rx = re.compile(rf"^{size}_{re.escape(arm)}_(\d+)$")
            c = sorted((p for p in a.src.iterdir() if rx.match(p.name)),
                       key=lambda p: int(rx.match(p.name).group(1)))
            assert len(c) > i, f"no CIF dir for {arm} rep {i} at {size} aa under {a.src}"
            return c[i]

        for tag, arm, rep in (("off-s0", a.base, 0), ("on-s0", a.arm, 0), ("off-s0_r1", a.base, 1)):
            dst = a.out / f"{size}_{tag}"
            dst.mkdir(parents=True, exist_ok=True)
            cif = next(srcdir(arm, rep).glob("*.cif"))
            shutil.copy2(cif, dst / cif.name)
            r = one(arm)
            runs.append({"arm": "on" if arm == a.arm else "off", "seed": 0, "tag": tag,
                         "target": r["target"], "fold_s": r["fold_s"], "sha256": r["sha256"],
                         "plddt": r["plddt"], "src_dir": srcdir(arm, rep).name})

    env["rescored_from"] = str(a.run)
    env["arm_under_test"] = a.arm
    a.runs_out.parent.mkdir(parents=True, exist_ok=True)
    a.runs_out.write_text(json.dumps({"env": env, "runs": runs}, indent=1))
    print(f"{a.arm}: {len(runs)} CIFs -> {a.out}, runs -> {a.runs_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
