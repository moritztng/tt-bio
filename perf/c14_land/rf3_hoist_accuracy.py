#!/usr/bin/env python3
"""RF3 CA-lDDT arm for `TT_BIO_DIT_COND_HOIST` -- the one reading between +0.2052 s and a flip.

`tenstorrent.py`'s comment claimed this flag was boltz-2-exclusive. It is not: `rf3/token_dit.py:85`
builds `DiffusionTransformer(atom_level=False)`, which is exactly the branch at `tenstorrent.py:9707`
the flag gates. So flipping the default is a change to RF3 as well, and RF3 has never been scored
for it.

Score it the way ask 8879's withdrawal says to, not the way its approval did. `TT_BIO_UNFUSED_SILU`
was approved on a Boltz-2 structural-RMSD-against-seed-scatter reading and had to be withdrawn when
Protenix-v2 turned out to lose 0.05-0.07 CA-lDDT vs 1HCL. `b2z2-union-land` showed why that screen
is blind: a model can scatter widely on this fixture while landing at the same quality every time,
so its structural spread is loose and its accuracy spread is tight. Only the comparison against the
experimental answer separates the arms. This driver therefore folds seeds per arm and hands them to
`perf/b2z2_fusebias/score.py`, whose `native_summary` is CA-lDDT against 1HCL per pseudo-domain.

FIRING CONTROL, and it is free. `cdk2x2_512` is CDK2 followed by its own residues 1-214, so both
pseudo-domains have a native answer -- and the two arms must write DIFFERENT CIFs. If base and on
are byte-identical at the same seed the lever did not fire on this path and the accuracy reading is
vacuous, not clean. The driver says so instead of reporting a null as a pass. It also asserts the
env flag reached the module in the child, because setting a variable nothing reads succeeds
silently and would serve both arms the same code.

This is an ACCURACY run, not a wall-clock read: it may be taken on a contended pair. `fold_s` is
recorded for bookkeeping and is NOT a measurement.

    rf3_hoist_accuracy.py --cifdir <dir> --out <runs.json> [--seeds 0,1,2,3] [--sizes 512]

then

    python3 perf/b2z2_fusebias/score.py <dir> --runs <runs.json> --split 298 \
        --arms base,on --out <scored.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "perf" / "size512" / "fixtures"
FLAG = "TT_BIO_DIT_COND_HOIST"
READER = "_B2_DIT_COND_HOIST"
ARMS = {"base": "0", "on": "1"}


def _plddt(results: dict) -> float:
    """The model's own plDDT out of its results.json, whatever it nested it under.

    Prefer an exact `plddt`: `confidence_score` is 0.8*plDDT + 0.2*pTM and lands ~0.05 out with a
    sign that flips with size, which is what made the plDDT rows in perf/k10_p2 disagree with the
    B-factor column. The scorer re-checks whatever we report against that column anyway.
    """
    hits: dict[str, float] = {}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (int, float)) and "plddt" in k.lower():
                    hits[k] = float(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(results)
    for k in ("plddt", "complex_plddt", "mean_plddt"):
        if k in hits:
            return hits[k]
    return hits[sorted(hits)[0]] if hits else float("nan")


def child() -> int:
    """One fold, in its own process, with the flag set before `tt_bio` is imported."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    os.environ[FLAG] = a.value
    sys.path.insert(0, str(REPO))
    from tt_bio import tenstorrent as T

    want = a.value not in ("0", "", "false", "False")
    got = getattr(T, READER)
    assert got == want, f"{FLAG}={a.value} did not reach {READER} (module reads {got})"

    argv = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    from tt_bio.main import cli

    t0 = time.perf_counter()
    rc = 0
    try:
        cli(argv, standalone_mode=False)
    except SystemExit as e:
        rc = e.code or 0
    a.report.write_text(json.dumps({"rc": rc, "fold_s": round(time.perf_counter() - t0, 3),
                                    "reader": bool(got)}))
    return rc


def fold(size: int, arm: str, seed: int, tag: str, cifdir: Path, msa_dir: Path,
         model: str, extra: list[str]) -> dict:
    target = f"cdk2x2_{size}"
    dest = cifdir / f"{size}_{tag}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix=f"rf3hoist_{tag}_") as work:
        out_dir = Path(work) / "out"
        report = Path(work) / "report.json"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child",
               "--value", ARMS[arm], "--report", str(report), "--",
               "predict", str(FIX / f"{target}.yaml"), "--model", model,
               "--seed", str(seed), "--out_dir", str(out_dir),
               "--msa_dir", str(msa_dir), "--output_format", "cif", *extra]
        print(f"[fold] {tag}  {' '.join(cmd[6:])}", flush=True)
        rc = subprocess.call(cmd)
        rep = json.loads(report.read_text()) if report.exists() else {"rc": rc, "fold_s": None}
        assert rep["rc"] == 0 and rc == 0, f"{tag} fold failed rc={rc}/{rep['rc']}"
        cifs = sorted(out_dir.rglob("*.cif"))
        assert cifs, f"{tag} wrote no CIF under {out_dir}"
        res = sorted(out_dir.rglob("results.json"))
        results = json.loads(res[0].read_text()) if res else {}
        for c in cifs:
            shutil.copy2(c, dest / c.name)
        sha = hashlib.sha256((dest / cifs[0].name).read_bytes()).hexdigest()
    return {"target": target, "tag": tag, "arm": arm, "seed": seed,
            "plddt": _plddt(results), "sha256": sha,
            "fold_s_NOT_A_MEASUREMENT": rep["fold_s"], "fold_s": rep["fold_s"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", default="512")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--model", default="rf3")
    ap.add_argument("--extra", default="", help="extra args passed to every fold, space separated")
    a = ap.parse_args()

    assert FLAG not in os.environ, (
        f"{FLAG} is pinned in the environment; it would serve every arm the same code")

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))
    import ab_flag_levers as AB

    sizes = [int(s) for s in a.sizes.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    extra = a.extra.split() if a.extra else []
    a.cifdir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="rf3hoist_msa_") as msa:
        msa_dir = Path(msa)
        for size in sizes:
            AB._seed_msa(FIX / f"cdk2x2_{size}.yaml",
                         (FIX / f"cdk2x2_{size}.a3m").read_text(), msa_dir)
        for size in sizes:
            # Arms interleave seed by seed so a drift over the session hits both equally, and the
            # first fold is repeated LAST as the A/A floor -- after every flag-on fold, so order
            # and contamination are controlled the way b2z2-union-land controlled them.
            for seed in seeds:
                for arm in ("base", "on"):
                    runs.append(fold(size, arm, seed, f"{arm}-s{seed}",
                                     a.cifdir, msa_dir, a.model, extra))
            runs.append(fold(size, "base", seeds[0], f"base-s{seeds[0]}_r1",
                             a.cifdir, msa_dir, a.model, extra))

    by = {r["tag"]: r for r in runs}
    firing = {}
    for size in sizes:
        for seed in seeds:
            b, o = by.get(f"base-s{seed}"), by.get(f"on-s{seed}")
            if b and o:
                firing[f"{size} s{seed}"] = b["sha256"] != o["sha256"]
    aa = {t: by[t]["sha256"] == by[t[:-3]]["sha256"] for t in by if t.endswith("_r1")}

    report = {"host": socket.gethostname(), "model": a.model, "flag": FLAG,
              "env": {k: v for k, v in sorted(os.environ.items())
                      if k.startswith(("TT_", "TT_BIO_"))},
              "runs": runs, "arms_differ": firing, "aa_bitexact": aa}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))

    print("\n=== firing control: base vs on must differ ===")
    for k, v in firing.items():
        print(f"  {k:12s} arms differ: {v}")
    print("=== A/A floor: the repeat must be bit-exact ===")
    for k, v in aa.items():
        print(f"  {k:16s} bit-exact: {v}")
    if not all(firing.values()):
        print("\nNOT FIRING on at least one seed -- the accuracy reading there is VACUOUS, "
              "not clean. Do not score it as a pass.")
    print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    if "--child" in sys.argv:
        sys.argv.remove("--child")
        raise SystemExit(child())
    raise SystemExit(main())
