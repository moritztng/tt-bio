#!/usr/bin/env python3
"""Does a flag make a model's output depend on the core grid?

WHY THIS EXISTS AS A SEPARATE INSTRUMENT. "Output that depends on which card ran it" is a standing
hard stop and is not scored against the Angstrom bar. The release gate has exactly one check for
it, the `l1-budget` arm, and that arm has a blind spot this row hit: it folds **protenix-v2**, and a
lever that never executes in a protenix-v2 fold gets a GATE PASS that says nothing about it.
Measured, not argued -- `TT_BIO_MM_SHORT_M_BW` passed that arm across native/8x8/narrow with one
md5 while making the Boltz-2 structure grid-dependent, because protenix-v2 makes 0 calls to it
against 576 served on Boltz-2 (`perf/c14_land/mmshort_firing/summary.json`).

So the check this instrument makes that the arm does not: **it refuses to return a verdict unless
the lever actually fired.** A green from a check that never ran the code it is checking is the
failure mode, not an edge case.

FOUR FOLDS, and the flag-off pair is not optional:

    flag=1 gridA   vs   flag=1 gridB     the question
    flag=0 gridA   vs   flag=0 gridB     the control

If the control pair disagrees, the model is grid-dependent WITHOUT the flag and the finding is
about the shipped tree rather than about the lever -- a bigger result, and one that must stop the
run rather than be reported as a flag verdict.

Admissible on a loud host and a busy board-pair sibling: a CIF digest and a call count do not move
with the AICLK. Nothing here is a wall-clock claim.

    python3 perf/c14_land/grid_independence.py \
        --model boltz2 --fixture perf/size512/fixtures/cdk2x2_512.yaml \
        --flag TT_BIO_MM_SHORT_M_BW \
        --wrap tt_bio.tenstorrent:_short_m_proj_config \
        --grids 11,10 8,8 --steps 6 --card 2 --out perf/c14_land/gi_mmshort.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def seed_msa(fixture: Path, workroot: Path) -> Path | None:
    """Pin the MSA, because an unpinned one silently breaks the comparison.

    `tt_bio.main predict` resolves an MSA from the ColabFold server for every model in
    MSA_DEFAULT_MODELS. Four legs would each fetch their own, and two legs that disagree on their
    input cannot be compared on their output -- the run would read GRID-DEPENDENT for a reason
    that has nothing to do with the grid. The fixtures ship a cached `.a3m` beside each `.yaml`,
    so seed one directory once and give every leg the same `--msa_dir`.
    """
    a3m = fixture.with_suffix(".a3m")
    if not a3m.exists():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_b2x", REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    d = workroot / "msa"
    d.mkdir(parents=True, exist_ok=True)
    m._seed_msa(fixture, a3m.read_text(), d)
    return d


def fold(model, fixture, flag, on, grid, steps, card, wrap, workroot, log_dir, msa_dir=None):
    """One fold. Returns (md5, firing_calls, rc, log path)."""
    out = Path(tempfile.mkdtemp(prefix=f"gi-{flag}-{int(on)}-{grid[0]}x{grid[1]}-",
                                dir=str(workroot)))
    fire = out / "firing"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}:{REPO / 'perf' / 'c14_land' / 'gishim'}"
    env[flag] = "1" if on else "0"
    env["TT_BIO_FORCE_GRID"] = f"{grid[0]},{grid[1]}"
    env["TT_VISIBLE_DEVICES"] = str(card)
    env["TT_BIO_LEASE_CARDS"] = str(card)
    env["TT_BIO_LEASE_HOLDER"] = "worker:c14-land-tail"
    if wrap:
        env["C14_GI_WRAP"] = wrap
        env["C14_GI_OUT"] = str(fire)
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(REPO / fixture),
           "--model", model, "--sampling_steps", str(steps), "--diffusion_samples", "1",
           "--seed", "42", "--out_dir", str(out)]
    if msa_dir is not None:
        cmd += ["--msa_dir", str(msa_dir)]
    log = log_dir / f"{flag}_{int(on)}_{grid[0]}x{grid[1]}.log"
    with open(log, "wb") as fh:
        rc = subprocess.call(cmd, cwd=str(REPO), env=env, stdout=fh,
                             stderr=subprocess.STDOUT)
    cifs = sorted(out.rglob("*.cif"))
    md5 = hashlib.md5(cifs[0].read_bytes()).hexdigest() if cifs else None
    calls = 0
    for f in glob.glob(str(fire) + ".*"):
        try:
            calls += json.load(open(f))["calls"]
        except Exception:                                             # noqa: BLE001
            pass
    shutil.rmtree(out, ignore_errors=True)
    # rc is the SUBPROCESS's status, not a pipeline's, and the md5 is asserted separately --
    # a fold that exits 0 and writes no CIF is still a failed leg (`rc-after-a-pipeline-is-the-
    # last-commands-status`).
    return {"md5": md5, "firing_calls": calls, "rc": rc, "log": str(log)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True, help="repo-relative")
    ap.add_argument("--flag", required=True, help="env var name, e.g. TT_BIO_MM_SHORT_M_BW")
    ap.add_argument("--wrap", default=None,
                    help="module:attr to count, e.g. tt_bio.tenstorrent:_short_m_proj_config. "
                         "Without it the run cannot prove the lever fired and the verdict is "
                         "downgraded to UNPROVEN however the digests come out.")
    ap.add_argument("--grids", nargs=2, required=True, help="two grids as gx,gy")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--card", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    grids = [tuple(int(v) for v in g.split(",")) for g in args.grids]
    workroot = REPO / "perf" / "c14_land" / "gi_work"
    workroot.mkdir(parents=True, exist_ok=True)
    log_dir = args.out.parent / (args.out.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    msa_dir = seed_msa(REPO / args.fixture, workroot)

    out = {
        "model": args.model, "fixture": args.fixture, "flag": args.flag, "wrap": args.wrap,
        "msa_dir": str(msa_dir) if msa_dir else None,
        "grids": [list(g) for g in grids], "steps": args.steps, "card": args.card,
        "git_head": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "digest + call count only; no wall-clock claim, so a loud host and a busy "
                "board-pair sibling do not invalidate it. MSA pinned to the fixture's cached "
                "a3m, so all four legs fold the same input.",
        "legs": [],
    }

    def dump():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))

    for on in (True, False):
        for g in grids:
            r = fold(args.model, args.fixture, args.flag, on, g, args.steps,
                     args.card, args.wrap, workroot, log_dir, msa_dir)
            r.update({"flag_on": on, "grid": list(g)})
            out["legs"].append(r)
            dump()
            print(f"  {args.flag}={int(on)} grid {g[0]}x{g[1]}  rc={r['rc']}  "
                  f"md5={r['md5']}  fired={r['firing_calls']}", flush=True)

    on_legs = [l for l in out["legs"] if l["flag_on"]]
    off_legs = [l for l in out["legs"] if not l["flag_on"]]
    out["all_legs_produced_a_structure"] = all(l["md5"] for l in out["legs"])
    out["on_md5s"] = sorted({l["md5"] for l in on_legs})
    out["off_md5s"] = sorted({l["md5"] for l in off_legs})
    out["firing_on"] = sum(l["firing_calls"] for l in on_legs)
    out["firing_off"] = sum(l["firing_calls"] for l in off_legs)

    if not out["all_legs_produced_a_structure"]:
        out["verdict"] = "INVALID: a leg produced no structure; see the logs"
    elif len(out["off_md5s"]) > 1:
        out["verdict"] = (
            f"CONTROL DIRTY -- {args.model} is grid-dependent WITHOUT {args.flag} "
            f"({out['off_md5s']}). That is a property of the shipped tree, not of the flag, and "
            "it outranks any flag verdict. Stop and report it.")
    elif args.wrap and out["firing_on"] == 0:
        out["verdict"] = (
            f"BLIND -- {args.flag} never fired in this fold (0 calls to {args.wrap}), so the "
            "matching digests say nothing about it. This is the l1-budget arm's failure mode and "
            "it is why this instrument counts.")
    elif len(out["on_md5s"]) > 1:
        out["verdict"] = (
            f"GRID-DEPENDENT -- {args.flag} makes {args.model}'s output depend on the core grid "
            f"({out['on_md5s']}) while the flag-off control is stable. Standing hard stop: NO-GO "
            "as a shipped default whatever its seconds. It may still ship opt-in, default off.")
    elif not args.wrap:
        out["verdict"] = (
            "UNPROVEN -- digests agree across grids, but with no --wrap the run cannot show the "
            "lever fired, so this is not evidence of grid-neutrality.")
    else:
        out["verdict"] = (
            f"GRID-NEUTRAL -- {args.flag} fired {out['firing_on']} times and the output is "
            f"identical across {len(grids)} grids, with a stable flag-off control.")
    dump()
    print("\n" + out["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
