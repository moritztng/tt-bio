#!/usr/bin/env python3
"""Does TT_BIO_APB_CONCAT_HEADS make the fold's output depend on the core grid?

WHY THIS EXISTS. Region T (TT_BIO_TRIATT_B8) was measured at a real +0.2020 s, was accuracy-clear
at 0.37848 A against a 0.60 A bar, and was still refused as a shipped default, because the release
gate's l1-budget arm showed its output moved with the core grid: the same input folded on an 8x8
grid and on the native grid gave two different structures. That is a standing hard stop in this
fleet's brief -- "output that depends on which card ran it" -- and it is NOT scored against the
Angstrom bar, because JapanFold dispatches across mixed and harvested grids.

TT_BIO_APB_CONCAT_HEADS is in the same risk class and nobody has checked it. It keeps the padded
head lanes, so proj_g and proj_o run at n_heads*padded_head_dim = 1024 instead of the narrow 768.
Adding zero lanes is exactly neutral in isolation (x + 0 = x in IEEE arithmetic), but the K
dimension of both matmuls changes, so the blocking changes, so the ORDER the real 768 lanes are
summed in changes -- and the blocking is derived from the live core grid. That is precisely the
mechanism that sank region T, arriving by a different route.

PRE-REGISTERED OUTCOMES, written before the run and committed before the first device open:

  A. base arm gives ONE digest across all grids AND on arm gives ONE digest across all grids
     -> APB is grid-independent. The precondition is discharged and the landing decision is
        purely the seconds question. (The two arms' digests differ from each other; that is
        expected and is not what is being tested.)
  B. base arm gives ONE digest, on arm gives MORE THAN ONE
     -> APB makes the output depend on which card ran it. HARD STOP: NO-GO as a shipped default
        whatever its seconds, exactly as region T. It may still ship opt-in, defaulting off.
  C. base arm itself gives MORE THAN ONE digest
     -> shipping main is already grid-dependent on boltz-2. That is a pre-existing correctness
        bug, more serious than anything on this row, and this test is inconclusive about APB.
        Report the digests and hand it up; do not read it as evidence either way about the flag.

The control that makes a green meaningful is leg-internal: each leg folds twice (a discarded
warmup and one scored fold) and both digests are recorded, so a leg that cannot even reproduce
itself is visible rather than being averaged into an agreement.

Contention note: this is a DIGEST comparison, not a wall-clock read. The board-pair power budget
moves the AICLK and therefore the seconds; it does not move which bits a fold computes. So this
runs on a loud box on purpose, and no timing guard is taken and no seconds are claimed.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "perf" / "c14_land" / "apb_fold_ab.py"


def leg(arm: str, grid, card: str, size: str, scratch: Path) -> dict:
    label = "native" if grid is None else f"{grid[0]}x{grid[1]}"
    out = scratch / f"{arm}_{label}.json"
    env = os.environ.copy()
    env["TT_BIO_APB_CONCAT_HEADS"] = "1" if arm == "on" else "0"
    env["TT_VISIBLE_DEVICES"] = card
    env["TT_BIO_LEASE_CARDS"] = card
    env["TT_BIO_LEASE_HOLDER"] = "worker:c14-land-tail"
    env.pop("TT_BIO_FORCE_GRID", None)
    if grid is not None:
        env["TT_BIO_FORCE_GRID"] = f"{grid[0]},{grid[1]}"
    cmd = [sys.executable, str(DRIVER), "--arm", arm, "--size", size, "--block", "0",
           "--folds", "1", "--card", card, "--out", str(out)]
    t0 = time.time()
    log = scratch / f"{arm}_{label}.log"
    with open(log, "wb") as fh:
        rc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    row = {"arm": arm, "grid": label, "forced": grid, "rc": rc,
           "wall_s_NOT_A_MEASUREMENT": round(time.time() - t0, 1)}
    if out.exists():
        d = json.loads(out.read_text())
        row["grid_seen"] = d.get("env", {}).get("grid")
        row["git_head"] = d.get("env", {}).get("git_head")
        row["module_flag"] = d.get("env", {}).get("module_flag")
        row["warmup_sha256"] = (d.get("warmup") or {}).get("cif_sha256")
        row["fold_sha256"] = [f.get("cif_sha256") for f in d.get("folds", [])]
        row["served_declined"] = [f.get("apb_served_declined") for f in d.get("folds", [])]
        row["plddt"] = [f.get("plddt") for f in d.get("folds", [])]
    print(f"  {arm:4s} {label:>7s} rc={rc} digest={str(row.get('fold_sha256'))[:24]} "
          f"counter={row.get('served_declined')}", flush=True)
    return row


def verdict(rows: list) -> dict:
    def digests(arm):
        s = set()
        for r in rows:
            if r["arm"] == arm and r["rc"] == 0:
                s.update([r["warmup_sha256"]] + list(r["fold_sha256"] or []))
        return {d for d in s if d}
    b, o = digests("base"), digests("on")
    failed = [f"{r['arm']}:{r['grid']}" for r in rows if r["rc"] != 0]
    if failed:
        # A refused leg is not a result. This campaign has three times read a non-zero exit as
        # an outcome (a contended leg as an accuracy failure, a grepped banner as the imported
        # tree, an argparse refusal as an arm), so the scorer refuses rather than scoring what
        # survived.
        return {"verdict": "REFUSED", "why": f"legs did not run: {', '.join(failed)}",
                "base_digests": sorted(b), "on_digests": sorted(o), "arms_differ": None}
    if len(b) != 1:
        v, why = "C_MAIN_GRID_DEPENDENT", ("the base arm itself does not agree across grids -- "
                                           "shipping main is grid-dependent here and this test "
                                           "says nothing about the flag")
    elif len(o) == 1:
        v, why = "A_GRID_INDEPENDENT", ("one digest per arm across every grid; the flag changes "
                                        "the answer but not with the grid")
    else:
        v, why = "B_FLAG_GRID_DEPENDENT", ("the flag's output moves with the core grid -- hard "
                                           "stop as a shipped default, same class as region T")
    return {"verdict": v, "why": why,
            "base_digests": sorted(b), "on_digests": sorted(o),
            "arms_differ": bool(b and o and b != o)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="2")
    ap.add_argument("--size", default="298")
    ap.add_argument("--grids", default="native,8x8,9x9")
    ap.add_argument("--out", default=str(REPO / "perf" / "c14_land" / "apb_grid_independence.json"))
    a = ap.parse_args()
    grids = [None if g == "native" else tuple(int(v) for v in g.split("x"))
             for g in a.grids.split(",")]
    scratch = REPO / "perf" / "c14_land" / "apb_grid_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    res = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": os.uname().nodename, "card": a.card, "size_aa": a.size,
           "note": __doc__.strip(), "legs": []}
    # Arms interleaved by grid so a machine-state change cannot line up with one arm. Irrelevant
    # for a digest, kept because the next reader should not have to ask.
    for g in grids:
        for arm in ("base", "on"):
            res["legs"].append(leg(arm, g, a.card, a.size, scratch))
            Path(a.out).write_text(json.dumps(res, indent=1))
    res.update(verdict(res["legs"]))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\nVERDICT {res['verdict']}: {res['why']}")
    print(f"  base digests {len(res['base_digests'])}  on digests {len(res['on_digests'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
