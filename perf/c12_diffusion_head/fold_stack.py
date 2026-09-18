#!/usr/bin/env python3
"""The fold-level arm set for this row's TWO levers, with the STACK checked in its own right.

C12 closed 2026-09-18 with VERDICT: STOP, and one of its two surviving lessons binds this row
directly: **two individually-correct levers can compose into a silently wrong transform that
neither owning row can detect.** `TT_BIO_DIT_COND_HOIST` and `TT_BIO_FUSE_COND_MULADD` each passed
their own float64 panel alone; combined, 512 aa plDDT collapsed 0.864 -> 0.368, and the combined arm
was the FASTEST, so timing alone would have selected the broken one.

This row ships two flags -- `TT_BIO_APB_HEAD_MAJOR_QKV` (token transformer + trunk) and
`TT_BIO_APB_ATOM_HEAD_MAJOR_QKV` (atom blocks). They touch disjoint sites and each is a tile
re-point that is bit-exact against its own A1 arm at the op level, so the composition risk is
lower than the cond-hoist case. "Lower" is not "checked", which is exactly the lesson. So four
configurations, interleaved rep by rep in one session, and the stack's accuracy is read against
`off` DIRECTLY rather than inferred from the two singles:

    off     both flags 0     the baseline, and the A/A floor comes from its own repeats
    token   APB only
    atom    ATOM only
    both    the stack        <- checked on its own, never as token + atom

Per arm: fold seconds, the CIF sha256, plDDT, and each lever's served/declined counters, because a
lever that silently stops firing reads as a clean no-op otherwise. The accuracy verdict quotes the
all-atom Kabsch deviation in Angstrom against the 512 aa bar of 0.60 A with the seed-scatter floor
of 1.84 A beside it -- bit-exactness is not the bar here, accuracy is.

`--selftest` exercises everything that needs no device: arm ordering, the digest and plDDT
extraction (against CIFs already committed under `perf/other512/cif/`), and the verdict arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "perf" / "other512"))

FIXTURE = ROOT / "perf" / "size512" / "fixtures" / "cdk2x2_512.yaml"
BAR_A = 0.60           # the 512 aa structural kill bar
SEED_FLOOR_A = 1.84    # re-running with a different seed moves the structure this far

# (arm, APB flag, ATOM flag)
ARMS = [
    ("off", "0", "0"),
    ("token", "1", "0"),
    ("atom", "0", "1"),
    ("both", "1", "1"),
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _cif_in(d: Path) -> Path | None:
    cifs = sorted(d.rglob("*.cif"))
    return cifs[0] if cifs else None


def read_structure(d: Path):
    """The digest, plDDT and atom table of the one CIF this arm wrote."""
    from cif_rmsd import bfactor_plddt, read_atoms
    cif = _cif_in(d)
    if cif is None:
        return {"cif": None, "sha256": None, "plddt": None, "n_atoms": None}, None
    col = bfactor_plddt(cif)
    plddt = None if col is None or col.get("mean_ca") is None else round(col["mean_ca"] * 100.0, 4)
    keys, xyz = read_atoms(cif)
    return ({"cif": cif.name, "sha256": _sha(cif), "plddt": plddt, "n_atoms": len(keys)},
            (keys, xyz))


def deviation_a(ref, arm):
    """All-atom Kabsch RMSD in Angstrom, or None when the two are not comparable."""
    from cif_rmsd import kabsch_rmsd
    if ref is None or arm is None:
        return None
    (rk, rx), (ak, ax) = ref, arm
    if rk != ak:
        return None                      # different atom sets: not an RMSD, a different structure
    return round(kabsch_rmsd(rx, ax), 6)


def verdict(rows, aa_floor_s):
    """The stack's own accuracy and timing verdict, plus what each single contributes."""
    by = {r["arm"]: r for r in rows}
    out = {"aa_floor_s": aa_floor_s, "bar_A": BAR_A, "seed_floor_A": SEED_FLOOR_A}
    base = by.get("off", {}).get("fold_s")
    for arm in ("token", "atom", "both"):
        r = by.get(arm)
        if not r or base is None or not r.get("fold_s"):
            continue
        gain = round(base - r["fold_s"], 4)
        out[arm] = {
            "fold_s": r["fold_s"], "gain_s": gain,
            "over_aa_floor": None if not aa_floor_s else round(abs(gain) / aa_floor_s, 2),
            "deviation_A": r.get("deviation_A"),
            "bit_identical": r.get("sha256") is not None and r["sha256"] == by["off"].get("sha256"),
            "plddt": r.get("plddt"),
        }
    b = out.get("both")
    if b:
        singles = [out.get("token", {}).get("gain_s"), out.get("atom", {}).get("gain_s")]
        if all(s is not None for s in singles):
            b["sum_of_singles_s"] = round(sum(singles), 4)
            b["stack_vs_sum"] = round(b["gain_s"] - sum(singles), 4)
        dev = b.get("deviation_A")
        b["accuracy_verdict"] = (
            "not comparable -- different atom sets" if dev is None else
            f"{dev:.4f} A against a {BAR_A} A bar, seed floor {SEED_FLOOR_A} A: "
            + ("CLEARS" if dev <= BAR_A else "FAILS"))
        # The lesson: the stack is judged on its OWN reading, never on the two singles.
        b["judged_on"] = "the both-arm's own fold and its own CIF, not token + atom"
    return out


def counters(env_json: Path):
    """The lever counters a fold wrote, so a dark lever cannot read as a clean no-op."""
    if not env_json.is_file():
        return None
    try:
        return json.loads(env_json.read_text())
    except ValueError:
        return None


def fold_argv(arm, rep, d, py):
    """The fold, wrapped in the canonical lever census.

    Through `scripts/lever_census.py` rather than bare, for two reasons: it is the instrument the
    release gate itself uses (`--tt-bio <python> ... -- -m tt_bio.main predict ...`), and it is the
    only thing that records each lever's served/declined counts. A lever that silently stops firing
    otherwise reads as a clean no-op, which is how this campaign lost track of dark levers twice.
    """
    return [py, str(ROOT / "scripts" / "lever_census.py"), "--tt-bio", py,
            "--label", f"{arm}_{rep}", "--out", str(d / "levers.json"),
            "--", "-m", "tt_bio.main", "predict", str(FIXTURE), "--model", "boltz2",
            "--single_sequence", "--sampling_steps", "200", "--diffusion_samples", "1",
            "--seed", "0", "--out_dir", str(d)]


def run_arm(arm, apb, atom, rep, outdir, py, reps_dir):
    """One fold. Returns the row; never raises, so one bad arm cannot void the session."""
    d = reps_dir / f"512_{arm}_{rep}"
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ,
               TT_BIO_APB_HEAD_MAJOR_QKV=apb,
               TT_BIO_APB_ATOM_HEAD_MAJOR_QKV=atom)
    cmd = fold_argv(arm, rep, d, py)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    wall = round(time.perf_counter() - t0, 4)
    (d / "fold.log").write_text(p.stdout[-40000:] + "\n--- stderr ---\n" + p.stderr[-40000:])
    meta, atoms = read_structure(d)
    row = {"arm": arm, "rep": rep, "APB": apb, "ATOM": atom, "rc": p.returncode,
           "wall_s": wall, "fold_s": wall if p.returncode == 0 else None,
           "levers": counters(d / "levers.json"), **meta}
    return row, atoms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--py", default=sys.executable)
    ap.add_argument("--out", default="fold_stack.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    assert FIXTURE.is_file(), f"fixture missing: {FIXTURE}"
    reps_dir = OUT / "fold_arms"
    rows, atoms = [], {}
    # Interleaved rep by rep, cold rep discarded per arm: a first-call JIT once put +2.3461 s
    # inside a timed arm, 4.7x the effect this row is looking for.
    for rep in range(a.reps + 1):
        for arm, apb, at in ARMS:
            r, ax = run_arm(arm, apb, at, rep, OUT, a.py, reps_dir)
            print(f"rep {rep} {arm:<6} rc={r['rc']} wall={r['wall_s']}s sha={r['sha256']} "
                  f"plddt={r['plddt']}")
            if rep:
                rows.append(r)
                atoms.setdefault(arm, ax)
    med = {}
    for arm, _apb, _at in ARMS:
        vals = sorted(x["fold_s"] for x in rows if x["arm"] == arm and x["fold_s"])
        med[arm] = vals[len(vals) // 2] if vals else None
    offs = sorted(x["fold_s"] for x in rows if x["arm"] == "off" and x["fold_s"])
    aa_floor = round(offs[-1] - offs[0], 4) if len(offs) > 1 else None
    summary = [{"arm": arm, "fold_s": med[arm],
                "sha256": next((x["sha256"] for x in rows if x["arm"] == arm), None),
                "plddt": next((x["plddt"] for x in rows if x["arm"] == arm), None),
                "deviation_A": deviation_a(atoms.get("off"), atoms.get(arm)),
                "levers": next((x["levers"] for x in rows if x["arm"] == arm), None)}
               for arm, _a, _b in ARMS]
    rec = {"fixture": FIXTURE.name, "reps": a.reps, "arms": summary, "runs": rows,
           "verdict": verdict(summary, aa_floor)}
    (OUT / a.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec["verdict"], indent=1))
    print(f"\n{OUT / a.out}")
    return 0 if all(x["fold_s"] for x in summary) else 1


def selftest() -> int:
    """Everything that needs no device: extraction, ordering, and the verdict arithmetic."""
    ok = True

    def check(label, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(("PASS " if good else "FAIL ") + label + (f": {got!r} != {want!r}" if not good else ""))

    # ordering: four arms per rep, cold rep first, `both` always present and last
    order = [arm for _rep in range(3) for arm, _a, _b in ARMS]
    check("four arms per rep", len(order) // 3, 4)
    check("stack arm is last in each rep", order[3], "both")
    check("off arm is first in each rep", order[0], "off")

    # extraction against CIFs already committed by an earlier fold A/B
    ref_dir = ROOT / "perf" / "other512" / "cif" / "512_on_1"
    if not ref_dir.is_dir():
        print("SKIP extraction: no committed 512_on_1 CIF")
    else:
        meta, atoms = read_structure(ref_dir)
        check("digest is 16 hex chars", len(meta["sha256"] or ""), 16)
        check("atoms parsed", bool(meta["n_atoms"] and meta["n_atoms"] > 100), True)
        check("plddt in range", bool(meta["plddt"] and 0 < meta["plddt"] <= 100), True)
        check("self deviation is exactly zero", deviation_a(atoms, atoms), 0.0)

    # the verdict: a stack whose reading is worse than both singles must NOT be excused by them
    rows = [{"arm": "off", "fold_s": 14.90, "sha256": "aaaa", "plddt": 86.4, "deviation_A": 0.0},
            {"arm": "token", "fold_s": 14.70, "sha256": "bbbb", "plddt": 86.3, "deviation_A": 0.21},
            {"arm": "atom", "fold_s": 14.80, "sha256": "cccc", "plddt": 86.4, "deviation_A": 0.11},
            {"arm": "both", "fold_s": 14.55, "sha256": "dddd", "plddt": 36.8, "deviation_A": 3.10}]
    v = verdict(rows, aa_floor_s=0.05)
    check("stack gain measured on its own", v["both"]["gain_s"], 0.35)
    check("sum of singles recorded separately", v["both"]["sum_of_singles_s"], 0.30)
    check("stack vs sum recorded", v["both"]["stack_vs_sum"], 0.05)
    check("stack FAILS the bar on its own reading", "FAILS" in v["both"]["accuracy_verdict"], True)
    check("singles cleared it individually",
          all(v[s]["deviation_A"] <= BAR_A for s in ("token", "atom")), True)
    check("over its own A/A floor", v["both"]["over_aa_floor"], 7.0)
    # and the honest case
    rows[3] = {"arm": "both", "fold_s": 14.55, "sha256": "aaaa", "plddt": 86.4,
               "deviation_A": 0.0}
    v2 = verdict(rows, aa_floor_s=0.05)
    check("bit-identical stack is reported as such", v2["both"]["bit_identical"], True)
    check("and CLEARS", "CLEARS" in v2["both"]["accuracy_verdict"], True)
    check("not-comparable is not a pass",
          "not comparable" in verdict(
              rows[:3] + [{"arm": "both", "fold_s": 14.5, "sha256": "x", "plddt": 1,
                           "deviation_A": None}], 0.05)["both"]["accuracy_verdict"], True)
    # the fold argv: the census wrapper, the separator, and the flags NOT smuggled into argv
    argv = fold_argv("both", 1, Path("/tmp/x"), "/usr/bin/python3")
    check("census wrapper is the child", argv[1].endswith("scripts/lever_census.py"), True)
    check("census gets an --out", "--out" in argv, True)
    check("cli is separated by --", argv[argv.index("--") + 1:argv.index("--") + 3],
          ["-m", "tt_bio.main"])
    check("fixture is the 512 aa one", Path(argv[-3]).name if argv[-4] == "--out_dir" else
          FIXTURE.name, FIXTURE.name)
    check("flags are not passed as argv", any(x.startswith("TT_BIO_") for x in argv), False)
    check("sampling steps are the model's own, not reduced",
          argv[argv.index("--sampling_steps") + 1], "200")

    print("\nSELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
