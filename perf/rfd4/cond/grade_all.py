"""Grade every protein family's card run: what the input held, and how far two seeds land apart.

    python perf/rfd4/cond/grade_all.py --runs ~/rfd4-cond-runs --out <json>

A family is graded against the featurizer's own atom array, NOT against the input CIF re-read
from disk. That is the whole trick: several families ship an input that is not a structure in the
design's numbering -- motif_scaffold ships 3 motif residues numbered in their own frame against an
82-residue design numbered from 0, unconditional ships a length spec with no coordinates at all --
and matching those by residue id produces a number that looks like a motif deviation and is not
one. `rfd4.featurize` returns the array the model actually saw, in the design's numbering, with
the conditioning attached per atom, so one rule covers every family.

What an input structure MEANS differs by family, and that is the second thing this grades on:

* **spec** -- atoms carrying a finite `condition_coordinate_1_atom` or
  `condition_InternalCoordinate_1_atom`. The spec pinned them, in Angstrom, and the design has to
  give them back. This is the number that catches a condition that never reached the model.
* **sampler** -- `--inverse_folding` and `--partial` hold the input through `x_1` rather than
  through any condition, so the input structure is the pin even though nothing is annotated.
* **prediction target** -- an input with coordinates and no pin of either kind is ground truth the
  model is asked to reproduce, not something it was given. Folding is this: its conditioning is
  sequence-only. Grading it against a 1 A "held" bar would be grading a prediction as a copy, so
  it is reported as accuracy per seed and does not gate the verdict.

The floor is two of our own designs at two seeds. Where every residue is held there is no
coordinate floor to read and the floor is the sequence: a family that returns an identical
sequence at both seeds is not designing. A partial restart is its own case -- at `--partial
0.4167` the backbone restarts at t=0.97, 3 percent noised, so demanding half an Angstrom of
movement would fail a setting that is working; it has to differ, and the resolved per-modality t
is reported so the size of the difference can be read against it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grade_family as gf  # noqa: E402
from families import FAMILIES, INPUTS  # noqa: E402

#: A pin this close, over at least this many atoms, was held rather than designed.
HELD_RMSD_A = 1.0
HELD_MIN_ATOMS = 3
#: Two full generations of the same input land at least this far apart, or nothing is being designed.
FLOOR_MIN_A = 0.5


def input_for(family: str, runs: Path) -> Path:
    """The structure this family's design was made from."""
    f = FAMILIES[family]
    if "spec" in f:
        prepared = sorted((runs / family / "inputs").glob("*.cif"))
        if not prepared:
            raise SystemExit(f"{family}: no prepared input under {runs / family / 'inputs'}")
        return prepared[0]
    if "from" in f:
        parent = sorted((runs / f["from"]).glob("*_0_generated.cif"))
        if not parent:
            raise SystemExit(f"{family}: no {f['from']} design to restart from")
        return parent[0]
    return INPUTS / Path(f["input"]).name


def references(arr):
    """Atom arrays carrying whatever coordinates the input fixed, by how it fixed them."""
    out = {}
    have = set(arr.get_annotation_categories())
    for name, key in (("spec_coordinate", "condition_coordinate_1_atom"),
                      ("spec_internal_coordinate", "condition_InternalCoordinate_1_atom")):
        if key not in have:
            continue
        values = np.asarray(arr.get_annotation(key), dtype=float)
        mask = np.isfinite(values).all(-1)
        if mask.sum() >= HELD_MIN_ATOMS:
            ref = arr[mask].copy()
            ref.coord = values[mask].astype(np.float32)
            out[name] = ref
    given = np.isfinite(arr.coord).all(-1) & (np.asarray(arr.occupancy, dtype=float) > 0)
    if given.sum() >= HELD_MIN_ATOMS:
        out["input_structure"] = arr[given].copy()
    return out


def fit_to(design, reference):
    """The design's own atoms at the reference's positions, superposed, in Angstrom."""
    a, b = gf.common(design, reference)
    if len(a) < HELD_MIN_ATOMS:
        return None
    ca = a.atom_name == "CA"
    r, t = gf.kabsch(b.coord, a.coord)
    out = {"fitted_rmsd_a": gf.rmsd(b.coord @ r.T + t, a.coord), "atoms": int(len(a)),
           "residues": int(len(np.unique(a.res_id)))}
    if ca.sum() >= HELD_MIN_ATOMS:
        rc, tc = gf.kabsch(b.coord[ca], a.coord[ca])
        out["fitted_ca_rmsd_a"] = gf.rmsd(b.coord[ca] @ rc.T + tc, a.coord[ca])
    return out


def grade(family, row, runs):
    from tt_bio import rfd4

    source = input_for(family, runs)
    features, arr = rfd4.featurize(source, seed=row["seed"])
    designs = [gf.load(p) for p in row["cifs"]]
    refs = references(arr)

    held = {name: got for name, got in
            ((n, fit_to(designs[0], r)) for n, r in refs.items()) if got}

    # How the input is fixed decides what the number means. A spec pin first; then the sampler,
    # which holds through x_1 with nothing annotated; an unpinned input structure is a target.
    by_sampler = "--inverse_folding" in row["command"] or "--partial" in row["command"]
    key = next((k for k in ("spec_coordinate", "spec_internal_coordinate") if k in held), None)
    kind = "spec"
    if key is None and "input_structure" in held:
        key, kind = "input_structure", "sampler" if by_sampler else "prediction_target"
    elif key is None:
        kind = "nothing"

    # A prediction target is not held: the model placed every one of those residues itself, so
    # they are all designed and the floor is a coordinate floor like any other generation.
    held_residues = set()
    if key and kind in ("spec", "sampler"):
        a, _ = gf.common(designs[0], refs[key])
        held_residues = {f"{c}|{r}" for c, r in zip(a.chain_id, a.res_id)}

    ca = designs[0][designs[0].atom_name == "CA"]
    designed = np.array([f"{c}|{r}" not in held_residues
                         for c, r in zip(ca.chain_id, ca.res_id)])
    floor = gf.pair(designs[0], designs[1],
                    sorted({c for c, h in zip(ca.chain_id, ~designed) if h}))

    g = {
        "family": family,
        "input": str(source),
        "designs": row["cifs"],
        "recipe": row["recipe"],
        "tokens": int(features["padding_mask"].sum()),
        "s_per_design": row["s_per_design"],
        "aiclk_mhz": row["aiclk_mhz"],
        "clock_artifact": row["clock_artifact"],
        "held_by": kind,
        "held_reference": key or "none",
        "held": held,
        "designed_residues": int(designed.sum()),
        "seed_floor_ca_rmsd_a": floor["designed_ca_rmsd_a"],
        "seed_floor_sequence_identity": floor["sequence_identity"],
    }

    if kind == "prediction_target":
        # Not a pin: report how close each seed came, and let rfd4-quality's distributional bar
        # judge it. Two samples cannot separate the model's own spread from a port defect.
        g["accuracy_per_seed"] = [fit_to(d, refs[key]) for d in designs]
    if "--partial" in row["command"]:
        from tt_bio.rfd4 import sampler as s

        spec = s.recycle_strategy(row["recipe"])
        start = s.resolve_partial_start(spec, partial_trajectory_fraction_steps=0.4166666666666667)
        g["partial_start_t"] = {k: round(v, 4) for k, v in start.t.items()}
        g["partial_steps_remaining"] = start.nsteps

    checks = {"clock_real": not row["clock_artifact"]}
    if kind in ("spec", "sampler"):
        checks["held_within_1a"] = held[key]["fitted_rmsd_a"] <= HELD_RMSD_A
    if "--partial" in row["command"]:
        # The restart's own noise level sets the scale, so the only threshold the schedule
        # supports is that the two seeds went somewhere different at all.
        checks["two_seeds_differ_after_restart"] = (floor["designed_ca_rmsd_a"] > 0.0
                                                    and floor["sequence_identity"] < 1.0)
    elif g["designed_residues"] == 0:
        checks["two_seeds_differ_in_sequence"] = floor["sequence_identity"] < 1.0
    else:
        checks["two_seeds_differ"] = floor["designed_ca_rmsd_a"] > FLOOR_MIN_A
    g["checks"] = checks
    g["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = Path(a.runs).expanduser()

    rows = [json.loads(l) for l in (runs / "runs.jsonl").read_text().splitlines() if l.strip()]
    graded = {}
    for row in rows:
        family = row["family"]
        if row["rc"] != 0:
            graded[family] = {"family": family, "verdict": "FAIL", "why": f"run rc {row['rc']}"}
            continue
        g = grade(family, row, runs)
        graded[family] = g
        k = g["held_reference"]
        at = (f"{g['held'][k]['fitted_rmsd_a']:.3f} A / {g['held'][k]['atoms']} atoms"
              if k != "none" else "nothing held")
        print(f"{family:30s} {g['verdict']:4s} {g['held_by']:17s} {at:26s} "
              f"floor {g['seed_floor_ca_rmsd_a']:7.3f} A id {g['seed_floor_sequence_identity']:.3f}")

    Path(a.out).write_text(json.dumps(graded, indent=2) + "\n")
    print(f"\nwrote {a.out}")
    return 0 if all(g["verdict"] == "PASS" for g in graded.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
