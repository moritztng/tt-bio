"""Grade every protein family's card run: what the conditioning held, and the seed floor.

    python perf/rfd4/cond/grade_all.py --runs ~/rfd4-cond-runs --out <json>

Reads `runs.jsonl` the card run wrote, pairs each family's two designs with the input they were
made from, and calls `grade_family` on them. A spec family is graded against the prepared CIF the
CLI itself wrote under `<out_dir>/inputs/`, which is the structure the model actually saw; a
shipped-CIF family against its staged `.cif.zst`. Partial diffusion is graded against the binder
design it restarted from, which is the only sense in which it was conditioned at all.

Only the binder has a full upstream design to pair against, and that one is rfd4-e2e's 128-token
run, not this 225-token spec, so it is cited rather than recomputed here.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grade_family  # noqa: E402
from families import FAMILIES, INPUTS  # noqa: E402


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
        source = input_for(family, runs)
        designs = [grade_family.load(p) for p in row["cifs"]]
        per_chain, held, matched = grade_family.held_chains(designs[0], grade_family.load(source))
        floor = grade_family.pair(designs[0], designs[1], held)
        g = {
            "family": family,
            "input": str(source),
            "designs": row["cifs"],
            "recipe": row["recipe"],
            "s_per_design": row["s_per_design"],
            "aiclk_mhz": row["aiclk_mhz"],
            "clock_artifact": row["clock_artifact"],
            "input_overlap_atoms": matched,
            "held_chains": held,
            "per_chain_fitted_rmsd_to_input_a": per_chain,
            "seed_floor_a": floor["designed_ca_rmsd_a"],
            "seed_floor": floor,
        }
        checks = {
            "input_matched": matched > 0,
            "held_within_1a": (not held) or max(
                per_chain[c]["fitted_rmsd_a"] for c in held) <= grade_family.HELD_RMSD_A,
            "two_seeds_differ": floor["designed_ca_rmsd_a"] > 0.5,
            "clock_real": not row["clock_artifact"],
        }
        g["checks"] = checks
        g["verdict"] = "PASS" if all(checks.values()) else "FAIL"
        graded[family] = g
        print(f"{family:30s} {g['verdict']:4s} held={held} floor={g['seed_floor_a']:.3f} A "
              f"id={floor['sequence_identity']:.3f} {row['s_per_design']:.1f} s/design "
              f"@ {row['aiclk_mhz']} MHz")

    Path(a.out).write_text(json.dumps(graded, indent=2) + "\n")
    print(f"\nwrote {a.out}")
    return 0 if all(g["verdict"] == "PASS" for g in graded.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
