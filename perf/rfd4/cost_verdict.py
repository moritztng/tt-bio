"""Turn measured oracle folds into the one number the orchestrator needs: is beam search
oracle-bound, or RFD4-bound?

`state/rfd4/BEAM.md` section 4 derives the split in closed form. For the shipped
`inference_beam_example` schedule a final design costs 600 RFD4 trunk passes and 8 oracle folds,
so the oracle costs more than RFD4 itself exactly when

    t_oracle > (600 / 8) * t_trunk = 75 * t_trunk

and the same 75 falls out of the class defaults (1200 passes, 16 folds). RFD4's per-trunk-pass
cost on Blackhole is `rfd4-trunk`'s to measure and is not known here, so this does NOT pretend to
decide the split. It inverts it instead: a measured `t_oracle` fixes the trunk-pass time at which
the two sides are equal, which is a threshold `rfd4-trunk` can check its own number against the
moment it has one.

    python3 perf/rfd4/cost_verdict.py            # reads perf/rfd4/results/, writes cost_verdict.json

Rows that carry no clock are dropped, loudly. On Blackhole the AICLK sets the fold time, so an
unclocked row is not a slow measurement, it is not a measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "perf" / "rfd4" / "results"

#: BEAM.md section 4: trunk passes and oracle folds per FINAL design, beam width cancels.
RECIPES = {
    "inference_beam_example (B=3, L=2)": {"passes": 600, "folds": 8},
    "class defaults (B=4, L=4)": {"passes": 1200, "folds": 16},
    "ppi_novelty_of3_search (B=2, L=2)": {"passes": 360, "folds": 2},
}
#: Below this the card was throttled and the row is an artifact, not a slower model.
CLOCK_FLOOR = 1200


def usable(row: dict) -> tuple[bool, str]:
    """A row is usable only if it ran on the granted card AND carries a clock above the floor."""
    if row.get("clock_dev") not in (0, None):
        return False, f"clock read from board {row['clock_dev']}, not the granted card"
    clocks = row.get("aiclk_median_per_rep")
    if not clocks:
        return False, "no clock sampled -- not a measurement"
    if min(clocks) < CLOCK_FLOOR:
        return False, f"clock floor: min median {min(clocks)} MHz under {CLOCK_FLOOR}"
    if row.get("runtime_s_median") is None:
        return False, "no runtime"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", default=str(RESULTS / "oracle_cost.jsonl"))
    ap.add_argument("--out", default=str(RESULTS / "cost_verdict.json"))
    args = ap.parse_args()

    kept, dropped = [], []
    path = Path(args.rows)
    for line in path.read_text().splitlines() if path.exists() else []:
        if not line.strip():
            continue
        row = json.loads(line)
        ok, why = usable(row)
        tag = f"{row.get('model')} {row.get('rung_tokens')} tok {row.get('protocol')}"
        (kept if ok else dropped).append(row if ok else {"row": tag, "reason": why})

    folds = {}
    for row in kept:
        folds.setdefault(row["model"], {}).setdefault(row["protocol"], {})[row["rung_tokens"]] = {
            "fold_s": row["runtime_s_median"],
            "aiclk_median_per_rep": row["aiclk_median_per_rep"],
            "board": row.get("board"), "card": row.get("card"), "host": row.get("host"),
            "settings": row.get("settings"),
        }

    # The inversion: what must one RFD4 trunk pass cost for the two sides to be equal?
    thresholds = {}
    for model, protocols in folds.items():
        oracle = protocols.get("oracle") or {}
        for rung, cell in sorted(oracle.items()):
            for name, r in RECIPES.items():
                ratio = r["passes"] / r["folds"]
                thresholds.setdefault(f"{model}@{rung}", {})[name] = {
                    "oracle_fold_s": cell["fold_s"],
                    "passes_per_fold": ratio,
                    "trunk_pass_s_for_a_50_50_split": round(cell["fold_s"] / ratio, 5),
                    "reading": (f"beam search is ORACLE-bound if one RFD4 trunk pass is faster "
                                f"than {cell['fold_s'] / ratio:.4f} s at {rung} tokens, "
                                f"RFD4-bound if slower"),
                }

    out = {
        "source": str(path),
        "kept_rows": len(kept), "dropped_rows": dropped,
        "folds": folds,
        "breakeven": thresholds,
        "note": ("t_trunk is rfd4-trunk's to measure. This fixes the threshold it has to be "
                 "compared against; it does not decide the split on its own."),
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

    for d in dropped:
        print(f"DROPPED {d['row']}: {d['reason']}")
    for key, recipes in sorted(thresholds.items()):
        r = recipes["inference_beam_example (B=3, L=2)"]
        print(f"{key:>18}: oracle fold {r['oracle_fold_s']} s -> "
              f"50/50 at t_trunk = {r['trunk_pass_s_for_a_50_50_split']} s")
    print(f"\n{len(kept)} usable rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
