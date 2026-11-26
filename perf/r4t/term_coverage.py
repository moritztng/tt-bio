#!/usr/bin/env python3
"""Which RFD4 loss terms can upstream's test_dataset actually exercise?

Reads the committed fixture (`tt_bio/train/rfd4_data.py`, upstream's own crops) and counts, per
example, the features that make each optimization term non-zero. No model, no device, seconds to
run. A term that no example drives is a term the G3 parity bar grades trivially (both sides zero),
so the graded set has to be chosen rather than taken in dataset order.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

#: Built by `perf/r4t/build_fixture.py` from upstream's own pipeline; read with numpy alone so
#: this runs on any branch, with or without `tt_bio.train.rfd4_data`.
FIXTURE = Path(__file__).resolve().parents[2] / "tt_bio/train/fixtures/rfd4_train_16.npz"


def counts(batch, k):
    """Per example: tokens, and the driver count behind each term that can be zero."""
    rows = []
    n_ex = batch["padding_mask"].shape[0] // k
    for e in range(n_ex):
        r = e * k
        rows.append({
            "example_id": str(batch["example_ids"][e]),
            "tokens": int(batch["padding_mask"][r].sum()),
            "insertion_prediction": int((batch["insertion_source_idx"][r] >= 0).sum()),
            "signed_volume": int((batch["tetrahedral_vertices"][r] >= 0).all(-1).sum()),
            "sequence_prediction": int((batch["condition_sequence_1_residue"][r] > 0).sum()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixture", type=Path, default=FIXTURE)
    a = ap.parse_args()
    if not a.fixture.exists():
        raise SystemExit(f"{a.fixture} is missing; build it with perf/r4t/build_fixture.py "
                         f"(it lives on wk/r4t-objective) or pass --fixture")
    with np.load(a.fixture, allow_pickle=False) as z:
        b = {k: z[k] for k in z.files}
    rows = counts(b, int(b["exploration_k"]))
    terms = ("insertion_prediction", "signed_volume", "sequence_prediction")
    print(f"{'example':<24}{'tok':>5}" + "".join(f"{t:>22}" for t in terms))
    for row in rows:
        print(f"{row['example_id']:<24}{row['tokens']:>5}" + "".join(f"{row[t]:>22}" for t in terms))
    print()
    for t in terms:
        hit = [r["example_id"] for r in rows if r[t]]
        print(f"{t}: {len(hit)} of {len(rows)} examples" + (f" -> {', '.join(hit)}" if len(hit) <= 3 else ""))
    print("\nbb_ca and local_latents are dense: every example drives them.")


if __name__ == "__main__":
    main()
