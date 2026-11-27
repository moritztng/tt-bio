#!/usr/bin/env python3
"""Which RFD4 loss terms can upstream's test_dataset actually exercise?

Counts, per example, the tokens upstream's own masks supervise for each optimization term that
can come out zero. No model and no device: it reads the committed fixture (upstream's own crops)
with numpy and reimplements the three masks from `flow_matching/product_space_flow_matcher.py`
and `model_masks.py`.

A term no example drives is one the G3 parity bar grades trivially, zero on both sides, while
reporting a pass. The bar is right to treat agreement at zero as agreement; the graded SET is
what has to be chosen, so this names the examples that drive each term.

Masks, in upstream's own notation (P padding, C is_in_context_conditioning, L
supports_local_latent, R is_resolved_mask):
  sequence_prediction  P & ~C & L & R & (condition_sequence_1_residue == <M>), <M> = 0.
      Supervision is on tokens that are NOT sequence-conditioned, so a fully conditioned
      example scores zero however much conditioning it carries.
  insertion_prediction insertion_source_idx >= 0 (the unindexed tokens), zero if none.
  signed_volume        tetrahedral_vertices >= 0 on all four vertices (the chiral centres).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

#: Built by `perf/r4t/build_fixture.py` from upstream's pipeline; read with numpy alone so this
#: runs on any branch, with or without `tt_bio.train.rfd4_data`.
FIXTURE = Path(__file__).resolve().parents[2] / "tt_bio/train/fixtures/rfd4_train_16.npz"

#: `UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"]`, the unconditioned sequence token.
MASK_TOKEN = 0

TERMS = ("sequence_prediction", "insertion_prediction", "signed_volume")


def drivers(batch: dict, k: int) -> list[dict]:
    """Per example: tokens, and how many tokens each zero-able term is supervised on."""
    rows = []
    for e in range(batch["padding_mask"].shape[0] // k):
        r = e * k
        p = batch["padding_mask"][r].astype(bool)
        sup = p & ~batch["is_in_context_conditioning"][r].astype(bool) \
                & batch["supports_local_latent"][r].astype(bool) \
                & batch["is_resolved_mask"][r].astype(bool)
        unconditioned = batch["condition_sequence_1_residue"][r] == MASK_TOKEN
        rows.append({
            "example_id": str(batch["example_ids"][e]),
            "tokens": int(p.sum()),
            "sequence_prediction": int((sup & unconditioned).sum()),
            "insertion_prediction": int((batch["insertion_source_idx"][r] >= 0).sum()),
            "signed_volume": int((batch["tetrahedral_vertices"][r] >= 0).all(-1).sum()),
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
        batch = {key: z[key] for key in z.files}
    rows = drivers(batch, int(batch["exploration_k"]))

    print(f"{'example':<24}{'tok':>5}" + "".join(f"{t:>21}" for t in TERMS))
    for row in rows:
        print(f"{row['example_id']:<24}{row['tokens']:>5}" + "".join(f"{row[t]:>21}" for t in TERMS))
    print()
    for t in TERMS:
        hit = [r["example_id"] for r in rows if r[t]]
        tail = f" -> {', '.join(hit)}" if 0 < len(hit) <= 3 else ""
        print(f"{t}: {len(hit)} of {len(rows)} examples{tail}")
    print("\nbb_ca and local_latents are dense: every example drives them.")


if __name__ == "__main__":
    main()
