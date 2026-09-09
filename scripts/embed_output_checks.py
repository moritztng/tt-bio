#!/usr/bin/env python3
"""Assert on what `tt-bio embed` / `tt-bio saprot` actually wrote.

Reads the npz/parquet/manifest artifacts from runs already on disk and checks the
claims the CLI help makes: batch size does not move a per-sequence embedding, the
three pool modes are three different vectors and each is the function it names, the
per-residue rows align 1:1 with the sequence, and the manifest describes the files
that exist. Usage: embed_output_checks.py <run_dir_root>
"""
import json
import sys
from pathlib import Path

import numpy as np

FAILS = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(msg)
        print("  FAIL  " + msg)
    else:
        print("  ok    " + msg)


def load(d):
    return {p.stem: np.load(p) for p in sorted(Path(d).glob("*.npz"))}


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch")
    lens = {"L37": 37, "L100": 100, "L129": 129}

    b1, b8 = load(root / "emb_b1"), load(root / "emb_b8")
    check(set(b1) == set(lens), "batch-1 run wrote one npz per sequence: %s" % sorted(b1))
    for sid in lens:
        a, b = b1[sid]["per_residue"], b8[sid]["per_residue"]
        check(a.shape == (lens[sid], 1152),
              "%s per_residue is [L,d] with L=%d (got %s)" % (sid, lens[sid], a.shape))
        same = np.array_equal(a, b)
        mx = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
        check(same, "%s batch 1 vs 8 bit-identical (maxabs %.3e)" % (sid, mx))

    pools = {p: load(root / ("emb_" + p)) for p in ("mean", "max", "cls")}
    for sid in lens:
        pr = pools["mean"][sid]["per_residue"]
        check(np.allclose(pools["mean"][sid]["pooled"], pr.mean(axis=0), atol=1e-5),
              "%s pool=mean equals the mean over per_residue rows" % sid)
        check(np.allclose(pools["max"][sid]["pooled"], pr.max(axis=0), atol=1e-5),
              "%s pool=max equals the max over per_residue rows" % sid)
        cls = pools["cls"][sid]["pooled"]
        check(not np.allclose(cls, pr.mean(axis=0), atol=1e-3),
              "%s pool=cls is a different vector from the mean (a <cls> token, not a re-pool)" % sid)
        # Negative control for the pooler checks: max must not equal mean.
        check(not np.allclose(pr.max(axis=0), pr.mean(axis=0), atol=1e-3),
              "%s NEG max != mean, so the pooler checks can fail" % sid)

    lg = load(root / "emb_logits")
    for sid in lens:
        check("logits" in lg[sid].files and lg[sid]["logits"].shape == (lens[sid], 64),
              "%s --logits writes [L,64] (got %s)"
              % (sid, lg[sid]["logits"].shape if "logits" in lg[sid].files else "absent"))
        check("logits" not in b8[sid].files, "%s no logits array without --logits" % sid)

    for name in ("emb_b8", "emb_logits", "emb_pq"):
        m = json.loads((root / name / "manifest.json").read_text())
        want_logits = name == "emb_logits"
        check(m["logits"] is want_logits, "%s manifest logits flag is %s" % (name, want_logits))
        check((m["shapes"]["logits"] is not None) == want_logits,
              "%s manifest shapes.logits present iff logits were written" % name)
        for e in m["sequences"]:
            check((root / name / e["file"]).exists(),
                  "%s manifest names a file that exists: %s" % (name, e["file"]))

    import pandas as pd
    df = pd.read_parquet(root / "emb_pq" / "embeddings.parquet")
    check(len(df) == 3 and len(df["pooled"].iloc[0]) == 1152,
          "parquet has 3 rows of d=1152 pooled vectors (got %d rows)" % len(df))
    ids = list(df["id"]) if "id" in df.columns else []
    check(ids == sorted(lens, key=lambda k: list(lens).index(k)) or set(ids) == set(lens),
          "parquet ids match the input ids: %s" % ids)

    print("\n%d checks, %d failed" % (CHECKS, len(FAILS)))
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
