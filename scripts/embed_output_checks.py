#!/usr/bin/env python3
"""Assert on what `tt-bio embed` / `tt-bio saprot` actually wrote.

Reads the npz/parquet/manifest artifacts from runs already on disk and checks the
claims the CLI help makes: a sequence batched alone reproduces --batch_size 1 exactly,
batching it with a longer sequence moves it only by the bf16 reduction order the help
now quotes, the three pool modes are three different vectors and each is the function
it names, the per-residue rows align 1:1 with the sequence, and the manifest describes
the files that exist. Usage: embed_output_checks.py <run_dir_root>
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
        x, y = a.astype(np.float64), b.astype(np.float64)
        mx = float(np.abs(x - y).max())
        pcc = float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
        # Batching changes the bucketed length and with it the bf16 reduction order. The
        # help quotes 3.1e-2 / PCC 0.9987 at L=37, the worst of these three; anything
        # past that is a different effect and worth looking at.
        check(mx <= 4e-2 and pcc >= 0.998,
              "%s batch 1 vs 8 within the quoted bf16 band (maxabs %.3e, PCC %.6f)"
              % (sid, mx, pcc))

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

    # The claim the help does make exactly: padding cannot reach a sequence's own rows,
    # so a batch of one reproduces --batch_size 1 bit for bit.
    alone = load(root / "pad_out_alone")
    check(np.array_equal(alone["L37"]["per_residue"], b1["L37"]["per_residue"]),
          "L37 batched alone is bit-identical to --batch_size 1")
    for tag in ("200", "600"):
        x = load(root / ("pad_out_" + tag))["L37"]["per_residue"].astype(np.float64)
        r = alone["L37"]["per_residue"].astype(np.float64)
        check(np.abs(x - r).max() <= 4e-2,
              "L37 with a %s-mer partner stays in the band (maxabs %.3e)"
              % (tag, np.abs(x - r).max()))

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
