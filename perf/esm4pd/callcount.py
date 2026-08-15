#!/usr/bin/env python3
"""P-B: how many TriangleMultiplication calls one real 512 aa fold makes, counted not assumed.

The floor arithmetic is a per-call headroom times a call count, and the state doc uses two
different counts in two places (1084 in §2, 1076 in §4.1). A 0.7 % disagreement is small, but the
count multiplies every term in the floor, so it gets counted.

It records the shape of every call, not just the total, because a call on the 512 aa pair track and
a call inside a batched confidence head do not cost the same and must not be multiplied by the same
per-call wall.

Same fold boundary and same fixture as `fold_ab4.py`, so the wall it reports is comparable to the
36.343 s baseline. The counter wraps `__call__` and adds two dict operations per call, which is
host work outside any timed device region.
"""
import argparse, collections, json, os, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

COUNTS = collections.Counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmfold2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    assert Path(T.__file__).resolve().is_relative_to(ROOT), "tt_bio from %s" % T.__file__

    orig = T.TriangleMultiplication.__call__

    def counted(self, x, mask=None):
        COUNTS["%s|%s" % ("end" if self.ending else "start",
                          "x".join(str(int(d)) for d in x.shape))] += 1
        return orig(self, x, mask)

    T.TriangleMultiplication.__call__ = counted

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    one_fold, meta, _ = B.build_fold(
        a.model, ROOT / (".msa_ab512_%d" % a.size),
        a.fixdir / ("cdk2x2_%d.yaml" % a.size), a.fixdir / ("cdk2x2_%d.a3m" % a.size))
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()

    COUNTS.clear()
    fold_s, m = one_fold()          # warm fold, discarded
    warm_total = sum(COUNTS.values())
    walls = []
    per_fold = []
    for _ in range(a.rounds):
        COUNTS.clear()
        s, m = one_fold()
        walls.append(s)
        per_fold.append(dict(COUNTS))
        print("  fold %8.3f s  trimul calls %d  plddt %s"
              % (s, sum(COUNTS.values()), m.get("plddt")), flush=True)

    total = sum(COUNTS.values())
    R = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": [g.x, g.y], "model": a.model, "size": a.size,
         "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
         "warm_fold_s": round(fold_s, 3), "warm_trimul_calls": warm_total,
         "fold_s": [round(v, 3) for v in walls],
         "fold_s_median": round(st.median(walls), 3), "plddt": m.get("plddt"),
         "total_trimul_calls": total, "by_shape": dict(COUNTS),
         "per_fold_counts_equal": all(c == per_fold[0] for c in per_fold)}
    a.out.write_text(json.dumps(R, indent=1))
    print("\nTOTAL TriangleMultiplication calls per fold: %d" % total)
    for k, v in sorted(COUNTS.items(), key=lambda kv: -kv[1]):
        print("  %-40s %d" % (k, v))
    print("wrote " + str(a.out))


if __name__ == "__main__":
    main()
