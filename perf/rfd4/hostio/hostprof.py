"""Card-free profile of RFD4's per-batch host work: spec build + featurize, and one design's write_cif.

    python perf/rfd4/hostio/hostprof.py --spec pdl1-512.yaml --out DIR [--reps 5] [--prof]

Writes DIR/<phase>.prof (cProfile) and prints wall per rep as JSON lines. The design written is
the stand-in engine output of tests/test_rfd4_write_designs.py, real shapes.
"""
import argparse
import cProfile
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--prof", action="store_true")
    a = p.parse_args()
    from test_rfd4_write_designs import _stand_in
    from tt_bio import rfd4

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    def timed(name, fn):
        prof = cProfile.Profile() if a.prof else None
        ts = []
        for r in range(a.reps):
            t0 = time.perf_counter()
            if prof and r > 0:
                prof.enable()
            v = fn(r)
            if prof and r > 0:
                prof.disable()
            ts.append(round(time.perf_counter() - t0, 4))
        if prof:
            prof.dump_stats(out / f"{name}.prof")
        print(json.dumps({"phase": name, "spec": Path(a.spec).name, "wall_s": ts}), flush=True)
        return v

    def feat(r):
        ex = rfd4.prepare_examples(a.spec, out / f"in{r}")[0]
        return rfd4.featurize(ex.cif, seed=1234)

    features, atom_array = timed("featurize", feat)
    output = _stand_in(features, 1)
    timed("write", lambda r: rfd4.write_design(output, features, atom_array, out / f"w{r}.cif", row=0))
    print(json.dumps({"tokens": int(features["padding_mask"].shape[1])}))


if __name__ == "__main__":
    main()
