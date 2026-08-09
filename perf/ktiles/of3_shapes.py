#!/usr/bin/env python3
"""Collect the batched ttnn.matmul shapes a real OpenFold3 fold issues, at 298 aa.

E1 and E4 both need real operand shapes: the state doc's site audit says
openfold3_diffusion_transformer.py:193 and tenstorrent.py:269 are the two Nt<=2 AV matmuls
worth applying, and openfold3_atom_transformer.py:153/159 is rank 5 and blocked on whether
ttnn accepts a MatmulMultiCoreReuseProgramConfig there at all. Deriving those shapes by hand
is how a contract test ends up asserting on a shape the fold never runs, so read them off a fold.

Folds through the same scripts/gpu_vs_tt/tt_baseline.py path as perf/ktiles/fold_ab.py, with
ttnn.matmul wrapped for the duration, tallying every call whose operands are both batched.
"""
import argparse, collections, json, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openfold3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import ttnn
    real = ttnn.matmul
    seen = collections.Counter()

    def probe(x, y, **kw):
        try:
            sx, sy = tuple(x.shape), tuple(y.shape)
            if len(sx) >= 4 and len(sy) >= 4:
                bx = 1
                for d in sx[:-2]:
                    bx *= d
                if bx >= 2:
                    seen[(sx, sy, str(x.dtype), str(kw.get("dtype")))] += 1
        except Exception:
            pass
        return real(x, y, **kw)

    import tt_baseline as B
    msa_dir = Path(tempfile.mkdtemp(prefix="of3-shapes-msa-"))
    one_fold, meta, _state = B.build_fold(
        a.model, msa_dir, ROOT / "examples" / "prot300.yaml",
        Path(B.FIXTURES) / "prot300.a3m")
    ttnn.matmul = probe
    try:
        one_fold()
    finally:
        ttnn.matmul = real

    rows = [{"a": list(k[0]), "b": list(k[1]), "dtype": k[2], "out_dtype": k[3], "calls": n}
            for k, n in sorted(seen.items(), key=lambda kv: -kv[1])]
    Path(a.out).write_text(json.dumps(
        {"model": a.model, "target": "examples/prot300.yaml", "rows": rows}, indent=1))
    print(f"\n{'calls':>8}  {'rank':>4}  {'a':<24} {'b':<24} dtype")
    for r in rows:
        print(f"{r['calls']:>8}  {len(r['a']):>4}  {str(r['a']):<24} {str(r['b']):<24} "
              f"{r['dtype']}  out={r['out_dtype']}")
    from tt_bio.tenstorrent import cleanup
    cleanup()


main()
