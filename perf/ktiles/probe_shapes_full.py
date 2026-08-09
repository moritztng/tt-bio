#!/usr/bin/env python3
"""Log the LOGICAL shapes and the call site of every batched_matmul in one real fold.

probe_shapes.py records only the tile counts the chooser keys on, which hides tile padding: a
logical K of 600 and a logical K of 608 are both Kt=19. The opendde parity bisect needs the
difference, because a padded K is contracted over garbage columns and two factories need not agree
on what those columns hold.
"""
import argparse, json, sys, tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["protenix-v2", "opendde"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_bio.protenix as P

    seen = Counter()
    orig = T.batched_matmul

    def spy(x, y, compute_kernel_config=None):
        f = sys._getframe(1)
        sa, sb = tuple(int(d) for d in x.shape), tuple(int(d) for d in y.shape)
        pa = tuple(int(d) for d in x.padded_shape) if hasattr(x, "padded_shape") else sa
        pb = tuple(int(d) for d in y.padded_shape) if hasattr(y, "padded_shape") else sb
        cfg = None
        if len(sa) == 4 and len(sb) == 4 and x.dtype == y.dtype:
            cfg = T._batched_reuse_config(sa[0] * sa[1], -(-sa[2] // 32), -(-sa[3] // 32),
                                          -(-sb[3] // 32), 4 if x.dtype == T.ttnn.float32 else 2)
        seen[(sa, sb, pa, pb, str(x.dtype), f"{Path(f.f_code.co_filename).name}:{f.f_lineno}",
              cfg is not None, None if cfg is None else cfg.per_core_M)] += 1
        return orig(x, y, compute_kernel_config=compute_kernel_config)

    T.batched_matmul = spy
    P.batched_matmul = spy
    import tt_baseline as B
    msa_dir = Path(tempfile.mkdtemp(prefix="ktiles-probe2-"))
    one_fold, _meta, _state = B.build_fold(
        a.model, msa_dir, ROOT / "examples" / "prot300.yaml", Path(B.FIXTURES) / "prot300.a3m")
    one_fold()
    rows = []
    for k, v in sorted(seen.items(), key=lambda kv: -kv[1]):
        sa, sb, pa, pb, dt, site, applied, pcm = k
        pad = "PADDED" if (sa != pa or sb != pb) else "exact"
        print(f"  {site:24s} {str(sa):20s} x {str(sb):20s} {dt.split('.')[-1]:9s} {pad:6s} "
              f"applied={applied} pcm={pcm} calls={v}")
        if pad == "PADDED":
            print(f"      padded: {pa} x {pb}")
        rows.append(dict(a=list(sa), b=list(sb), a_padded=list(pa), b_padded=list(pb), dtype=dt,
                         site=site, applied=applied, per_core_M=pcm, calls=v, padding=pad))
    a.out.write_text(json.dumps({"model": a.model, "rows": rows}, indent=1))
    from tt_bio.tenstorrent import cleanup
    cleanup()


main()
