#!/usr/bin/env python3
"""Which (batch, Mt, Kt, Nt) keys does batched_matmul actually see in a real 298 aa fold, and
which of them does _batched_reuse_config take? The op-isolated A/B used shapes lifted from the
qb2 diffusion ledger; this checks they are the shapes the fold really runs."""
import argparse, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["protenix-v2", "opendde"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tempfile
    import tt_bio.tenstorrent as T
    import tt_bio.protenix as P

    seen = Counter()
    orig = T._batched_reuse_config

    def spy(batch, mt, kt, nt, eb):
        cfg = orig(batch, mt, kt, nt, eb)
        seen[(batch, mt, kt, nt, eb, cfg is not None)] += 1
        return cfg

    T._batched_reuse_config = spy
    # batched_matmul closes over the module global, so rebinding the name is enough.
    import tt_baseline as B
    msa_dir = Path(tempfile.mkdtemp(prefix="ktiles-probe-"))
    one_fold, _meta, _state = B.build_fold(
        a.model, msa_dir, ROOT / "examples" / "prot300.yaml", Path(B.FIXTURES) / "prot300.a3m")
    one_fold()
    rows = [dict(batch=k[0], Mt=k[1], Kt=k[2], Nt=k[3], elem_bytes=k[4], applied=k[5], calls=v)
            for k, v in sorted(seen.items(), key=lambda kv: -kv[1])]
    for r in rows:
        print(f"  B={r['batch']:5d} Mt={r['Mt']:3d} Kt={r['Kt']:3d} Nt={r['Nt']:3d} "
              f"eb={r['elem_bytes']} applied={r['applied']} calls={r['calls']}")
    a.out.write_text(json.dumps({"model": a.model, "rows": rows}, indent=1))
    from tt_bio.tenstorrent import cleanup
    cleanup()


main()
