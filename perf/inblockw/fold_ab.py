#!/usr/bin/env python3
"""Fold-level A/B for the tuned `in0_block_w` program config, one arm per process.

`--arm off` rebinds `tt_bio.tenstorrent._linear` to a plain `ttnn.linear`, so the off arm is
byte-for-byte the call the switched sites made before this branch. Production config otherwise
(10 recycles / 200 sampling steps / 1 sample / seed 0), cold fold discarded.

WARROOM section 3: qb2 is ttnn 0.68.0, so this reports a RATIO on this card. No absolute
folds/s from here is a campaign result.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix-v2")
    ap.add_argument("--arm", choices=["on", "off"], required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--target", type=Path, default=ROOT / "examples" / "prot300.yaml")
    ap.add_argument("--a3m", type=Path,
                    default=ROOT / "scripts/gpu_vs_tt/fixtures/prot300.a3m")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B

    fired = {"n": 0}
    if a.arm == "off":
        def plain(x, w, **kw):
            return ttnn.linear(x, w, **kw)
        T._linear = plain
    else:
        tuned = T._linear

        def counting(x, w, **kw):
            if T._tuned_config_for(x, w) is not None:
                fired["n"] += 1
            return tuned(x, w, **kw)
        T._linear = counting

    one_fold, meta, _state = B.build_fold(a.model, ROOT / f".msa_ab", a.target, a.a3m)
    cold_s, cold_m = one_fold()
    assert cold_m.get("msa"), "fold ran without an MSA"
    fired["n"] = 0
    times, plddt = [], None
    for _ in range(a.repeat):
        t, m = one_fold()
        times.append(t)
        plddt = m["plddt"]
    cif = sorted(Path(meta["struct_dir"]).glob("*.cif"))
    sha = hashlib.sha256(cif[0].read_bytes()).hexdigest()[:16] if cif else None

    res = dict(model=a.model, arm=a.arm, target=str(a.target),
               cold_s=round(cold_s, 3), times=[round(t, 3) for t in times],
               median_s=round(statistics.median(times), 3),
               min_s=round(min(times), 3), plddt=plddt,
               n_tokens=cold_m.get("n_tokens"), cif_sha16=sha,
               fired_calls_per_fold=fired["n"] // max(1, a.repeat),
               grid=list(T.COMPUTE_GRID_MAIN))
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
