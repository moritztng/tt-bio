#!/usr/bin/env python3
"""E6 fold A/B: the chunked-Transition config, on vs off, at production settings.

One arm per process. The `off` arm rebinds `_transition_linear` to the plain
`ttnn.linear(core_grid=...)` call the branch replaces, so it is byte-for-byte today's fold.
Cold fold discarded; the CIF and pLDDT of the last warm fold are recorded for both arms.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

WT = Path("/home/ttuser/.coworker/wt/perfwar-chunked-transition-cb")
sys.path.insert(0, str(WT))
sys.path.insert(0, str(WT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("on", "off"), required=True)
    ap.add_argument("--model", default="protenix-v2")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as T

    if a.arm == "off":
        def _plain(x, w, ckc, dtype, memory_config, activation=None):
            return ttnn.linear(x, w, compute_kernel_config=ckc, dtype=dtype,
                               memory_config=memory_config, core_grid=T.CORE_GRID_MAIN,
                               activation=activation)
        T._transition_linear = _plain

    import tt_baseline as B

    one_fold, meta, state = B.build_fold(
        a.model, Path("/tmp/e6-msa"), WT / "examples" / "prot300.yaml",
        WT / "scripts" / "gpu_vs_tt" / "fixtures" / "prot300.a3m")
    cold_s, cold_m = one_fold()
    assert cold_m.get("msa"), "fold ran without an MSA"
    times, plddt = [], None
    for _ in range(a.repeat):
        t, m = one_fold()
        times.append(t)
        plddt = m.get("plddt")
    cifs = sorted(Path(meta["struct_dir"]).glob("*.cif"))
    sha = hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16] if cifs else None
    if cifs:
        (Path(a.out).parent / f"e6_{a.model}_{a.arm}.cif").write_bytes(cifs[0].read_bytes())
    res = dict(arm=a.arm, model=a.model, cold_s=round(cold_s, 2),
               warm_s=[round(t, 3) for t in times],
               median_s=round(statistics.median(times), 3),
               plddt=plddt, cif_sha16=sha,
               fired=T._transition_program_config.cache_info()._asdict() if a.arm == "on" else None,
               date=time.strftime("%Y-%m-%d %H:%M"))
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print("RESULT", json.dumps(res), file=sys.stderr)
    state.reset()
    T.cleanup()


if __name__ == "__main__":
    sys.exit(main())
