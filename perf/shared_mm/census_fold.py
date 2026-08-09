#!/usr/bin/env python3
"""One warm 298 aa fold with the ttnn.matmul census active, for any tt-bio model.

E1 censused OpenFold3 only. Two of its sites were left unowned, and one of them
(tenstorrent.py:1796/1804, the AttentionPairBias unfused attention) is in shared code, so the
question 'which models reach it' has to be answered by running them, not by grepping.

The cold fold is excluded: it compiles kernels and issues the same calls a second time, which
would double every count. Reset the counter after it and let the atexit dump write the warm fold.
"""
import argparse, importlib.util, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_baseline():
    spec = importlib.util.spec_from_file_location(
        "tt_baseline", ROOT / "scripts" / "gpu_vs_tt" / "tt_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", default=str(ROOT / "examples" / "prot300.yaml"))
    ap.add_argument("--a3m", default=str(ROOT / "scripts" / "gpu_vs_tt" / "fixtures" / "prot300.a3m"))
    ap.add_argument("--msa-dir", default=str(Path.home() / ".tt_bio_msa_cache"))
    ap.add_argument("--folds", type=int, default=2,
                    help="2 = census the warm fold (cold fold discarded); 1 = census the cold fold, "
                         "which issues the same calls and halves the device time")
    a = ap.parse_args()

    base = _load_baseline()
    one_fold, meta, _state = base.build_fold(a.model, Path(a.msa_dir), Path(a.target), Path(a.a3m))
    import builtins
    if a.folds > 1:
        t = time.perf_counter()
        one_fold()
        print(f"cold fold {time.perf_counter() - t:.3f} s", flush=True)
        builtins._census_reset()
    t = time.perf_counter()
    one_fold()
    print(f"censused fold {time.perf_counter() - t:.3f} s", flush=True)


if __name__ == "__main__":
    main()
