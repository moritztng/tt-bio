#!/usr/bin/env python3
"""Fold-level A/B for the shared AttentionPairBias matmul sites.

One process per model. The cold fold is discarded, then the two arms alternate fold by fold so
host drift cannot land on one of them: 'off' rebinds tt_bio.tenstorrent.batched_matmul to a plain
ttnn.matmul, which is byte-for-byte the call the sites made before this branch, and 'on' restores
the shipped helper. Both configs end up in the ttnn program cache, so neither arm pays a compile
the other does not.

  TT_VISIBLE_DEVICES=1 PYTHONPATH=$WT python3 perf/shared_mm/fold_ab.py --model protenix-v2 --pairs 3
"""
import argparse, importlib.util, json, time
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
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--target", default=str(ROOT / "examples" / "prot300.yaml"))
    ap.add_argument("--a3m", default=str(ROOT / "scripts" / "gpu_vs_tt" / "fixtures" / "prot300.a3m"))
    ap.add_argument("--msa-dir", default=str(Path.home() / ".tt_bio_msa_cache"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = _load_baseline()
    import ttnn
    import tt_bio.tenstorrent as T
    shipped = T.batched_matmul

    def plain(x, y, **kw):
        return ttnn.matmul(x, y, **kw)

    one_fold, meta, _state = base.build_fold(a.model, Path(a.msa_dir), Path(a.target), Path(a.a3m))

    T.batched_matmul = plain
    t = time.perf_counter()
    one_fold()
    print(f"cold fold (off) {time.perf_counter() - t:.3f} s", flush=True)
    T.batched_matmul = shipped
    t = time.perf_counter()
    one_fold()
    print(f"cold fold (on)  {time.perf_counter() - t:.3f} s", flush=True)

    res = {"off": [], "on": []}
    for i in range(a.pairs):
        for arm, fn in (("off", plain), ("on", shipped)):
            T.batched_matmul = fn
            t = time.perf_counter()
            one_fold()
            dt = time.perf_counter() - t
            res[arm].append(dt)
            print(f"pair {i} {arm}: {dt:.3f} s", flush=True)

    off, on = res["off"], res["on"]
    mean = lambda v: sum(v) / len(v)
    out = dict(model=a.model, off_s=off, on_s=on, off_mean=mean(off), on_mean=mean(on),
               ratio=mean(off) / mean(on), saved_ms=(mean(off) - mean(on)) * 1e3, meta=meta)
    print(json.dumps({k: v for k, v in out.items() if k != "meta"}, indent=1), flush=True)
    json.dump(out, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
