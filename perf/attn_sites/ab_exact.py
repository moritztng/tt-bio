#!/usr/bin/env python3
"""End-to-end bit-exactness A/B for the tuned attn@v program config.

One process, one device, one set of weights: fold once with the tuned config engaged and once
with it forced off, and compare the written mmCIF byte for byte. Stronger than a branch-vs-main
A/B because everything except the three program configs is provably identical.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

import tt_bio.tenstorrent as T  # noqa: E402


def _off(*a, **k):
    return None


_off.cache_clear = lambda: None


def cif_digest(d: Path):
    out = {}
    for p in sorted(d.rglob("*.cif")):
        out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmfold2")
    ap.add_argument("--target", default="examples/prot300.yaml")
    ap.add_argument("--a3m", default="prot300.a3m")
    args = ap.parse_args()

    import tt_baseline as TB

    tuned = T._attn_value_program_config
    hits = {"n": 0}
    def counting(*a, **k):
        r = tuned(*a, **k)
        if r is not None:
            hits["n"] += 1
        return r
    counting.cache_clear = tuned.cache_clear
    T._attn_value_program_config = counting

    one_fold, meta, _state = TB.build_fold(
        args.model, REPO / "perf" / "attn_sites" / "_msa",
        REPO / args.target, Path(TB.FIXTURES) / args.a3m)
    print(f"[ab] model={args.model} meta={meta}", flush=True)

    res = {}
    for label in ("tuned", "off"):
        if label == "off":
            T._attn_value_program_config = _off
        out = Path(meta["struct_dir"])
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        one_fold()
        res[label] = cif_digest(out)
        print(f"[ab] {label}: {res[label]}", flush=True)

    print(f"[ab] tuned-config selections during the tuned fold: {hits['n']}")
    same = res["tuned"] == res["off"] and res["tuned"]
    print("[ab] RESULT:", "BIT-IDENTICAL" if same else "DIFFER / NO OUTPUT")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
