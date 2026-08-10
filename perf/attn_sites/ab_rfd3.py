#!/usr/bin/env python3
"""End-to-end bit-exactness A/B for the tuned attn@v config, RFD3 design path.

Same method as ab_exact.py's fold path: one process, one device, one set of weights, one seed.
Run the design once with the tuned program config engaged and once with it forced off, and
compare the written mmCIFs byte for byte. Everything except the two program configs is provably
identical, which a branch-vs-main A/B cannot claim.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import tt_bio.tenstorrent as T  # noqa: E402


def _off(*a, **k):
    return None


_off.cache_clear = lambda: None


def cif_digest(d: Path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(d.rglob("*.cif"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="perf/attn_sites/rfd3_iai_298.yaml")
    ap.add_argument("--timesteps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import tt_bio.main as tb_main

    tuned = T._attn_value_program_config
    hits = {"n": 0}

    def counting(*a, **k):
        r = tuned(*a, **k)
        if r is not None:
            hits["n"] += 1
        return r
    counting.cache_clear = tuned.cache_clear
    T._attn_value_program_config = counting

    res = {}
    for label in ("tuned", "off"):
        if label == "off":
            T._attn_value_program_config = _off
            hits["at_switch"] = hits["n"]
        out = REPO / f"_ab_rfd3_{label}"
        shutil.rmtree(out, ignore_errors=True)
        sys.argv = ["tt-bio", "design", args.spec, "--model", "rfd3", "--from_pdb",
                    "--num_timesteps", str(args.timesteps), "--num_designs", "1",
                    "--seed", str(args.seed), "--out_dir", str(out)]
        tb_main.cli(standalone_mode=False)
        res[label] = cif_digest(out)
        print(f"[ab] {label}: {res[label]}", flush=True)

    print(f"[ab] tuned-config selections during the tuned design: {hits.get('at_switch', hits['n'])}")
    same = bool(res["tuned"]) and res["tuned"] == res["off"]
    print("[ab] RESULT:", "BIT-IDENTICAL" if same else "DIFFER / NO OUTPUT")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
