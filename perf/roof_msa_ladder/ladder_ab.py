#!/usr/bin/env python3
"""The MSA depth ladder, priced: paired interleaved folds in ONE process, plus the rung's compile.

Two arms of `TT_BIO_MSA_LADDER` on the same fixture, the same process and the same card:

    off   the shipped single multiple, so a 35-row alignment pads to 1024
    on    the ladder, so it pads to 64

They alternate within a pair and the pair order flips every block, so a monotone drift across the
run cannot masquerade as an effect. The per-arm cold fold is taken FIRST and reported separately:
the arms run different shapes, so one arm's cold fold does not compile the other's kernels, and the
difference between an arm's cold fold and its warm median is the compile cost of its rung -- the
number that decides whether a rung earns its place.

Every fold's CIF is kept under its own name, so the same run gives the A/A floor (two folds of one
arm against each other) and the arm-to-arm deviation. On a card with a known nondeterminism that
floor is not a formality, it is the only thing that makes the arm number readable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH is not None:
        OUT_PATH.write_text(json.dumps(OUT, indent=2) + "\n")


def patch_cfg():
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()


def run_size(ttnn, T, B, size, pairs, cifdir, bracket):
    import tt_bio.token_axis as TA

    # fold_ab_multi's config patch reads these off tt_baseline, which only defines
    # SAMPLING_STEPS; the recycle count is the per-model default the fold itself resolves.
    B.RECYCLING_STEPS = B._resolve_recycling_steps(None, "boltz2")
    patch_cfg()
    T.get_device()
    fix = ROOT / "perf" / "size512" / "fixtures"
    # A "size" may name a local deep-MSA fixture instead of a token count, so one harness covers
    # both axes: the token sweep lives in perf/size512, the depth sweep next to this file.
    yml, a3m = fix / f"cdk2x2_{size}.yaml", fix / f"cdk2x2_{size}.a3m"
    if not yml.is_file():
        yml, a3m = HERE / f"{size}.yaml", HERE / f"{size}.a3m"
    one_fold, meta, _state = B.build_fold("boltz2", HERE / f".msa_{size}", yml, a3m)
    dev = T.get_device()
    rec = {"n_msa": meta["n_msa"], "hardware": meta["hardware"], "grid": meta.get("grid"),
           "card_type": meta.get("card_type"), "recycling_steps": meta["recycling_steps"],
           "struct_dir": meta["struct_dir"], "folds": []}
    OUT.setdefault("sizes", {})[str(size)] = rec

    # The padded depth each arm actually ran, read off the module rather than assumed.
    depths: list[int] = []
    track_ms: list[float] = []
    mod = T.MSA
    o_mod = mod.__dict__["__call__"]

    def w_mod(self_obj, *a, **k):
        depths.append(int(a[1].shape[1]))
        if not bracket:
            return o_mod(self_obj, *a, **k)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = o_mod(self_obj, *a, **k)
        ttnn.synchronize_device(dev)
        track_ms.append(1e3 * (time.perf_counter() - t0))
        return out

    mod.__call__ = w_mod
    cifdir.mkdir(parents=True, exist_ok=True)

    def fold(arm, tag):
        os.environ["TT_BIO_MSA_LADDER"] = arm
        depths.clear(); track_ms.clear()
        t, m = one_fold()
        cifs = {}
        for f in sorted(Path(meta["struct_dir"]).glob("*.cif")):
            cifs[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            shutil.copy2(f, cifdir / f"{size}_{tag}_{arm}_{f.name}")
        r = {"arm": arm, "tag": tag, "s": round(t, 3), "plddt": m.get("plddt"),
             "padded_rows": sorted(set(depths)), "cif_sha256": cifs,
             "loadavg": open("/proc/loadavg").read().split()[:3]}
        if bracket:
            r["msa_track_s"] = round(sum(track_ms) / 1e3, 4)
            r["msa_calls"] = len(track_ms)
        rec["folds"].append(r)
        pl = r["plddt"]
        print(f"  {size} {tag:8s} ladder={arm}  {t:7.3f} s  depth {r['padded_rows']} "
              f"plddt {pl if pl is None else round(float(pl), 4)}"
              + (f"  track {r['msa_track_s']:.3f} s" if bracket else ""), flush=True)
        dump()
        return t

    try:
        # Cold folds first, one per arm. "off" goes first so the shipped shape is the one that
        # pays for the model load's own warmup, not the arm under test.
        rec["cold_s"] = {"off": round(fold("0", "cold"), 3), "on": round(fold("1", "cold"), 3)}
        arms = {"0": [], "1": []}
        for i in range(pairs):
            for arm in (("0", "1") if i % 2 == 0 else ("1", "0")):
                arms[arm].append(fold(arm, f"warm{i}"))
        rec["warm_s"] = {k: [round(x, 3) for x in v] for k, v in arms.items()}
        rec["warm_median_s"] = {k: round(st.median(v), 3) for k, v in arms.items()}
        deltas = [round(a - b, 3) for a, b in zip(arms["0"], arms["1"])]
        rec["paired_delta_s"] = deltas
        rec["paired_delta_median_s"] = round(st.median(deltas), 3)
        rec["speedup"] = round(rec["warm_median_s"]["0"] / rec["warm_median_s"]["1"], 4)
        rec["rung_compile_s"] = {k: round(rec["cold_s"][n] - rec["warm_median_s"][k], 3)
                                 for k, n in (("0", "off"), ("1", "on"))}
        print(f"  {size} MEDIAN off {rec['warm_median_s']['0']:.3f} s  on "
              f"{rec['warm_median_s']['1']:.3f} s  ->  {rec['speedup']:.4f}x, paired deltas "
              f"{deltas}", flush=True)
    finally:
        mod.__call__ = o_mod
        os.environ.pop("TT_BIO_MSA_LADDER", None)
    dump()


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--sizes", default="512")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--bracket", action="store_true",
                    help="sync-bracket every MSA.__call__. Attribution only: it perturbs the "
                         "fold, so the headline ratio is taken without it.")
    a = ap.parse_args()
    OUT_PATH = a.out
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import tt_bio.token_axis as TA

    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "host": os.uname().nodename,
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "ladder": list(TA.MSA_PAD_LADDER),
                  "bracket": a.bracket, "pairs": a.pairs,
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "ttnn": getattr(ttnn, "__version__", "?")}
    dump()
    for s in a.sizes.split(","):
        run_size(ttnn, T, B, s, a.pairs, a.cifdir, a.bracket)
    T.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
