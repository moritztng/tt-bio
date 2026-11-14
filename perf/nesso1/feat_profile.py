"""Where Nesso-1's host time goes, phase by phase, without opening a device.

Times each step `tt_bio.nesso1.screen` runs before `model.predict`, in a fresh output dir:
find_ccd, preprocess (YAML parse + RDKit conformer), ESM-2, build_dataset, then
`dataset[0]` `--items` times (the per-record featurisation a screen pays per ligand). Writes
a JSON report with seconds, host load, and a sha256 per feature tensor so a change to the
featurisation can be checked for identical output. `--profile` adds a cProfile top list per
phase.

    python perf/nesso1/feat_profile.py --yaml perf/nesso1/inputs/ladder/aa512 --out /tmp/np
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
import pstats
import shutil
import socket
import time
from pathlib import Path

import torch


def digest(item: dict) -> dict:
    out = {}
    for k in sorted(item):
        v = item[k]
        if isinstance(v, torch.Tensor):
            t = v.detach().contiguous().cpu()
            out[k] = f"{t.dtype}:{tuple(t.shape)}:" + hashlib.sha256(
                t.view(torch.uint8).numpy().tobytes() if t.numel() else b"").hexdigest()[:16]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    from tt_bio import nesso1_input as NI

    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    rep = {"host": socket.gethostname(), "nproc": os.cpu_count(),
           "torch_threads": torch.get_num_threads(), "load_start": os.getloadavg(),
           "phases": {}, "profiles": {}}

    def phase(name, fn):
        pr = cProfile.Profile() if a.profile else None
        t0 = time.perf_counter()
        if pr:
            pr.enable()
        r = fn()
        if pr:
            pr.disable()
        dt = time.perf_counter() - t0
        rep["phases"].setdefault(name, []).append(round(dt, 4))
        if pr:
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
            rep["profiles"].setdefault(name, s.getvalue())
        print(f"[feat] {name} {dt:.3f}s", flush=True)
        return r

    yamls = NI.find_yamls(Path(a.yaml))
    paths = NI.resolve_paths(out)
    ccd = phase("find_ccd", lambda: NI.find_ccd())
    manifest, failed = phase("preprocess", lambda: NI.preprocess(yamls, paths, ccd, num_workers=a.num_workers))
    assert not failed, failed
    seqs, given = NI.collect_esm(yamls)
    phase("esm", lambda: NI.run_esm(seqs, paths.esm_dir))
    ds = phase("build_dataset", lambda: NI.build_dataset(paths, manifest, ccd))
    digs = []
    for _ in range(a.items):
        torch.manual_seed(a.seed)
        item = phase("getitem", lambda: ds[0])
        assert not item.get("exception"), item
        digs.append(digest(item))
    rep["features_identical_across_items"] = all(d == digs[0] for d in digs)
    rep["features"] = digs[0]
    rep["features_sha"] = hashlib.sha256(json.dumps(digs[0], sort_keys=True).encode()).hexdigest()[:16]
    rep["load_end"] = os.getloadavg()
    Path(a.report).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("phases", "features_sha", "features_identical_across_items",
                                           "load_start", "load_end")}))


if __name__ == "__main__":
    main()
