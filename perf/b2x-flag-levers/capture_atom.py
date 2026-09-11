#!/usr/bin/env python3
"""Measure the atom transformer's real DRAM traffic with and without TT_BIO_ATOM_AXIS_BUCKET.

The flag changes only the tensor extent: 224 query windows become 140, the op graph, the dtypes
and the program count are all unchanged. That makes it the fold's cleanest bytes-versus-
transactions separator, and the separation is only readable if the byte side is MEASURED rather
than modelled. `perf/b2x-atom-padding/predict.py` models it as 0.657 MB of weights plus
1.3538 MB/window, i.e. 0.6258x; this checks that against a capture.

Both arms are captured in one process off one device open (the p300c wedges on its fourth open
since reset), and both go through `capture_difftx.wrap`, so the counts land on exactly the same
instrument as the corrected `real_traffic.py` rows rather than on a fourth byte counter.

The atom layers are `DiffusionTransformerLayer` instances too -- the window count is the batch
dim, so the arms are distinguishable by the capture signature itself: `difftx|1x224x32x128`
against `difftx|1x140x32x128`.

No timing here. Graph capture perturbs what it measures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x_difflayer"))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))

import capture_difftx as CD                                                   # noqa: E402
import real_traffic as RT                                                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--recycles", type=int, default=1)
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *x, **k: None)
    import ab_flag_levers as AB

    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = a.steps, a.recycles
    import tempfile
    work = Path(tempfile.mkdtemp(prefix="b2x-flaglev-bytes-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    fix = REPO / "perf" / "size512" / "fixtures"
    AB._seed_msa(fix / f"cdk2x2_{a.size}.yaml", (fix / f"cdk2x2_{a.size}.a3m").read_text(), msa_dir)
    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)

    dev = get_device()
    CD.DEV["d"] = dev
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2x-flag-levers-bytes", cfg)
    state.model.progress_fn = lambda *x, **k: None

    CD.wrap(TT.DiffusionTransformerLayer, "difftx", 3)
    target = fix / f"cdk2x2_{a.size}.yaml"

    for arm, bucket in (("off", False), ("on", True)):
        TT._ATOM_AXIS_BUCKET = bucket
        try:
            state.model.structure_module.score_model.reset_static_cache()
        except Exception:                                                     # noqa: BLE001
            pass
        CD.SEEN.clear()
        for p in struct_dir.glob("*"):
            if p.is_file():
                p.unlink()
        state.predict_one(target, cfg)
        print(f"arm bucket={arm}: sigs so far {sorted(CD.DUMP)}", flush=True)

    out = {"size": a.size, "steps": a.steps, "recycles": a.recycles,
           "instrument": "perf/b2x_difflayer/real_traffic.py via capture_difftx.wrap",
           "calls": {}}
    for s, call in CD.DUMP.items():
        c = RT.counts(call)
        c.pop("per_op_detail", None)
        c["per_op_top"] = c.pop("per_op")[:12]
        c["inputs"] = call["inputs"]
        out["calls"][s] = c
        print(f"  {s:26s} real {c['real_MB']:8.3f} MB  once {c['once_MB']:8.3f}  "
              f"floor {c['floor_MB']:8.3f}  ops {c['n_ops']}", flush=True)

    atom = {s: v for s, v in out["calls"].items() if "x32x128" in s}
    if len(atom) == 2:
        hi = max(atom, key=lambda s: atom[s]["real_MB"])
        lo = min(atom, key=lambda s: atom[s]["real_MB"])
        nw_hi = int(hi.split("|")[1].split("x")[1])
        nw_lo = int(lo.split("|")[1].split("x")[1])
        out["atom_delta"] = {
            "off_sig": hi, "on_sig": lo,
            "windows_off": nw_hi, "windows_on": nw_lo,
            "window_ratio": round(nw_lo / nw_hi, 5),
            "real_MB_off": round(atom[hi]["real_MB"], 3),
            "real_MB_on": round(atom[lo]["real_MB"], 3),
            "byte_ratio": round(atom[lo]["real_MB"] / atom[hi]["real_MB"], 5),
            "deleted_MB_per_call": round(atom[hi]["real_MB"] - atom[lo]["real_MB"], 3),
            "n_ops_off": atom[hi]["n_ops"], "n_ops_on": atom[lo]["n_ops"],
            "predicted_byte_ratio": 0.6258,
        }
        print("\natom delta:", json.dumps(out["atom_delta"], indent=1), flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
