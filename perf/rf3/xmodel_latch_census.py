"""Fold one target on a pair-track model, read the L1/CB latch counters and digest the structure.

`_L1_OUT_RUNG` and `_BMM_CFG_RUNG` (the two latches `wk/rf3-l1-sibling-latch-narrow-fix` narrows
instead of retiring) sit on DEFAULT paths every pair-track model reaches, but only RF3 was
re-verified at a fold. This runs the same census `perf/rf3/trunk_decompose.py` runs for RF3
against the shipped fold path of the other models, so "does this class ever refuse on their
shapes, and does the model still emit the same structure" are numbers off the run.

Nothing moves unless the device refuses: rung 0 is the shipped config byte for byte. So a model
with 0 refusals on both latches never leaves rung 0 and cannot have changed.

It runs unchanged against main (which has neither the counters nor the ladders) so the A/B uses
one instrument. Several of these models are run-to-run nondeterministic on Blackhole, so read a
branch-vs-main digest difference only beside an A/A control on the same arm.

One model per process (one device context per process):

    TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 python3 perf/rf3/xmodel_latch_census.py \\
        --model openfold3 --aa 768 --out /tmp/latch_openfold3.json
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--aa", type=int, default=768)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recycling_steps", type=int, default=1)
    ap.add_argument("--sampling_steps", type=int, default=10)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)

    from perf_regression import SPECS, _build_cfg
    sys.path.insert(0, str(ROOT / "perf" / "rf3"))
    from make_inputs import cdk2

    from tt_bio import tenstorrent as tt_mod
    from tt_bio import esmfold2 as _E
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts

    _E.set_progress(lambda *a, **k: None)
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    work = Path(tempfile.mkdtemp(prefix="latch-%s-" % args.model))
    struct_dir, msa_dir = work / "out", work / "msa"
    struct_dir.mkdir(parents=True); msa_dir.mkdir(parents=True)

    # The fleet size fixture: CDK2 (1HCL) tiled to N aa, single chain, no MSA. Same sequence the
    # RF3 ladder folds, so a refusal here is on the same shapes RF3 was screened at.
    inp = work / ("cdk2_%d.yaml" % args.aa)
    inp.write_text("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: %s\n"
                   % cdk2(args.aa))

    cfg = _build_cfg(args.model, SPECS.get(args.model, {}), struct_dir, msa_dir)
    cfg["recycling_steps"] = args.recycling_steps
    cfg["sampling_steps"] = args.sampling_steps
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("latch", cfg)
    state.pfn = lambda *a, **k: None
    if cfg["model"] == "boltz2":
        state.model.progress_fn = lambda *a, **k: None

    # ONE cold fold, everything counted. A warm fold would absorb the first refusal (refusals are
    # cached per shape class), which is exactly the event this census exists to see.
    t0 = time.perf_counter()
    metrics, _best, _feats = state.predict_one(inp, dict(cfg, struct_dir=str(struct_dir)))
    wall = time.perf_counter() - t0

    h = hashlib.sha256()
    for f in sorted(struct_dir.rglob("*")):
        if f.is_file():
            h.update(f.name.encode()); h.update(f.read_bytes())
    digest = h.hexdigest()[:16]

    # getattr defaults so the SAME script runs against main, which has neither the counters nor
    # the rung ladders. An A/B needs one instrument, not two.
    def _d(name):
        return {str(k): v for k, v in getattr(tt_mod, name, {}).items()}

    def _s(name):
        return sorted(str(k) for k in getattr(tt_mod, name, ()))

    rep = {"model": args.model, "aa": args.aa, "fold_s": round(wall, 3),
           "struct_digest": digest,
           "metrics": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                       for k, v in dict(metrics or {}).items()},
           "latch_stats": {k: dict(v) for k, v in getattr(tt_mod, "LATCH_STATS", {}).items()},
           "l1_out_rung": _d("_L1_OUT_RUNG"),
           "l1_out_refused": _s("_L1_OUT_REFUSED"),
           "bmm_cfg_rung": _d("_BMM_CFG_RUNG"),
           "bmm_cfg_refused": _s("_BMM_CFG_REFUSED"),
           "transpose_l1_refused": _s("_TRANSPOSE_L1_REFUSED")}
    Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    print("%s %d aa, %.1f s, struct sha256[:16] %s" % (args.model, args.aa, wall, digest))
    for name, st in rep["latch_stats"].items():
        print("  %-16s served %-8d refused %-4d blocked %-6d declined %d"
              % (name, st["served"], st["refused"], st["blocked"], st["declined"]))
    print("  l1_out refused classes: %s" % (rep["l1_out_refused"],))
    if rep["l1_out_rung"] or rep["bmm_cfg_rung"]:
        print("  rungs moved: l1_out %s  bmm_cfg %s" % (rep["l1_out_rung"], rep["bmm_cfg_rung"]))


if __name__ == "__main__":
    main()
