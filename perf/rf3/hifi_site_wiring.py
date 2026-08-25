#!/usr/bin/env python3
"""Does the per-site `tri_att_sdpa_hifi` selector reach a5's arithmetic? Counted, not argued.

a10 is a5 asked for the way it would ship: `PAIRFORMER_FLAGS['tri_att_sdpa_hifi']` instead of the
process-global `triatt_sdpa._CKC_OVERRIDE`. The flag is read at CONSTRUCTION, so a10 cannot be
interleaved with the other arms and gets its own process -- one arm per invocation.

This is the ROUTE-COUNTER form of the check, and it is deliberately not the digest form. The
counters (`sdpa_hifi_calls`, fused-vs-stock, the q/k chunk pick, the resolved config) are
configuration facts: they say which kernel was asked for and how many calls it served. They are
answerable on a card whose matmuls are not trustworthy, which a bit-identity digest is not.
The digest is recorded anyway, with the caveat that on pc card 0 an a1-vs-a1 digest difference is
the card, not the arm (`pc-card0-512aa-fold-nondeterminism`).

`sdpa_hifi_calls` is the discriminator. tenstorrent.py:1005 increments it only when
`_tri_att_sdpa` is PASSED a ckc, i.e. only on the per-site path, so:

    a1    sdpa_hifi_calls == 0, ckc_resolved == op_default
    a5    sdpa_hifi_calls == 0, ckc_resolved == HiFi4,approx=False,fp32_dest_acc=True  (global)
    a10   sdpa_hifi_calls == pm_served > 0, ckc_resolved == op_default (the global is untouched)

a10 wired right means: every call the fused kernel serves is a call that carried the per-site
config, and the chunk picks match a5's. A chunk-pick difference means `fp32_dest_acc`'s doubled DST
pushed one arm onto a narrower q_chunk, i.e. a route change wearing a fidelity change's clothes.
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "scripts" / "rf3_port"))


def digest_dir(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(d.rglob("*")):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=512)
    ap.add_argument("--arm", required=True, help="one arm from scripts/rf3_port/arms.py")
    ap.add_argument("--repeat", type=int, default=1, help="folds in this process, after the first")
    ap.add_argument("--recycling_steps", type=int, default=1)
    ap.add_argument("--sampling_steps", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)

    from perf_regression import SPECS, _build_cfg
    sys.path.insert(0, str(ROOT / "perf" / "rf3"))
    from make_inputs import cdk2

    from tt_bio import tenstorrent as tt
    from tt_bio import esmfold2 as _E
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    from tt_bio.rf3.remap import PAIRFORMER_FLAGS
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from arms import apply_arm, clear_l1_latches, route_counters

    _E.set_progress(lambda *a, **k: None)
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    assert Path(tt.__file__).resolve().is_relative_to(ROOT), \
        "tt_bio resolves to %s, not this checkout -- set PYTHONPATH" % tt.__file__

    # BEFORE load_model: the per-site flag and fp32_softmax are both read at construction.
    applied = apply_arm(args.arm)
    print(json.dumps({"applied": applied}), flush=True)

    work = Path(tempfile.mkdtemp(prefix="rf3-hifi-site-%s-%d-" % (args.arm, args.aa)))
    struct_dir, msa_dir = work / "out", work / "msa"
    struct_dir.mkdir(parents=True)
    msa_dir.mkdir(parents=True)
    inp = work / ("cdk2_%d.yaml" % args.aa)
    inp.write_text("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: %s\n"
                   % cdk2(args.aa))

    cfg = _build_cfg("rf3", SPECS.get("rf3", {}), struct_dir, msa_dir)
    cfg["recycling_steps"] = args.recycling_steps
    cfg["sampling_steps"] = args.sampling_steps
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("hifisite", cfg)
    state.pfn = lambda *a, **k: None

    folds = []
    for i in range(args.repeat + 1):
        clear_l1_latches()
        for f in struct_dir.rglob("*"):
            if f.is_file():
                f.unlink()
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(inp, dict(cfg, struct_dir=str(struct_dir)))
        wall = time.perf_counter() - t0
        rec = {"fold": i, "cold": i == 0, "fold_s": round(wall, 3),
               "n_tokens": (metrics or {}).get("n_tokens"),
               "cif_digest": digest_dir(struct_dir),
               "counters": route_counters(),
               "triatt_fused_hifi_stats": dict(tt.TRIATT_FUSED_HIFI_STATS)
               if hasattr(tt, "TRIATT_FUSED_HIFI_STATS") else None}
        folds.append(rec)
        print(json.dumps(rec), flush=True)

    Path(args.out).write_text(json.dumps(
        {"aa": args.aa, "arm": args.arm, "host": os.uname().nodename,
         "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": list(tt.COMPUTE_GRID_MAIN),
         "recycling_steps": args.recycling_steps, "sampling_steps": args.sampling_steps,
         "applied": applied,
         "pairformer_flags": {k: repr(v) for k, v in PAIRFORMER_FLAGS.items()},
         "folds": folds}, indent=1) + "\n")


if __name__ == "__main__":
    main()
