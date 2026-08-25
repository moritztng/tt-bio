#!/usr/bin/env python3
"""RF3's fused triangle-attention compute-config arms, interleaved in one process at one size.

The arm under test is `tri_att_sdpa_hifi`: the fused SDPA's compute kernel config, which the
kernel takes per call. Shipped it is the op default `(HiFi2, math_approx on, no fp32_dest_acc)`;
the selector asks for `(HiFi4, math_approx off, fp32_dest_acc on)`, which is the reduction width
the GPU reference runs. Both arms share the route, the fixture, the checkpoint, the device and the
program cache -- only the config moves -- so the difference is the arithmetic and nothing else.

Interleaved in ONE process rather than a process per arm, for the same reason `ladder_arm_ab.py`
is: the effect is a few percent, a per-arm process pays a second warm-up, and the process-to-
process spread is the term that would swamp it. The A/A control is the spread WITHIN arm `a1`
across its own warm folds, and it is printed first, because a difference inside it is not a result.

Three things are counted per fold rather than assumed:

* which route each triangle-attention call took (`fused` vs the stock op, which takes no compute
  config at all and so runs the op default whatever the arm asked for),
* which (q_chunk, k_chunk) the kernel served -- `fp32_dest_acc` doubles the DST a tile needs, so
  an arm can be pushed onto a narrower q_chunk and pay a ROUTE change inside what reads as a
  fidelity change,
* the L1 refusal memos, which are CLEARED between arms. Measured at 768 aa: the op default tries
  q_chunk 768, is refused and retires it, and any arm running later in the same process inherits
  the retirement and is never offered q768. Without the clear, the arm ORDER decides the chunks.

    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 python3 perf/rf3/hifi_ckc_ab.py \\
        --aa 768 --arms a1,a5 --sweeps 3 --out perf/rf3/results/hifi_ckc_768.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
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
    ap.add_argument("--aa", type=int, required=True)
    ap.add_argument("--arms", default="a1,a5", help="comma-separated arms from rf3_port/arms.py")
    ap.add_argument("--sweeps", type=int, default=3,
                    help="warm sweeps through the arm list, after one discarded cold sweep")
    ap.add_argument("--recycling_steps", type=int, default=10, help="RF3 ships 10")
    ap.add_argument("--sampling_steps", type=int, default=50, help="RF3 ships 50")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)

    from perf_regression import SPECS, _build_cfg
    sys.path.insert(0, str(ROOT / "perf" / "rf3"))
    from make_inputs import cdk2

    from tt_bio import tenstorrent as tt
    from tt_bio import triatt_sdpa as pm
    from tt_bio import esmfold2 as _E
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    from tt_bio.rf3.remap import PAIRFORMER_FLAGS
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from arms import ARMS, ROUTE, clear_l1_latches, route_counters

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit("unknown arm %r; have %s" % (a, sorted(ARMS)))
        if ARMS[a].get("sdpa_hifi_site"):
            raise SystemExit(
                "%s selects the per-site flag, which is read at CONSTRUCTION and so cannot be "
                "interleaved. Score it in its own process (accuracy_cell) as the identity check "
                "against a5, not here." % a)

    _E.set_progress(lambda *a, **k: None)
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    assert Path(tt.__file__).resolve().is_relative_to(ROOT), \
        "tt_bio resolves to %s, not this checkout -- set PYTHONPATH" % tt.__file__

    # Every arm here is the SHIPPED route: the fused SDPA with the ragged tail masked. Only the
    # compute kernel config moves. Set once, before the model is built, and asserted per fold.
    PAIRFORMER_FLAGS["fp32_softmax"] = False
    if tt._FP32_SOFTMAX:
        raise SystemExit("BOLTZ2_FP32_SOFTMAX=1 forces the materialised route; unset it.")

    work = Path(tempfile.mkdtemp(prefix="rf3-hifi-ckc-%d-" % args.aa))
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
    state.bind_run("hifickcab", cfg)
    state.pfn = lambda *a, **k: None

    folds = []
    order = arms * (1 + args.sweeps)
    for i, arm in enumerate(order):
        clear_l1_latches()
        tt.SDPA_RAGGED_PAD_STATS[0] = 0
        tt.SDPA_RAGGED_SITES.clear()
        pm._CKC_OVERRIDE = (None if "sdpa_ckc" not in ARMS[arm]
                            else pm.ckc_from_env(ARMS[arm]["sdpa_ckc"]))
        for f in struct_dir.rglob("*"):
            if f.is_file():
                f.unlink()
        t0 = time.perf_counter()
        metrics, _best, _feats = state.predict_one(inp, dict(cfg, struct_dir=str(struct_dir)))
        wall = time.perf_counter() - t0
        rec = {"i": i, "arm": arm, "cold": i < len(arms), "fold_s": round(wall, 3),
               "digest": digest_dir(struct_dir),
               "route": ROUTE[arm],
               "counters": route_counters(),
               "ragged_sites": {k: list(v) for k, v in tt.SDPA_RAGGED_SITES.items()},
               "ragged_padded": tt.SDPA_RAGGED_PAD_STATS[0],
               "n_tokens": (metrics or {}).get("n_tokens"),
               "plddt": (metrics or {}).get("plddt")}
        folds.append(rec)
        print("%2d %-4s%s %9.3f s  digest %s  ckc %-28s picks %s  fused/stock %s"
              % (i, arm, " cold" if rec["cold"] else "    ", wall, rec["digest"],
                 rec["counters"]["ckc_resolved"], rec["counters"]["sdpa_chunk_picks"],
                 rec["counters"]["sdpa_route_counts"]), flush=True)
        Path(args.out).write_text(json.dumps({"partial": True, "folds": folds}, indent=1) + "\n")

    warm = [f for f in folds if not f["cold"]]
    per_arm = {}
    for arm in arms:
        ts = [f["fold_s"] for f in warm if f["arm"] == arm]
        digs = sorted({f["digest"] for f in warm if f["arm"] == arm})
        per_arm[arm] = {"n": len(ts), "folds_s": ts, "median_s": statistics.median(ts),
                        "spread_pct": (round(100 * (max(ts) - min(ts)) / min(ts), 3)
                                       if len(ts) > 1 else None),
                        "digests": digs}
    base = arms[0]
    rep = {"model": "rf3", "aa": args.aa, "arms": arms, "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": list(tt.COMPUTE_GRID_MAIN),
           "recycling_steps": args.recycling_steps, "sampling_steps": args.sampling_steps,
           "aa_control_arm": base,
           "aa_control_spread_pct": per_arm[base]["spread_pct"],
           "folds": folds, "per_arm": per_arm,
           "cost_vs_%s" % base: {a: round(per_arm[a]["median_s"] / per_arm[base]["median_s"], 4)
                                 for a in arms}}
    Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")

    print("--- rf3 %d aa, grid %s" % (args.aa, rep["grid"]))
    print("A/A control: arm %s spread %s %% across %d warm folds -- a difference inside this is "
          "not a result" % (base, per_arm[base]["spread_pct"], per_arm[base]["n"]))
    for a in arms:
        m = per_arm[a]
        print("  %-4s median %9.3f s  n=%d  spread %s %%  cost %.4fx  digests %s"
              % (a, m["median_s"], m["n"], m["spread_pct"],
                 m["median_s"] / per_arm[base]["median_s"], m["digests"]))


if __name__ == "__main__":
    main()
