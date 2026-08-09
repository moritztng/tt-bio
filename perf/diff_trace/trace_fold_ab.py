#!/usr/bin/env python3
"""D4 -- fold-level A/B of `fold(trace=)` on the production predict path, both models.

`trace_ab.py` measures the diffusion stage in isolation with one trunk. This runs the real
`predict_one` path (featurise -> trunk -> sample -> CIF) so the fold-level currency is measured
where the scoreboard wants it, and so opendde -- whose conditioning is built inside its own
fold() -- is covered by the same harness as protenix-v2.

Arm order is deliberate: eager, traced, traced, eager. The second traced fold is the test that
matters for correctness. `DiffusionModule._trace` is keyed on the atom count N only, so a second
fold of the same target hits the cached trace, and that trace has the FIRST fold's conditioning
tensors baked into it. If a trace survives across folds it must reproduce the eager coords; if it
does not, `fold(trace=True)` is unsafe for any process that folds the same-size target twice.
The trailing eager fold is the interleaving gate (memory ttnn-trace-interleaved-eager-corruption).

RNG: every fold runs at the same fixed seed and `edm_sample` re-seeds the global torch RNG at
sampler entry, so all four arms consume an identical noise stream; the initial-noise hash is
recorded per arm and compared (memory diffusion-port-parity-shared-draws).

    TT_VISIBLE_DEVICES=2,3 python3 perf/diff_trace/trace_fold_ab.py --model opendde \
        --out perf/diff_trace/trace_fold_ab_opendde_298aa.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

os.environ.setdefault("TT_BIO_TRACE_REGION_SIZE", str(1 << 30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="opendde")
    ap.add_argument("--target", default="examples/prot300.yaml")
    ap.add_argument("--a3m", default="scripts/gpu_vs_tt/fixtures/prot300.a3m")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="eager,traced,traced,eager")
    args = ap.parse_args()

    import torch
    import ttnn
    torch.set_grad_enabled(False)
    from tt_baseline import build_fold                                  # noqa: E402
    from tt_bio import protenix as P                                    # noqa: E402
    from tt_bio.tenstorrent import trace_region_size, cleanup           # noqa: E402

    work = Path(os.environ.get("TMPDIR", "/tmp")) / f"d4fold-{args.model}"
    msa_dir = work / "msa"
    one_fold, meta, state = build_fold(args.model, msa_dir, REPO / args.target, REPO / args.a3m)
    assert trace_region_size() > 0, "device opened without a trace region"
    dev = state.model.diffusion.dev if hasattr(state.model, "diffusion") else None

    R = {"model": args.model, "target": args.target, "ttnn": meta.get("ttnn_version"),
         "hardware": meta["hardware"], "visible": os.environ.get("TT_VISIBLE_DEVICES"),
         "trace_region_bytes": trace_region_size(), "arms": args.arms.split(",")}
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # --- instrument edm_sample (both models route through it) --------------------------
    rec = []
    _orig_edm = P.edm_sample

    def edm(diffusion_module, cond, n_atoms, **kw):
        d = diffusion_module.dev
        ttnn.synchronize_device(d)
        t = time.perf_counter()
        x = _orig_edm(diffusion_module, cond, n_atoms, **kw)
        ttnn.synchronize_device(d)
        rec.append({"stage_s": time.perf_counter() - t, "n_atoms": int(n_atoms),
                    "n_step": kw.get("n_step"), "trace": bool(kw.get("trace")),
                    "coords": x.detach().clone()})
        return x
    P.edm_sample = edm

    from tt_bio.protenix import DiffusionModule
    cap = {"s": 0.0, "n": 0}
    _orig_cap = DiffusionModule._capture_trace

    def cap_t(self, *a, **k):
        ttnn.synchronize_device(self.dev)
        t = time.perf_counter()
        try:
            return _orig_cap(self, *a, **k)
        finally:
            ttnn.synchronize_device(self.dev)
            cap["s"] += time.perf_counter() - t
            cap["n"] += 1
    DiffusionModule._capture_trace = cap_t

    # --- cold fold (kernel compile), never counted -------------------------------------
    meta["job_cfg"]["trace"] = False
    t_cold, m_cold = one_fold()
    R["cold_s"] = round(t_cold, 2)
    R["n_tokens"] = m_cold.get("n_tokens")
    R["n_residues"] = m_cold.get("n_residues")
    R["cold_plddt"] = m_cold.get("plddt")
    print(f"cold {t_cold:.2f}s tokens={R['n_tokens']} plddt={R['cold_plddt']:.4f}", flush=True)
    rec.clear()

    # --- arms ---------------------------------------------------------------------------
    folds = []
    for arm in R["arms"]:
        meta["job_cfg"]["trace"] = (arm == "traced")
        cap["s"] = 0.0
        cap["n"] = 0
        t, m = one_fold()
        r = rec[-1]
        folds.append({"arm": arm, "fold_s": round(t, 4), "stage_s": round(r["stage_s"], 4),
                      "n_step": r["n_step"], "trace_flag": r["trace"],
                      "plddt": m.get("plddt"), "capture_s": round(cap["s"], 4),
                      "captures": cap["n"]})
        print(f"  {arm}: fold {t:.3f}s stage {r['stage_s']:.3f}s "
              f"capture {cap['s']:.3f}s x{cap['n']} plddt {m.get('plddt'):.4f}", flush=True)
    R["folds"] = folds

    def h(t):
        return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()[:12]

    base = rec[0]["coords"]
    par = []
    for i, r in enumerate(rec):
        d = (base - r["coords"]).abs().max().item()
        par.append({"arm": R["arms"][i], "coords_hash": h(r["coords"]),
                    "bit_exact_vs_arm0": bool(torch.equal(base, r["coords"])),
                    "max_abs_delta_vs_arm0": d})
    R["parity"] = par
    for p in par:
        print("  parity:", p, flush=True)

    for tag in ("eager", "traced"):
        v = [f["stage_s"] for f in folds if f["arm"] == tag]
        w = [f["fold_s"] for f in folds if f["arm"] == tag]
        R[f"{tag}_stage_median_s"] = round(st.median(v), 4)
        R[f"{tag}_fold_median_s"] = round(st.median(w), 4)
    R["stage_delta_ms_per_fold"] = round(
        (R["eager_stage_median_s"] - R["traced_stage_median_s"]) * 1e3, 1)
    R["stage_ratio_eager_over_traced"] = round(
        R["eager_stage_median_s"] / R["traced_stage_median_s"], 4)
    R["capture_s_first_traced"] = next((f["capture_s"] for f in folds if f["arm"] == "traced"), None)
    R["net_ms_per_fold_incl_capture"] = round(
        R["stage_delta_ms_per_fold"] - (R["capture_s_first_traced"] or 0.0) * 1e3, 1)
    print(json.dumps({k: v for k, v in R.items() if k not in ("folds", "parity")}, indent=2),
          flush=True)
    out_p.write_text(json.dumps(R, indent=2))
    print("wrote", out_p, flush=True)
    state.reset()
    cleanup()


if __name__ == "__main__":
    main()
