#!/usr/bin/env python3
"""of3t-refatom, deliverable 1: the cheap discovery check, BEFORE the 0.93 h device arm.

`NEXT_ACTION_REFATOM.json` was written because the obvious version of this change fails
SILENTLY and in the flattering direction. Routing `trajwide`'s diffusion atom encoder through
`openfold3_host_prep.ref_atom_embed_device` looks like a one-line swap; that function builds
`RefAtomFeatureEmbedder` inside itself, calls it once and returns tensors, so the module is
transient. Then either half of the change is fatal on its own:

  * `weights_for(fwd, None, model=composed)` discovers by `walk_device_weights(composed)`, and
    a module no attribute of `composed` holds is not reachable by a walk -- the eight weights
    are simply absent from the parameter set, with no error;
  * the call sits OUTSIDE the taped per-step `fwd()`, so no cotangent reaches them on any step
    and the optimizer steps eight weights whose gradient is permanently `None`.

Both failures RAISE THE SCORE. A 20-step run with eight silently frozen weights dumps 581
tensors where `of3t-trajwide` dumped 573, so the coupled scope percentage goes UP while the
trajectory gets less honest. That is the number this campaign must not publish, and it costs
0.93 h of card time to produce. This check costs one noise level.

WHAT IT RUNS. `trajwide.build_ours(..., refatom="device")` and `trajwide.run_ours` unchanged,
over a one-level, one-step partition. It is the shipped program of the real arm, not a replica
of it: the only thing this file adds is a read-only snapshot of `AdamW.participation` taken
before `step()` clears it, because participation is incremented per parameter exactly when
`t.grad is not None` and that is assertion 2 stated in the optimizer's own terms.

THE FOUR BARS ARE FIXED HERE, BEFORE ANY NUMBER EXISTS (see `BARS`). Assertion 4 is the one
that catches a change of FUNCTION rather than of PLACE, so it is not a precision gate: cosine
and norm ratio decide it and the rel L2 is reported beside the host's own float32/float64
floor. TT fp32 is not IEEE fp32 (memory `tenstorrent-fp32-is-not-ieee-fp32`), so a bar at the
IEEE floor would refuse a correct device linear.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time

# --- the bars, fixed before the run ---------------------------------------------------------
BARS = {
    "n_refatom_params": 8,
    "grad_resolved_after_one_backward": 8,
    "tape_resolves_after_step": 980 + 8,
    "of_walked": 980 + 8,
    # Assertion 4: PLACE moved, FUNCTION unchanged. Cosine and norm ratio are the deciders;
    # rel L2 is a precision reading and its bar is loose on purpose.
    "function_cos_min": 0.99999,
    "function_norm_ratio_tol": 1.0e-3,
    "function_rel_l2_max": 5.0e-3,
}
OUT = "perf/of3t_refatom/CHECK.json"
SCRATCH = "/home/ttuser/of3t_runs/refatom_check"


def load(name, path):
    d = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def aiclk():
    """The raw AICLK register per device, hex as tt-smi reports it. 0x320 = 800, 0x546 = 1350."""
    import re
    try:
        s = subprocess.run([os.path.expanduser("~/.local/bin/tt-smi"), "-s"],
                           capture_output=True, text=True, timeout=120).stdout
    except Exception as e:
        return {"sampled": False, "why": repr(e)}
    return {"sampled": True, "at_utc": time.strftime("%FT%TZ", time.gmtime()),
            "aiclk_mhz_per_device": [int(x) for x in re.findall(r'"aiclk":\s*"(\d+)"', s)],
            "aiclk_reg_per_device": re.findall(r'"AICLK":\s*"(0x[0-9a-fA-F]+)"', s)}


def rel_stats(dev_t, ref_t):
    """rel L2, cosine and norm ratio of a device tensor against a host reference, in float64."""
    import torch
    a = dev_t.double().reshape(-1)
    b = ref_t.double().reshape(-1)
    na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    return {"rel_l2": float(torch.linalg.vector_norm(a - b) / (nb + 1e-300)),
            "cos": float((a @ b) / (na * nb + 1e-300)),
            "norm_ratio": float(na / (nb + 1e-300)),
            "n": int(a.numel())}


def main() -> int:
    t_start = time.time()
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    os.environ.setdefault("MKL_NUM_THREADS", "3")
    tw = load("trajwide", "perf/of3t_trajwide/trajwide.py")
    tw.refpath.install()

    import numpy as np
    import torch
    import ttnn
    from tt_bio.train.optim import AdamW
    from tt_bio import openfold3_host_prep as HP
    from tt_bio.openfold3_weights import _sub

    clk0 = aiclk()

    # --- the prize, re-derived from the reference gradient rather than quoted ---------------
    D = torch.load(tw.DIFFCAP, map_location="cpu", weights_only=False)
    g = D["grad_f64"]
    eight = sorted(n for n in g if "ref_atom_feature_embedder" in n)
    prize = {
        "names": eight,
        "n": len(eight),
        "pct_of_model_sq_grad_norm": 100.0 * sum(
            float(g[n].double().pow(2).sum()) for n in eight) / tw.MODEL_SQ_NORM,
        "model_sq_norm": tw.MODEL_SQ_NORM,
        "per_tensor_pct": {n: 100.0 * float(g[n].double().pow(2).sum()) / tw.MODEL_SQ_NORM
                           for n in eight},
    }

    # --- participation snapshot: read-only, taken before step() clears it -------------------
    snap = {}
    real_step = AdamW.step

    def step_with_snapshot(self, *a, **k):
        snap.update(dict(self.participation))
        return real_step(self, *a, **k)

    AdamW.step = step_with_snapshot
    try:
        kw, cot = D["kwargs"], D["cot"]
        act = ttnn.float32
        G = tw.build_ours(torch.load(tw.CAP, map_location="cpu", weights_only=False), kw, act,
                          "refatom", refatom="device")
        G["out_ref"] = D["out"]

        # --- assertion 4: the held embedder's own output against the HOST leg ---------------
        # `build_ours` already called the held module once to shape `aux`; that tensor is the
        # same function the tape will run, so it is the right thing to compare. The host leg
        # is run here at both widths, so the device reading sits beside the host's own floor.
        batch = kw["batch"]
        sq = lambda x: (x.reshape(x.shape[2:]) if x.dim() > 2 and x.shape[0] == 1
                        and x.shape[1] == 1 else x.squeeze(0))
        feats = {k: sq(batch[k]).float() for k in
                 ("ref_pos", "ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars",
                  "ref_space_uid")}
        feats["atom_mask"] = sq(batch["atom_mask"]).float()
        sd = torch.load(tw.CKPT, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
        rafe_sd = _sub(_sub(_sub(sd, "diffusion_module"), "atom_attn_enc"),
                       "ref_atom_feature_embedder")
        cl_h32, plm_h32 = HP.ref_atom_embed(rafe_sd, feats)
        cl_h64, plm_h64 = HP.ref_atom_embed(
            {k: v.double() for k, v in rafe_sd.items()},
            {k: v.double() for k, v in feats.items()})
        n_atom = G["n_atom"]
        cl_d = ttnn.to_torch(G["aux"]["cl0_d"])[0, :n_atom]
        plm_d = ttnn.to_torch(G["aux"]["plm0_d"])[0]
        function = {
            "cl_device_vs_host_f64": rel_stats(cl_d, cl_h64),
            "cl_device_vs_host_f32": rel_stats(cl_d, cl_h32),
            "cl_host_f32_vs_host_f64": rel_stats(cl_h32, cl_h64),
            "plm_device_vs_host_f64": rel_stats(plm_d, plm_h64),
            "plm_device_vs_host_f32": rel_stats(plm_d, plm_h32),
            "plm_host_f32_vs_host_f64": rel_stats(plm_h32, plm_h64),
            "note": "the device leg pads the atom axis to NP by padding its own inputs; all "
                    "eight linears are bias-free, so the pad rows are zero either way and the "
                    "comparison is over the first n_atom rows",
            "n_atom": n_atom, "NP": G["NP"],
        }

        # --- assertions 1-3: one level, one step, the shipped run_ours ----------------------
        log = []
        os.makedirs(SCRATCH, exist_ok=True)
        tw.run_ours(G, {1: [[0]]}, cot, steps=1, warmup=tw.SCHED["warmup_no_steps"],
                    log=log, d_out=SCRATCH, brk="none", zero_grad_model=False)
        ev = tw.run_ours.last
    finally:
        AdamW.step = real_step

    clk1 = aiclk()
    got_grad = sorted(n for n in snap if "ref_atom_feature_embedder" in n)
    row = log[-1]
    f = function
    checks = {
        "1_eight_in_parameter_set": {
            "want": BARS["n_refatom_params"], "got": ev["n_refatom_params"],
            "paths": ev["refatom_params"], "checkpoint_names": ev["refatom_names"],
            "pass": ev["n_refatom_params"] == BARS["n_refatom_params"]},
        "2_each_resolves_a_gradient": {
            "want": BARS["grad_resolved_after_one_backward"], "got": len(got_grad),
            "participating": got_grad,
            "participation_total": len(snap), "walked_total": row["of_walked"],
            "every_walked_parameter_participated": len(snap) == row["of_walked"],
            "pass": len(got_grad) == BARS["grad_resolved_after_one_backward"]},
        "3_tape_resolves_after_step": {
            "want": BARS["tape_resolves_after_step"], "got": row["tape_resolves_after_step"],
            "of_walked_want": BARS["of_walked"], "of_walked_got": row["of_walked"],
            "rebound": row["rebound"],
            "pass": (row["tape_resolves_after_step"] == BARS["tape_resolves_after_step"]
                     and row["of_walked"] == BARS["of_walked"])},
        "4_place_moved_not_function": {
            "bars": {k: BARS[k] for k in
                     ("function_cos_min", "function_norm_ratio_tol", "function_rel_l2_max")},
            "pass": all(
                f[k]["cos"] >= BARS["function_cos_min"]
                and abs(f[k]["norm_ratio"] - 1.0) <= BARS["function_norm_ratio_tol"]
                and f[k]["rel_l2"] <= BARS["function_rel_l2_max"]
                for k in ("cl_device_vs_host_f64", "plm_device_vs_host_f64"))},
    }
    res = {
        "what": "of3t-refatom deliverable 1: the discovery check, run BEFORE the 0.93 h "
                "device arm. Four assertions from NEXT_ACTION_REFATOM.json, bars fixed in "
                "this file before the run.",
        "bars_fixed_before_the_run": BARS,
        "verdict": "PASS" if all(c["pass"] for c in checks.values()) else "FAIL",
        "checks": checks,
        "function": function,
        "the_prize_rederived": prize,
        "step_row": row,
        "evidence": ev,
        "host": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
        "card": {"TT_VISIBLE_DEVICES": os.environ.get("TT_VISIBLE_DEVICES"),
                 "TT_BIO_LEASE_CARDS": os.environ.get("TT_BIO_LEASE_CARDS")},
        "aiclk_before": clk0, "aiclk_after": clk1,
        "aiclk_note": "sampled immediately before and after the taped level on this host. The "
                      "check makes no throughput claim, so no number here is divided by a "
                      "clock; the reading is recorded for attribution.",
        "reference_tree": tw.OF3PKG,
        "sha256": {"diffusion_boundary": tw.sha256(tw.DIFFCAP),
                   "cond_boundary": tw.sha256(tw.CAP),
                   "checkpoint": tw.sha256(tw.CKPT)},
        "wall_s": time.time() - t_start,
        "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "evidence"}, indent=1,
                     default=str)[:6000])
    print(f"wrote {OUT}: {res['verdict']}", flush=True)
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
