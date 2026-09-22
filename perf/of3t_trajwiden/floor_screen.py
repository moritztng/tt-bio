#!/usr/bin/env python3
"""of3t-trajwiden deliverable 1: what a 20-step trajectory costs at each scope rung, measured
on ONE host at ONE thread count, before anything long is committed.

WHY THIS IS NOT A RE-READ OF CEILING.json. `of3t-trajwide`'s CEILING.json projected the
48-block float64 pairformer stack at 673.577 GB of activations from one block measured with
`blocks_per_ckpt = None` -- the projection's own words, "price the real backward, not a
recompute". `of3t-frame384` then RAN the same object WITH per-block checkpointing and read
24.881 GB and 1629.178 s (REF_F64_N384.json, qb1, 16 threads). The projection was 27.3x high
on memory. D194. Both numbers are correct about different programs, and the program a
trajectory would run is the checkpointed one, because `of3t-trunkg043`'s `ref_grad.py` proved
the flag inert: 2736/2736 bit-identical on both policies at crop 64, max absolute difference
exactly 0.0. So this screen prices the CHECKPOINTED object.

THE TWO TIME FIGURES ARE NOT DIFFERENCED. CEILING's 2,936 s was 2 threads on qb2; frame384's
1,629.178 s was qb1 at 16. Not thread-matched, so neither bounds the other. Everything below
is taken at `--threads` on this host, in this process, and frame384's 48-block reading is used
ONLY as a held-out validation point for the block extrapolation -- same host, same thread
count, same checkpointing, a rung this screen deliberately does not re-pay.

WHAT A RUNG COSTS. One trajectory step is one forward and one backward of the model's own
sample axis plus an optimizer step. The sample axis is 48 noise levels. The diffusion arm runs
the single branch over all 48 and the pair branch once, so a rung's per-step cost is measured
at TWO sample counts and the marginal cost per level is the slope -- an assumed multiplier is
what made SCOPE_LADDER's 5.51 h projection for `diffusion_module` disagree by 2.6x with the
3.44 h `of3t-trajwide` actually paid.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import socket
import hashlib
import subprocess
import sys
import time

_PERF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PERF not in sys.path:
    sys.path.append(_PERF)
import refpath                                                            # noqa: E402

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
#: The 0.4.3 diffusion boundary. `refpath.DIFFCAP` names the qb2 copy; this screen runs on
#: qb1, where the same capture sits under a CONCLUDED slug's directory. Pinned by content
#: below (`sha256.diffusion_boundary`), not by which path answered -- D153 is the trap of a
#: reference path that resolves to something else, and a digest is the only answer to it.
DIFFCAP = ("/home/ttuser/of3t_softgrad/diffcap043/diffusion_boundary.pt"
           if os.path.exists("/home/ttuser/of3t_softgrad/diffcap043/diffusion_boundary.pt")
           else "/home/ttuser/of3t_hostleg/diffcap043/diffusion_boundary.pt")
DIFFCAP_EXPECT = "ea80f18e651dae517e3cdc0563c8e6b27d9a4c6bd2e32d8066dd7c4c70b060e6"
FRAME = "/home/ttuser/of3t_frame384"
OUT = "perf/of3t_trajwiden/FLOOR.json"

#: the campaign's exhaustive section table, `perf/of3t_orchestrator/SECTION_MASS_MEASURED.json`
MODEL_SQ_NORM = 10.279642678524981
#: measured 20-step totals this screen's extrapolation has to reproduce, both on qb2
ANCHORS = {
    "diffusion_module.diffusion_conditioning": {
        "artifact": "perf/of3t_trajretake/traj_retake_shipped.json",
        "pct_of_model": 36.94617946669957, "steps": 20, "both_sides_s": 129.63047623634338,
        "host": "tt-quietbox2", "threads": None,
        "note": "one process, the two sides zipped, so this is the pair"},
    "diffusion_module": {
        "artifact": "perf/of3t_trajwiden/traj_widen_shipped.json",
        "pct_of_model": 88.08194359237523, "steps": 20,
        "device_s": 3335.8588473796844, "reference_s": 9056.806815862656,
        "both_sides_s": 12392.66566324234, "host": "tt-quietbox2", "threads": 3,
        "note": "sides decoupled, each its own process"},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--blocks", default="1,2,4,8",
                    help="the pairformer block ladder; 48 is frame384's held-out point")
    ap.add_argument("--only", default="", help="comma-separated rung names to run")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    only = set(x for x in a.only.split(",") if x)

    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(a.threads)
    refpath.install()
    refpath.require(CKPT, DIFFCAP, f"{FRAME}/boundary_n384.pt", f"{FRAME}/block47_boundary.pt")

    import torch
    torch.set_num_threads(a.threads)
    tree = refpath.assert_resolved()
    got = sha256(DIFFCAP)
    if got != DIFFCAP_EXPECT:
        raise SystemExit(f"{DIFFCAP} digests {got}, not the 0.4.3 diffusion boundary "
                         f"{DIFFCAP_EXPECT} every reference number in this campaign is "
                         f"taken at. See D153.")

    res = {
        "what": "the per-step cost of a 20-step trajectory at each scope rung, measured on one "
                "host at one thread count, with the extrapolation to 20 steps and its residual "
                "against the two rungs the campaign has already paid in full.",
        "host": socket.gethostname(),
        "host_detail": {"cpu": platform.processor(), "cores": os.cpu_count(),
                        "threads_used": a.threads,
                        "torch": torch.__version__, "python": sys.version.split()[0],
                        "loadavg_at_start": os.getloadavg()},
        "reference_tree": tree,
        "policy": "float64 on every parameter and every activation, checkpoint upcast at load",
        "crop": a.crop,
        "sha256": {"diffusion_boundary": got, "checkpoint": sha256(CKPT),
                   "boundary_n384": sha256(f"{FRAME}/boundary_n384.pt")},
        "diffusion_boundary_path": DIFFCAP,
        "diffusion_boundary_path_note":
            "the capture refpath names on qb2 under of3t_softgrad; this host carries the "
            "byte-identical copy under of3t_hostleg, a concluded slug's directory. The digest "
            "above is what binds it, not the path.",
        "model_sq_norm": MODEL_SQ_NORM,
        "anchors": ANCHORS,
        "rungs": {},
    }
    t0 = time.time()

    def emit():
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, default=str)

    def rung(name, fn):
        if only and name not in only:
            return
        row = {"rung": name}
        try:
            row.update(fn())
        except Exception as e:
            import traceback
            traceback.print_exc()
            row["refused"] = f"{type(e).__name__}: {e}"
        row["peak_rss_gb_process_so_far"] = rss_gb()
        res["rungs"][name] = row
        print(f"[{time.time()-t0:.0f}s] {name}: {json.dumps(row, default=str)}", flush=True)
        emit()

    print(f"[{time.time()-t0:.0f}s] tree {tree}, {a.threads} threads on "
          f"{res['host']}", flush=True)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    D = torch.load(DIFFCAP, map_location="cpu", weights_only=False)
    kw = D["kwargs"]
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
    cfg = OF3ProjectEntry().get_model_config_with_presets(presets=["train"])

    def f64(x):
        return x.to(torch.float64) if torch.is_tensor(x) and x.is_floating_point() else x

    def load_head(mod, head):
        mod.load_state_dict({k[len(head):]: f64(v) for k, v in sd.items()
                             if k.startswith(head)}, strict=False)
        return mod

    def time_fwd_bwd(mod, call_kwargs, reps=1):
        """One forward, one backward, timed separately. `reps` repeats the pair and returns the
        MEDIAN, because a single reading on a shared box is a draw from its contention."""
        fs, bs = [], []
        for _ in range(reps):
            t1 = time.time()
            out = mod(**call_kwargs)
            fs.append(time.time() - t1)
            first = out[0] if isinstance(out, (tuple, list)) else out
            t1 = time.time()
            torch.autograd.backward([first], [torch.ones_like(first)])
            bs.append(time.time() - t1)
            mod.zero_grad(set_to_none=True)
        fs.sort(); bs.sort()
        return fs[len(fs) // 2], bs[len(bs) // 2]

    # ---- the conditioning rung, at the granularity the trajectory runs it ------------------
    def r_cond():
        from openfold3.core.model.layers.diffusion_conditioning import DiffusionConditioning
        t1 = time.time()
        m = load_head(DiffusionConditioning(
            **dict(cfg.architecture.diffusion_module.diffusion_conditioning)).to(torch.float64),
            "diffusion_module.diffusion_conditioning.").train()
        build = time.time() - t1
        row = {"build_s": build, "n_parameters": sum(1 for p in m.parameters()
                                                     if p.requires_grad),
               "pct_of_model": 36.94617946669957, "per_level": {}}
        base = dict(batch=kw["batch"], si_input=f64(kw["si_input"]),
                    si_trunk=f64(kw["si_trunk"]), zij_trunk=f64(kw["zij_trunk"]),
                    use_conditioning=bool(kw["use_conditioning"]),
                    chunk_size=kw.get("chunk_size"))
        for n in (1, 12):
            f, b = time_fwd_bwd(m, dict(base, t=f64(kw["t"][:, :n])), reps=2)
            row["per_level"][n] = {"forward_s": f, "backward_s": b, "one_fwd_bwd_s": f + b}
        # 48 levels in 4 accumulation samples of 12: the model's own sample axis, partitioned.
        row["accumulation"] = {"samples_per_step": 4, "levels_per_sample": 12}
        row["ref_per_step_s"] = 4 * row["per_level"][12]["one_fwd_bwd_s"]
        row["ref_20_step_s"] = 20 * row["ref_per_step_s"]
        return row

    rung("diffusion_module.diffusion_conditioning", r_cond)

    # ---- the whole diffusion_module, at one and at twelve levels ---------------------------
    def r_diff():
        from openfold3.core.model.structure.diffusion_module import DiffusionModule
        t1 = time.time()
        m = load_head(DiffusionModule(config=cfg.architecture.diffusion_module)
                      .to(torch.float64), "diffusion_module.").train()
        build = time.time() - t1
        row = {"build_s": build, "n_parameters": sum(1 for p in m.parameters()
                                                     if p.requires_grad),
               "pct_of_model": 88.08194359237523,
               "pct_of_model_is": "the 573 of 761 tensors of3t-trajwide's device arm reaches, "
                                  "not the section's full 89.2106 %",
               "per_level": {}}
        base = {k: f64(v) for k, v in kw.items()}
        for n in (1, 2):
            kwv = dict(base)
            if torch.is_tensor(kwv.get("t")):
                kwv["t"] = kwv["t"][:, :n]
            for key in ("xl_noisy", "x_noisy", "xl", "x"):
                if torch.is_tensor(kwv.get(key)) and kwv[key].dim() >= 3:
                    kwv[key] = kwv[key][:, :n]
            f, b = time_fwd_bwd(m, kwv, reps=1)
            row["per_level"][n] = {"forward_s": f, "backward_s": b, "one_fwd_bwd_s": f + b}
        # The slope is the marginal cost of a noise level; the intercept is the per-call work
        # that does not scale with it. SCOPE_LADDER multiplied the n=1 reading by 48 and got
        # 5.51 h where the run paid 2.52 h, which is what an assumed multiplier costs.
        y1, y2 = (row["per_level"][n]["one_fwd_bwd_s"] for n in (1, 2))
        slope, icept = y2 - y1, y1 - (y2 - y1)
        row["fit"] = {"marginal_s_per_noise_level": slope, "fixed_s_per_call": icept,
                      "from": "two readings, n = 1 and n = 2"}
        row["accumulation"] = {"samples_per_step": 4, "levels_per_sample": 12}
        row["ref_per_step_s"] = 4 * (icept + 12 * slope)
        row["ref_20_step_s"] = 20 * row["ref_per_step_s"]
        return row

    rung("diffusion_module", r_diff)

    # ---- the pairformer trunk, checkpointed, as a block ladder -----------------------------
    def r_pf():
        from openfold3.core.model.latent.pairformer import PairFormerStack
        B = torch.load(f"{FRAME}/boundary_n384.pt", map_location="cpu", weights_only=False)
        row = {"blocks": {}, "pct_of_model": 5.8282,
               "boundary": f"{FRAME}/boundary_n384.pt",
               "checkpointed": True,
               "checkpointing_is_inert": "of3t-trunkg043 ref_grad.py: 2736/2736 bit-identical "
                                         "on both policies at crop 64, max abs diff 0.0"}
        s, z = f64(B["s_in"]), f64(B["z_in"])
        sm, pm = f64(B["single_mask"]), f64(B["pair_mask"])
        row["shapes"] = {"s": list(s.shape), "z": list(z.shape),
                         "single_mask": list(sm.shape), "pair_mask": list(pm.shape)}
        for nb in [int(x) for x in a.blocks.split(",")]:
            c = dict(cfg.architecture.pairformer)
            c["no_blocks"] = nb
            m = PairFormerStack(**c).to(torch.float64).train()
            m.blocks_per_ckpt = 1                 # the program a trajectory would run
            m.load_state_dict({k[len("pairformer_stack."):]: f64(v) for k, v in sd.items()
                               if k.startswith("pairformer_stack.")}, strict=False)
            t1 = time.time()
            f, b = time_fwd_bwd(m, dict(s=s.clone(), z=z.clone(), single_mask=sm,
                                        pair_mask=pm), reps=1)
            row["blocks"][nb] = {"forward_s": f, "backward_s": b, "one_fwd_bwd_s": f + b,
                                 "wall_s": time.time() - t1, "peak_rss_gb": rss_gb()}
            print(f"[{time.time()-t0:.0f}s]   pairformer {nb} blocks: "
                  f"{f + b:.2f} s, rss {rss_gb():.2f} GB", flush=True)
            del m
            res["rungs"].setdefault("pairformer_stack", row)
            emit()
        xs = sorted(row["blocks"])
        ys = [row["blocks"][k]["one_fwd_bwd_s"] for k in xs]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        icept = my - slope * mx
        ss_t = sum((y - my) ** 2 for y in ys)
        ss_r = sum((y - (slope * x + icept)) ** 2 for x, y in zip(xs, ys))
        pred48 = slope * 48 + icept
        held = json.load(open(f"{FRAME}/REF_F64_N384.json"))
        row["fit"] = {"over_blocks": xs, "slope_s_per_block": slope, "intercept_s": icept,
                      "r2": (1 - ss_r / ss_t) if ss_t else None,
                      "predicted_48_block_one_fwd_bwd_s": pred48}
        row["held_out_48_block_point"] = {
            "artifact": "perf/of3t_frame384/REF_F64_N384.json",
            "measured_one_fwd_bwd_s": held["seconds_forward_backward"],
            "threads": held["threads"], "peak_rss_gb": held["peak_rss_gb"],
            "host": "tt-quietbox (qb1), per of3t_frame384/FRAME_N384.json hosts field",
            "residual_s": pred48 - held["seconds_forward_backward"],
            "residual_rel": (pred48 - held["seconds_forward_backward"])
            / held["seconds_forward_backward"],
            "why_held_out": "same host, same thread count, same checkpointing. Re-paying it "
                            "costs 27 min and moves no verdict; using it as the validation "
                            "point is what makes the ladder an extrapolation with a residual "
                            "rather than an assertion."}
        # The trunk sits in front of the diffusion module, so it runs ONCE per accumulation
        # sample, not once per noise level.
        row["accumulation"] = {"samples_per_step": 4,
                               "note": "the pair branch runs on sample 0 alone in the harness's "
                                       "partition, so the per-step multiplier is 1, and 4 is "
                                       "quoted beside it as the cost if every sample ran it"}
        m1 = held["seconds_forward_backward"]
        row["ref_per_step_s"] = m1
        row["ref_20_step_s"] = 20 * m1
        row["ref_20_step_s_if_every_sample"] = 80 * m1
        return row

    rung("pairformer_stack", r_pf)
    res["elapsed_s"] = time.time() - t0
    emit()
    print(f"[{time.time()-t0:.0f}s] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
