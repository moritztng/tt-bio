#!/usr/bin/env python3
"""Boltz-2 320 aa fold A/B for the two L1-path layout deletions.

320 tokens, not 512: both levers only fire where the trimul takes the L1 path and
`reblock_permute`'s L1 window serves it, which is 288 <= N <= 352. `token_axis.TOKEN_BUCKET` is
32, so this is the cell a 289-320 residue target actually folds at. At 512 aa neither lever fires
and the fold is byte-identical -- measured at the op level, and re-checked here by the CIF digest.

Arms, interleaved rep by rep in one process on one device open, each fold through the production
``_WorkerState.predict_one``:

  A   both levers off -- today's shipped behaviour on the L1 path
  B   both on: the fused forward move (`TRIMUL_GATED_MOVE_L1`) and the one-pass output move
      (`TRIMUL_BACK_ONE_PASS_L1`)
  A2  arm A again, as this session's own A/A floor

Approved as a STACK or not at all: perturbations at this site are strongly sub-additive and the
op-level screen already shows it (1.0767 x 1.0181 = 1.0962 predicted against 1.0969 measured).

The op-level screen is `perf/trix_layout/l1_levers_qb1c0.json`. This is the number that decides:
this campaign's history is four op-level levers reaching the fold at 25x-to-infinite error with
two flipping sign.

Structure, stage splitter and per-fold CIF hash are `perf/b2x_trimul/fold_ab_trimul_512.py`'s,
with the AICLK sampled DURING every timed phase added -- a number without a clock is not a
measurement on this hardware.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

# Must be set before anything can open the device: get_device reads it at open.
os.environ.setdefault("TT_BIO_TRACE_REGION_SIZE", str(1 << 30))

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
FIX = REPO / "perf" / "size512" / "fixtures"
sys.path.insert(0, str(REPO / "perf"))
from clocksample import during  # noqa: E402

N_RES = 320

RECYCLING_STEPS = 3       # the published cell's protocol
SAMPLING_STEPS = 200
DIFFUSION_SAMPLES = 1
SEED = 0

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def _seed_msa(target: Path, a3m_text: str, msa_dir: Path) -> None:
    from tt_bio.main import _read_bio_chains
    chains = _read_bio_chains(target)
    assert len(chains) == 1, f"{target} is not a monomer: {len(chains)} chains"
    seq = chains[0][1]
    rows = a3m_text.split("\n")
    assert rows[1] == seq, "a3m query row does not match the target sequence"
    msa_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(seq.encode()).hexdigest()[:16]
    (msa_dir / f"{h}.a3m").write_text(a3m_text)


def main() -> int:
    global SAMPLING_STEPS, RECYCLING_STEPS, OUT_PATH, N_RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=5, help="warm folds per arm in phase 1")
    ap.add_argument("--module-census", action="store_true",
                    help="one EXTRA fold with the trimul sync-bracketed, for the "
                         "in-fold module wall; never one of the timed arms")
    ap.add_argument("--arms", default="A,B,A2")
    ap.add_argument("--n", type=int, default=N_RES)
    ap.add_argument("--steps", type=int, default=SAMPLING_STEPS,
                    help="sampling steps; only lower it for a harness smoke test")
    ap.add_argument("--recycles", type=int, default=RECYCLING_STEPS)
    args = ap.parse_args()
    SAMPLING_STEPS = args.steps
    N_RES = args.n
    RECYCLING_STEPS = args.recycles

    OUT_PATH = args.out

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)

    dev = get_device()
    try:
        grid = dev.compute_with_storage_grid_size()
        grid = [grid.x, grid.y]
    except Exception:
        grid = None
    import socket
    OUT["env"] = {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "lease_cards": os.environ.get("TT_BIO_LEASE_CARDS"),
        "grid": grid,
        "arch": str(getattr(dev, "arch", lambda: "?")()),
        "torch": torch.__version__,
        "trace_region_bytes": int(os.environ["TT_BIO_TRACE_REGION_SIZE"]),
        "protocol": {"n_residues": N_RES, "recycling_steps": RECYCLING_STEPS,
                     "sampling_steps": SAMPLING_STEPS,
                     "diffusion_samples": DIFFUSION_SAMPLES, "seed": SEED,
                     "fixture": f"perf/size512/fixtures/cdk2x2_{N_RES}.yaml + a3m"},
    }
    try:
        import importlib.metadata as _md
        OUT["env"]["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    dump()

    work = Path(tempfile.mkdtemp(prefix="bioir-dispatch-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    target = FIX / f"cdk2x2_{N_RES}.yaml"
    _seed_msa(target, (FIX / f"cdk2x2_{N_RES}.a3m").read_text(), msa_dir)

    cfg = dict(
        model="boltz2", fast=False, output_format="cif",
        recycling_steps=RECYCLING_STEPS, sampling_steps=SAMPLING_STEPS,
        diffusion_samples=DIFFUSION_SAMPLES, seed=SEED, trace=False,
        msa_dir=str(msa_dir), struct_dir=str(struct_dir),
        use_msa_server=False, msa_db_path=None, use_envdb=False, msa_endpoint=None,
        single_sequence=False, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", msa_server_username=None,
        msa_server_password=None, api_key_value=None, max_msa_seqs=8192,
        write_pae=False, write_pde=False, write_embeddings=False, method=None,
        conf_kwargs=dict(
            predict_args={"recycling_steps": RECYCLING_STEPS,
                          "sampling_steps": SAMPLING_STEPS,
                          "diffusion_samples": DIFFUSION_SAMPLES,
                          "max_parallel_samples": 5},
            diffusion_process_args={
                "step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0,
                "noise_scale": 1.003, "rho": 7, "sigma_min": 0.0001,
                "sigma_max": 160.0, "sigma_data": 16.0, "P_mean": -1.2,
                "P_std": 1.5, "coordinate_augmentation": True,
                "alignment_reverse_diff": True, "synchronize_sigmas": True},
            pairformer_args={"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True},
            msa_args={"subsample_msa": False, "num_subsampled_msa": 1024,
                      "use_paired_feature": True, "msa_s": 64, "msa_blocks": 4,
                      "msa_dropout": 0.15, "z_dropout": 0.25,
                      "pairwise_head_width": 32, "pairwise_num_heads": 4,
                      "activation_checkpointing": True},
            steering_args={"fk_steering": False, "physical_guidance_update": False,
                           "contact_guidance_update": True, "num_particles": 3,
                           "fk_lambda": 4.0, "fk_resampling_interval": 3,
                           "num_gd_steps": 20},
            use_kernels=True, use_tenstorrent=True, trace=False,
            diffusion_trace=True,          # reserves the trace region; per-fold arm is the flag below
        ),
    )
    _ensure_local_artifacts(cfg)

    t_load = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("bioir-dispatch-graph", cfg)
    OUT["model_load_s"] = round(time.perf_counter() - t_load, 3)
    dump()

    score_model = state.model.structure_module.score_model

    # capture cost, priced separately from the stage it lands in
    cap = {"n": 0, "s": 0.0}
    _orig_capture = score_model._capture_diff_trace

    def _timed_capture(*a, **k):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = _orig_capture(*a, **k)
        ttnn.synchronize_device(dev)
        cap["n"] += 1
        cap["s"] += time.perf_counter() - t0
        return r
    score_model._capture_diff_trace = _timed_capture

    # ---- stage splitter -----------------------------------------------------
    # Sync ONLY at the three stage transitions, all of them outside the 200-step
    # loop, so the 200 in-loop progress emits stay free.
    class Splitter:
        def __init__(self, hook=None):
            self.hook = hook
            self.reset()

        def reset(self):
            self.marks = []      # (label, t)
            self.loop = []       # (step, t) -- NO sync, in-loop progress emits stay free
            self.n_diff = 0
            self.last = None

        def _mark(self, label):
            ttnn.synchronize_device(dev)
            self.marks.append((label, time.perf_counter()))

        def __call__(self, stage=None, step=0, total=0, *a, **k):
            if stage == "diffusion":
                self.n_diff += 1
                if self.n_diff == 1:
                    self._mark("diffusion_conditioning")   # trunk ends here
                elif self.n_diff == 2:
                    self._mark("sampler")                  # conditioning ends here
                    self.loop.append((step, self.marks[-1][1]))
                else:
                    self.loop.append((step, time.perf_counter()))
            elif stage == "confidence":
                self._mark("confidence")                   # sampler ends here
                if self.hook:
                    self.hook()

        def close(self):
            self._mark("end")
            out = {}
            for (lab, t0), (_l2, t1) in zip(self.marks, self.marks[1:]):
                out[lab] = round(t1 - t0, 4)
            return out

    _TRACE_DEFAULT = state.model.structure_module._diffusion_trace

    def fold(armname, hook=None):
        state.model.structure_module._diffusion_trace = _TRACE_DEFAULT
        sp = Splitter(hook)
        state.pfn = sp
        state.model.progress_fn = sp
        for p in struct_dir.glob("*"):
            p.unlink()
        cap["n"] = 0; cap["s"] = 0.0
        k0 = kernels()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
        stages = sp.close()
        loop = [(k, round(t - sp.loop[0][1], 4)) for k, t in sp.loop]
        # marks[0] is the trunk->conditioning boundary, so everything before it is
        # featurisation + trunk; the fold wall minus the recorded stages is the rest.
        pre = sp.marks[0][1] - t0
        cifs = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
                for f in sorted(struct_dir.glob("*.cif"))}
        assert cifs, "no CIF written"
        k1 = kernels()
        return {
            "arm": armname,
            "kernel_calls": {k: k1[k] - k0[k] for k in k1},
            "trace": _TRACE_DEFAULT,
            "fold_s": round(wall, 3),
            "prepare_and_trunk_s": round(pre, 4),
            "stages_s": stages,
            "capture_n": cap["n"], "capture_s": round(cap["s"], 4),
            "loop_marks": loop,
            "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
            "cif": cifs,
        }

    # ---- the timed A/B ------------------------------------------------------
    from tt_bio import tenstorrent as _T
    from tt_bio import reblock_permute as _RB

    # Read before any arm touches them: "the branch defaults them on" is part of the claim.
    OUT["defaults"] = {"trimul_gated_move_l1": _T._TRIMUL_GATED_MOVE_L1,
                       "trimul_back_one_pass_l1": _T._TRIMUL_BACK_ONE_PASS_L1}
    dump()

    ARMS = args.arms.split(",")

    def set_arm(name):
        on = name == "B"
        _T.set_trimul_gated_move_l1(on)
        _T.set_trimul_back_one_pass_l1(on)

    # A dead flag and a correct one write the same CIF, so count the kernels each arm reaches.
    def kernels():
        return {"gated": int(_RB.STATS_GATED[0]), "back": int(_RB.STATS_BACK[0]),
                "fwd": int(_RB.STATS[0])}

    runs = []
    with during(period=2.0) as clk:
        print("[phase1] cold fold (discarded)", flush=True)
        set_arm("A")
        r = fold("A"); r["cold"] = True; runs.append(r)
        print("  cold", r["fold_s"], "s", flush=True)
        OUT["runs"] = runs; dump()
        for i in range(args.reps):
            for nm in ARMS:
                set_arm(nm)
                r = fold(nm); r["cold"] = False; r["rep"] = i
                runs.append(r)
                print("  rep %d %-2s %8.3f s  plddt %s  kernels %s  cif %s"
                      % (i, nm, r["fold_s"], r["plddt"], r["kernel_calls"],
                         list(r["cif"].values())), flush=True)
                OUT["runs"] = runs; dump()
    OUT["clock"] = clk.summary()
    print(clk.line(0), flush=True)

    # ---- the in-fold module wall, asked for by the campaign brief -----------------------
    # A SEPARATE fold, never one of the timed arms: the bracket around each trimul call
    # serialises the pipeline and inflates the fold itself. Per call the inflation is the
    # 0.05142 ms host launch/drain floor `state/trix/FIXTERM-HANDOFF.md` measured on this part,
    # which is 0.5 % of an ~11 ms call, so the module wall is an upper bound and a tight one.
    if args.module_census:
        set_arm("B")
        orig_call = _T.TriangleMultiplication.__call__
        acc = {"n": 0, "ms": 0.0, "by_shape": {}}

        def _timed_call(self, x, *a, **k):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            r = orig_call(self, x, *a, **k)
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) * 1e3
            acc["n"] += 1
            acc["ms"] += dt
            key = "x".join(str(int(d)) for d in x.shape)
            e = acc["by_shape"].setdefault(key, [0, 0.0])
            e[0] += 1
            e[1] += dt
            return r
        _T.TriangleMultiplication.__call__ = _timed_call
        try:
            cr = fold("module_census")
        finally:
            _T.TriangleMultiplication.__call__ = orig_call
        OUT["in_fold_module"] = {
            "arm": "B", "fold_s_INFLATED": cr["fold_s"], "calls": acc["n"],
            "module_s": round(acc["ms"] / 1e3, 4),
            "ms_per_call": round(acc["ms"] / max(1, acc["n"]), 4),
            "bracket_floor_ms_per_call": 0.05142,
            "ms_per_call_bracket_corrected": round(acc["ms"] / max(1, acc["n"]) - 0.05142, 4),
            "share_of_fold": round(acc["ms"] / 1e3 / cr["fold_s"], 4),
            "by_shape": {k: {"calls": v[0], "ms": round(v[1], 2),
                             "ms_per_call": round(v[1] / v[0], 4)}
                         for k, v in sorted(acc["by_shape"].items())},
        }
        print("in-fold module: %d calls, %.3f s, %.4f ms/call (%.4f bracket-corrected), "
              "%.1f %% of a %.3f s fold that the brackets themselves inflated"
              % (acc["n"], acc["ms"] / 1e3, acc["ms"] / acc["n"],
                 acc["ms"] / acc["n"] - 0.05142,
                 100 * acc["ms"] / 1e3 / cr["fold_s"], cr["fold_s"]), flush=True)
        for k, v in OUT["in_fold_module"]["by_shape"].items():
            print("    %-22s %4d calls  %8.3f ms  %.4f ms/call"
                  % (k, v["calls"], v["ms"], v["ms_per_call"]), flush=True)
        dump()

    warm = [r for r in runs if not r["cold"]]
    med = {nm: round(st.median([r["fold_s"] for r in warm if r["arm"] == nm]), 4) for nm in ARMS}
    cifs = {nm: sorted({tuple(sorted(r["cif"].items())) for r in warm if r["arm"] == nm})
            for nm in ARMS}
    OUT["median_fold_s"] = med
    OUT["AA_floor_pct"] = round(100 * abs(med["A2"] - med["A"]) / med["A"], 3)
    OUT["kernel_calls"] = {nm: sorted({tuple(sorted(r["kernel_calls"].items()))
                                       for r in warm if r["arm"] == nm}) for nm in ARMS}
    OUT["x_vs_A"] = {nm: round(med["A"] / v, 4) for nm, v in med.items()}
    OUT["cif_identical_to_A"] = {nm: cifs[nm] == cifs["A"] for nm in ARMS}
    OUT["plddt"] = {nm: [r["plddt"] for r in warm if r["arm"] == nm] for nm in ARMS}
    dump()
    print(json.dumps(med), flush=True)
    print("A/A floor %%: %s" % OUT["AA_floor_pct"], flush=True)
    print(json.dumps(OUT["x_vs_A"]), flush=True)
    print("CIF identical to A:", json.dumps(OUT["cif_identical_to_A"]), flush=True)
    print("kernel calls per arm:", json.dumps(OUT["kernel_calls"], default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
