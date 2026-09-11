#!/usr/bin/env python3
"""Boltz-2 512 aa fold A/B for the trimul fusion unlock (P7).

Arms, interleaved in one process on one device open, each fold through the production
``_WorkerState.predict_one``:

  A   shipped default
  B   step 1: the pair mask applied on the far side of the channel move, which makes E6
      (`reblock_permute_gated`) eligible on every pairformer trimul
  C   step 1 + step 3: F1 (`dual_gemm_x0_x1`) allowed at the (4, 4) block key, which is
      boltz2's tail weight at c_z = 128 and which F1 declines today
  A2  the shipped default again, as this session's own A/A floor
  DEF whatever the module ships as its default, with no A/B switch touched at all, so the
      landed default is measured rather than the lever

Step 2 is not an arm: it is a measured loss at the block level (perf/b2x_trimul/step2_512_qb2c2.json).

Co-tenanted by design -- card 0 has been running a 44-leg parity gate all afternoon. The ratio
between interleaved arms survives that; the absolute fold second does not, and is not claimed here.
Structure of the harness, the stage splitter and the per-fold CIF hash are the template's
(perf/bioir_dispatch/trace_ab_boltz2_512.py), unchanged.
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

RECYCLING_STEPS = 3       # the published 512 aa cell's protocol, and arm A's on the H200
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
    global SAMPLING_STEPS, RECYCLING_STEPS, OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3, help="warm folds per arm in phase 1")
    ap.add_argument("--arms", default="A,B,C,A2")
    ap.add_argument("--steps", type=int, default=SAMPLING_STEPS,
                    help="sampling steps; only lower it for a harness smoke test")
    ap.add_argument("--recycles", type=int, default=RECYCLING_STEPS)
    args = ap.parse_args()
    SAMPLING_STEPS = args.steps
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
        "protocol": {"n_residues": 512, "recycling_steps": RECYCLING_STEPS,
                     "sampling_steps": SAMPLING_STEPS,
                     "diffusion_samples": DIFFUSION_SAMPLES, "seed": SEED,
                     "fixture": "perf/size512/fixtures/cdk2x2_512.yaml + 35-row a3m"},
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
    target = FIX / "cdk2x2_512.yaml"
    _seed_msa(target, (FIX / "cdk2x2_512.a3m").read_text(), msa_dir)

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
        return {
            "arm": armname,
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
    import tt_bio.trimul_tail as _F1
    from tt_bio import tenstorrent as _T

    # Read before any arm has touched them, so DEF restores what the module ships rather than
    # what this harness picked. Recorded, because "the default is on" is the claim under test.
    DEF_MASK_AFTER_MOVE = _T._TRIMUL_MASK_AFTER_MOVE
    DEF_F1_CZ128 = (4, 4) in _F1.F1_BLOCK_KEYS
    OUT["defaults"] = {"trimul_mask_after_move": DEF_MASK_AFTER_MOVE,
                       "trimul_tail_f1_cz128": DEF_F1_CZ128}
    dump()

    ARMS = args.arms.split(",")

    def set_arm(name):
        if name == "DEF":
            _T.set_trimul_mask_after_move(DEF_MASK_AFTER_MOVE)
            _F1.set_f1_cz128(DEF_F1_CZ128)
            return
        _T.set_trimul_mask_after_move(name in ("B", "C"))
        _F1.set_f1_cz128(name == "C")

    runs = []
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
            print("  rep %d %-2s %8.3f s  plddt %s  cif %s"
                  % (i, nm, r["fold_s"], r["plddt"], list(r["cif"].values())), flush=True)
            OUT["runs"] = runs; dump()
    set_arm("DEF")

    warm = [r for r in runs if not r["cold"]]
    med = {nm: round(st.median([r["fold_s"] for r in warm if r["arm"] == nm]), 4) for nm in ARMS}
    cifs = {nm: sorted({tuple(sorted(r["cif"].items())) for r in warm if r["arm"] == nm})
            for nm in ARMS}
    OUT["median_fold_s"] = med
    OUT["AA_floor_pct"] = round(100 * (med["A2"] - med["A"]) / med["A"], 3)
    OUT["x_vs_A"] = {nm: round(med["A"] / v, 4) for nm, v in med.items()}
    OUT["cif_identical_to_A"] = {nm: cifs[nm] == cifs["A"] for nm in ARMS}
    OUT["plddt"] = {nm: [r["plddt"] for r in warm if r["arm"] == nm] for nm in ARMS}
    dump()
    print(json.dumps(med), flush=True)
    print("A/A floor %%: %s" % OUT["AA_floor_pct"], flush=True)
    print(json.dumps(OUT["x_vs_A"]), flush=True)
    print("CIF identical to A:", json.dumps(OUT["cif_identical_to_A"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
