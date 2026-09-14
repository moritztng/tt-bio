#!/usr/bin/env python3
"""Boltz-2 512 aa fold A/B for the triangle product's L1 destination (catalogue row B1).

Arms, interleaved in one process on one device open, each fold through the production
``_WorkerState.predict_one``:

  A   shipped default: above TRIANGLE_MULT_L1_MAX_SEQ the product is written to DRAM
  B   TT_BIO_TRIMUL_OUT_L1: the product alone goes to L1, when one copy of it fits the
      share of the banks `_TRIMUL_TAIL_L1_SHARE` allows. Nothing else moves -- the path
      selector, the chunk width, the in-projection group and the operands are arm A's
  A2  the shipped default again, this session's A/A floor and determinism control

At 512 aa the op-level reading for this flip is 1.105x (0.2162 -> 0.1956 ms, prod A/A 1.15 %,
perf/ttx_deadends/b1_prod_ab.json). An op ratio is not a fold second and this script is what
decides whether it survives to one.

Harness, stage splitter and per-fold CIF hash are unchanged from
perf/b2x_trimul/fold_ab_trimul_512.py. Two things are not:

  --sizes   several sequence lengths in ONE device open and ONE model load, so a size
            sweep costs one process instead of four
  a cold fold PER ARM, not just for A. Arm B moves a memory config, so it asks for
            trimul programs arm A never compiles. Warming A alone puts that JIT inside
            arm B's first *timed* fold: at 384 aa / 4 steps that read A 7.336 s vs
            B 9.425 s, a 0.778x "regression" that was 100 % compile. Any arm added here
            needs its own warmup fold.

Invocation (one card, pinned, nothing else on the box):

  TT_VISIBLE_DEVICES=<c> TT_BIO_LEASE_CARDS=<c> TT_BIO_LEASE_HOLDER=worker:<slug> \
    python3 perf/ttx_deadends/b1_fold_ab_512.py --sizes 384,512,640,768 --reps 3 \
      --out perf/ttx_deadends/b1_fold_ab_<host>.json
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
    ap.add_argument("--sizes", default="512",
                    help="comma-separated residue counts, all in one device open")
    ap.add_argument("--reps", type=int, default=2, help="warm folds per arm")
    ap.add_argument("--arms", default="A,B,A2")
    ap.add_argument("--steps", type=int, default=SAMPLING_STEPS,
                    help="sampling steps; only lower it for a harness smoke test")
    ap.add_argument("--recycles", type=int, default=RECYCLING_STEPS)
    args = ap.parse_args()
    SAMPLING_STEPS = args.steps
    RECYCLING_STEPS = args.recycles
    SIZES = [int(x) for x in args.sizes.split(",")]
    for n in SIZES:
        assert (FIX / f"cdk2x2_{n}.yaml").exists(), f"no fixture for {n} aa"

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
        "loadavg_start": open("/proc/loadavg").read().split()[:3],
        "protocol": {"n_residues": SIZES, "recycling_steps": RECYCLING_STEPS,
                     "sampling_steps": SAMPLING_STEPS,
                     "diffusion_samples": DIFFUSION_SAMPLES, "seed": SEED,
                     "fixture": "perf/size512/fixtures/cdk2x2_<n>.yaml + its a3m"},
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
    targets = {n: FIX / f"cdk2x2_{n}.yaml" for n in SIZES}
    for n, t in targets.items():
        _seed_msa(t, (FIX / f"cdk2x2_{n}.a3m").read_text(), msa_dir)

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

    def fold(armname, target, hook=None):
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
    from tt_bio import tenstorrent as _T

    # Read before any arm has touched it, so the shipped default is recorded rather than
    # whatever this harness picked. The environment must not pin the flag, or every arm
    # silently serves the same code.
    assert os.environ.get("TT_BIO_TRIMUL_OUT_L1") is None, \
        "TT_BIO_TRIMUL_OUT_L1 is pinned in the environment; the arms would not differ"
    DEF_OUT_L1 = _T._TRIMUL_OUT_L1
    # Per size, and not once: the chunk width, the path selector and the L1 budget are all
    # functions of the sequence length, and `l1_fits` is the condition arm B is gated on. If
    # it reads False the two arms serve identical code and the ratio is an A/A by accident
    # (the eligibility trap -- a firing condition is not a code fact).
    def _size_defaults(n):
        chunk = _T._trimul_chunk_size(n, 128, 1)
        return {"chunk": chunk,
                "result_mc": str(_T._triangle_mul_memory_config(n).buffer_type),
                "l1_fits": bool(_T._trimul_l1_fits(1, chunk, n, 2, 1))}
    OUT["defaults"] = {"trimul_out_l1": DEF_OUT_L1,
                       "trimul_tail_l1": _T._TRIMUL_TAIL_L1,
                       "trimul_l1_max_seq": _T._trimul_l1_max_seq(),
                       "per_size": {str(n): _size_defaults(n) for n in SIZES}}
    dump()

    ARMS = args.arms.split(",")

    def set_arm(name):
        if name == "DEF":
            _T.set_trimul_out_l1(DEF_OUT_L1)
            return
        _T.set_trimul_out_l1(name == "B")

    OUT["sizes"] = {}
    for n in SIZES:
        target = targets[n]
        runs = []
        # One discarded cold fold PER ARM. Arm B compiles trimul programs arm A never asks
        # for, so warming A alone leaves that JIT inside B's first timed fold and inverts
        # the sign (0.778x read on a 0.988x effect at 384 aa / 4 steps).
        print("[%d aa] cold folds, one per arm (discarded)" % n, flush=True)
        for nm in ARMS:
            set_arm(nm)
            r = fold(nm, target); r["cold"] = True; runs.append(r)
            print("  cold %-2s %8.3f s" % (nm, r["fold_s"]), flush=True)
            OUT["sizes"][str(n)] = {"runs": runs}; dump()
        for i in range(args.reps):
            for nm in ARMS:
                set_arm(nm)
                r = fold(nm, target); r["cold"] = False; r["rep"] = i
                runs.append(r)
                print("  rep %d %-2s %8.3f s  plddt %s  cif %s"
                      % (i, nm, r["fold_s"], r["plddt"], list(r["cif"].values())), flush=True)
                OUT["sizes"][str(n)] = {"runs": runs}; dump()

        warm = [r for r in runs if not r["cold"]]
        med = {nm: round(st.median([r["fold_s"] for r in warm if r["arm"] == nm]), 4)
               for nm in ARMS}
        cifs = {nm: sorted({tuple(sorted(r["cif"].items())) for r in warm if r["arm"] == nm})
                for nm in ARMS}
        OUT["sizes"][str(n)] = {
            "runs": runs,
            "median_fold_s": med,
            "AA_floor_pct": round(100 * (med["A2"] - med["A"]) / med["A"], 3),
            "x_vs_A": {nm: round(med["A"] / v, 4) for nm, v in med.items()},
            "cif_identical_to_A": {nm: cifs[nm] == cifs["A"] for nm in ARMS},
            "plddt": {nm: [r["plddt"] for r in warm if r["arm"] == nm] for nm in ARMS},
            "loadavg_end": open("/proc/loadavg").read().split()[:3],
        }
        dump()
        res = OUT["sizes"][str(n)]
        print("%d aa: %s  B/A %s  A/A floor %s %%  CIF==A %s"
              % (n, json.dumps(med), res["x_vs_A"].get("B"), res["AA_floor_pct"],
                 json.dumps(res["cif_identical_to_A"])), flush=True)
    set_arm("DEF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
