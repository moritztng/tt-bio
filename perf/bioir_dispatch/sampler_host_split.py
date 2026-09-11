#!/usr/bin/env python3
"""Boltz-2 512 aa: what ttnn trace replay is worth, measured against BioIR's CUDA graph.

BioIR's CUDA graph is worth 2.366x on an H200 (5.785 -> 2.445 s/fold). This harness asks the
same question of our equivalent on Blackhole: capture the per-step DiT device stream in a ttnn
trace and replay it instead of dispatching it op by op.

One process, one device, one target. Arms alternate so session drift cannot be read as an
effect. Every fold goes through the production ``_WorkerState.predict_one`` path, so the
number is the shipped path's number, not a microbenchmark's.

Phases:
  1  timed A/B      cold fold discarded, then off/on/off/on/off/on. Stage split comes from the
                    model's own progress_fn, synchronised at stage transitions only (never
                    inside the 200-step loop, which would price the instrument into the arm
                    under test).
  2  replay probe   one traced fold; at the ``confidence`` transition the live trace is replayed
                    back to back with no host work between replays. That is the pure device
                    time per step, the denominator of the "how busy is the accelerator" question.
  3  call census    ttnn counters installed AFTER every timed region, then one fold per arm, to
                    count ttnn calls per sampling step on each path.
  4  dispatch floor bursts of one-tile bf16 adds: max(host issue, device execute) for the
                    smallest real op, i.e. the number a per-call dispatch argument multiplies by.

Results are written after every phase so a turn that runs out of time still lands what it has.
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
    ap.add_argument("--skip-census", action="store_true")
    ap.add_argument("--guidance-split", action="store_true",
                    help="phase 5 only: price the host-side sampler arithmetic by folding with "
                         "the contact-guidance update off. Diagnostic, not a proposal: guidance "
                         "changes the coordinates, so the arms are not parity-comparable.")
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

    def fold(trace: bool, hook=None):
        state.model.structure_module._diffusion_trace = trace
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
            "trace": trace,
            "fold_s": round(wall, 3),
            "prepare_and_trunk_s": round(pre, 4),
            "stages_s": stages,
            "capture_n": cap["n"], "capture_s": round(cap["s"], 4),
            "loop_marks": loop,
            "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
            "cif": cifs,
        }

    # ---- phase 1: the timed A/B --------------------------------------------
    runs = []
    if args.guidance_split:
        # phase 5 stands alone: one discarded cold fold, then the four arms.
        r0 = fold(False); r0["cold"] = True
        print("[phase5] cold (discarded)", r0["fold_s"], flush=True)
        OUT["phase5_cold"] = r0; dump()
        sa = state.model.steering_args
        gs = []
        for lab, guid, tr in (("guidance_on_eager", True, False),
                              ("guidance_off_eager", False, False),
                              ("guidance_off_traced", False, True),
                              ("guidance_on_eager_2", True, False)):
            sa["contact_guidance_update"] = guid
            r = fold(tr); r["arm"] = lab; r["guidance"] = guid
            gs.append(r)
            print("[phase5]", lab, r["fold_s"], r["stages_s"], r["cif"], flush=True)
            OUT["phase5"] = gs; dump()
        sa["contact_guidance_update"] = True
        dump()
        print("DONE", OUT_PATH, flush=True)
        return 0
    print("[phase1] cold fold (discarded)", flush=True)
    r = fold(False); r["cold"] = True; runs.append(r)
    print("  cold", r["fold_s"], "s", flush=True)
    OUT["phase1"] = runs; dump()
    for i in range(args.reps):
        for arm in (False, True):
            r = fold(arm); r["cold"] = False; r["rep"] = i
            runs.append(r)
            print(f"  rep{i} trace={arm}: fold {r['fold_s']}s stages {r['stages_s']} "
                  f"cap {r['capture_n']}x{r['capture_s']}s cif {list(r['cif'].values())}",
                  flush=True)
            OUT["phase1"] = runs; dump()

    warm = [r for r in runs if not r["cold"]]

    def med(sel, key):
        v = [r["stages_s"].get(key, 0.0) for r in warm if r["trace"] is sel]
        return round(st.median(v), 4) if v else None

    def medf(sel):
        return round(st.median([r["fold_s"] for r in warm if r["trace"] is sel]), 4)

    summ = {}
    for lab, sel in (("eager", False), ("traced", True)):
        summ[lab] = {
            "fold_s": medf(sel),
            "sampler_s": med(sel, "sampler"),
            "conditioning_s": med(sel, "diffusion_conditioning"),
            "confidence_s": med(sel, "confidence"),
            "prepare_and_trunk_s": round(st.median(
                [r["prepare_and_trunk_s"] for r in warm if r["trace"] is sel]), 4),
            "capture_s": round(st.median(
                [r["capture_s"] for r in warm if r["trace"] is sel]), 4),
            "folds": [r["fold_s"] for r in warm if r["trace"] is sel],
        }
    summ["sampler_speedup"] = round(summ["eager"]["sampler_s"] / summ["traced"]["sampler_s"], 5)
    summ["fold_speedup"] = round(summ["eager"]["fold_s"] / summ["traced"]["fold_s"], 5)
    summ["sampler_delta_ms"] = round(
        (summ["eager"]["sampler_s"] - summ["traced"]["sampler_s"]) * 1000, 1)
    summ["cif_bit_exact"] = len({tuple(sorted(r["cif"].values())) for r in warm}) == 1
    summ["cif_digests"] = sorted({v for r in warm for v in r["cif"].values()})
    OUT["summary"] = summ; dump()
    print("[phase1] summary", json.dumps(summ, indent=1), flush=True)

    # ---- phase 2: pure device replay rate ----------------------------------
    probe = {}

    def replay_probe():
        tr = getattr(score_model, "_diff_trace", None)
        if not tr:
            probe["error"] = "no live trace at the confidence boundary"
            return
        n = 20
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(dev, tr["tid"], cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        probe["device_only_ms_per_step"] = round((time.perf_counter() - t0) / n * 1000, 4)
        probe["n"] = n
        probe["trace_key"] = {"B": tr["B"], "N_padded": tr["N_padded"]}

    print("[phase2] replay probe", flush=True)
    pr = fold(True, hook=replay_probe)
    probe["probe_fold"] = pr
    OUT["phase2"] = probe; dump()
    print("[phase2]", json.dumps(probe, indent=1), flush=True)

    # ---- phase 4 before 3: the dispatch floor, before ttnn is wrapped -------
    floor = {}
    try:
        a = ttnn.from_torch(torch.zeros(32, 32), layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16, device=dev)
        b = ttnn.from_torch(torch.ones(32, 32), layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16, device=dev)
        burst, bursts = 4000, []
        for _ in range(5):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(burst):
                ttnn.add(a, b)
            ttnn.synchronize_device(dev)
            bursts.append((time.perf_counter() - t0) / burst * 1e6)
        floor = {"us_per_call": round(st.median(bursts), 4),
                 "bursts_us": [round(x, 4) for x in bursts], "burst_n": burst}
    except Exception as e:  # pragma: no cover
        floor = {"error": repr(e)}
    OUT["dispatch_floor"] = floor; dump()
    print("[phase4] dispatch floor", floor, flush=True)

    # ---- phase 3: ttnn call census -----------------------------------------
    if not args.skip_census:
        CTR = [0]
        patched = 0
        originals = {}
        for name in dir(ttnn):
            if name.startswith("_"):
                continue
            obj = getattr(ttnn, name, None)
            if not callable(obj) or isinstance(obj, type):
                continue

            def mk(f):
                def w(*a, **k):
                    CTR[0] += 1
                    return f(*a, **k)
                return w
            try:
                setattr(ttnn, name, mk(obj))
                originals[name] = obj
                patched += 1
            except Exception:
                pass

        class CountSplitter(Splitter):
            def __call__(self, stage=None, step=0, total=0, *a, **k):
                if stage == "diffusion":
                    self.n_diff += 1
                    if self.n_diff == 2:
                        self.c0 = CTR[0]
                elif stage == "confidence":
                    self.c1 = CTR[0]
                super().__call__(stage=stage, step=step, total=total)

        census = {"ttnn_names_wrapped": patched}
        for lab, arm in (("eager", False), ("traced", True)):
            state.model.structure_module._diffusion_trace = arm
            sp = CountSplitter()
            sp.c0 = sp.c1 = 0
            state.pfn = sp; state.model.progress_fn = sp
            for p in struct_dir.glob("*"):
                p.unlink()
            CTR[0] = 0
            t0 = time.perf_counter()
            state.predict_one(target, cfg)
            sp.close()
            calls = sp.c1 - sp.c0
            census[lab] = {"sampler_ttnn_calls": calls,
                           "per_step": round(calls / SAMPLING_STEPS, 1),
                           "fold_calls": CTR[0],
                           "instrumented_fold_s": round(time.perf_counter() - t0, 3)}
            print("[phase3]", lab, census[lab], flush=True)
            OUT["phase3"] = census; dump()
        for name, obj in originals.items():
            setattr(ttnn, name, obj)
        OUT["phase3"] = census; dump()

    if args.guidance_split:
        sa = state.model.steering_args
        gs = []
        for lab, guid, tr in (("guidance_on_eager", True, False),
                              ("guidance_off_eager", False, False),
                              ("guidance_off_traced", False, True),
                              ("guidance_on_eager_2", True, False)):
            sa["contact_guidance_update"] = guid
            r = fold(tr)
            r["arm"] = lab; r["guidance"] = guid
            gs.append(r)
            print("[phase5]", lab, r["fold_s"], r["stages_s"], r["cif"], flush=True)
            OUT["phase5"] = gs; dump()
        sa["contact_guidance_update"] = True

    dump()
    print("DONE", OUT_PATH, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
