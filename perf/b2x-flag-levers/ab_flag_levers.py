#!/usr/bin/env python3
"""Boltz-2 512 aa: what the two default-off byte flags are worth, measured on the fold.

Two pre-existing flags, neither shipped on:

  A  BOLTZ2_TOKEN_DIT_SDPA    fused SDPA in the token DiT instead of the materialised
                              1x16x512x512 score matrix. -77.60 MB/call, -8 dispatched ops/call.
  B  TT_BIO_ATOM_AXIS_BUCKET  size the atom axis on the real atom count (ceil(N/448)*448 = 4480)
                              instead of padded_seq * 14 = 7168. 224 windows -> 140.

They act on different phases and neither changes the other's shapes, so the arms are base, A, B
and both-on, and the both-on arm is measured rather than added.

One process, one device open (the p300c wedges on the 4th open since its last reset). Arms
alternate inside the process with the incumbent run twice per rep, so the session's own A/A floor
is a by-product of the same data rather than a separate run. Every fold goes through the
production ``_WorkerState.predict_one``.

Phases, each written to --out as it completes:
  1  512 aa timed A/B   base/A/base/B/AB per rep, medians, A/A floor from the two base folds
  2  298 aa control     the monomeric fixture, one fold per arm plus a repeat of base, CIFs kept
  3  latent defect      TT_BIO_TOKEN_BUCKET=0 at 298 aa: padded_seq*14 is not a multiple of 32 and
                        the window reshape cannot partition it. Confirm it breaks with the flag
                        off before claiming the flag fixes it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics as st
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
FIX = REPO / "perf" / "size512" / "fixtures"

RECYCLING_STEPS = 3
SAMPLING_STEPS = 200
DIFFUSION_SAMPLES = 1
SEED = 0

# base first so the cold fold and the A/A floor both come from the shipped default.
ARMS = {
    "base": (False, False),   # (atom_bucket, dit_sdpa)
    "A":    (False, True),
    "B":    (True, False),
    "AB":   (True, True),
}
ORDER = ["base", "A", "base", "B", "AB"]

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


def build_cfg(msa_dir: Path, struct_dir: Path) -> dict:
    return dict(
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
            diffusion_trace=False,      # the published 23.504 s cell's protocol
        ),
    )


def main() -> int:
    global SAMPLING_STEPS, RECYCLING_STEPS, OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True, help="where the 298 control CIFs land")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--steps", type=int, default=SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=RECYCLING_STEPS)
    ap.add_argument("--skip-512", action="store_true")
    ap.add_argument("--keep-512", action="store_true",
                    help="keep every 512 aa CIF under --cifdir as 512_<arm>_<n>/, so the "
                         "domain-split diagnostic gets N per arm instead of one closing fold")
    ap.add_argument("--skip-defect", action="store_true")
    args = ap.parse_args()
    SAMPLING_STEPS, RECYCLING_STEPS, OUT_PATH = args.steps, args.recycles, args.out

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree -- the venv's installed package "
        "would score a different tree (memory parity-gate-scores-installed-package-not-checkout)")

    dev = get_device()
    try:
        g = dev.compute_with_storage_grid_size()
        grid = [g.x, g.y]
    except Exception:
        grid = None
    import socket
    OUT["env"] = {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "lease_cards": os.environ.get("TT_BIO_LEASE_CARDS"),
        "grid": grid, "torch": torch.__version__,
        "arch": str(getattr(dev, "arch", lambda: "?")()),
        "tt_bio_file": _TB.__file__,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(),
        "protocol": {"recycling_steps": RECYCLING_STEPS, "sampling_steps": SAMPLING_STEPS,
                     "diffusion_samples": DIFFUSION_SAMPLES, "seed": SEED,
                     "diffusion_trace": False,
                     "fixtures": ["cdk2x2_512.yaml", "cdk2x2_298.yaml"]},
        "flag_defaults": {"BOLTZ2_TOKEN_DIT_SDPA": TT._B2_TOKEN_DIT_SDPA,
                          "TT_BIO_ATOM_AXIS_BUCKET": TT._ATOM_AXIS_BUCKET,
                          "ATOM_BUCKET": TT.ATOM_BUCKET},
    }
    try:
        import importlib.metadata as _md
        OUT["env"]["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    dump()

    # Both levers ship ON since 2026-09-11, so the module value is no longer the thing to assert;
    # what still has to hold is that the ENVIRONMENT does not pin either one, because the arms are
    # set in-process and an env pin would silently make every arm the same arm.
    assert not (set(os.environ) & {"BOLTZ2_TOKEN_DIT_SDPA", "TT_BIO_ATOM_AXIS_BUCKET"}), \
        "neither flag may be pinned in the environment; the arms are set in-process"

    work = Path(tempfile.mkdtemp(prefix="b2x-flaglev-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for name in ("cdk2x2_512", "cdk2x2_298"):
        _seed_msa(FIX / f"{name}.yaml", (FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)

    t_load = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2x-flag-levers", cfg)
    OUT["model_load_s"] = round(time.perf_counter() - t_load, 3)
    dump()

    diff_mod = state.model.structure_module.score_model
    seen = {}
    _orig_pop = diff_mod._populate_diffusion_cache

    def _pop(*a, **k):
        r = _orig_pop(*a, **k)
        seen["shape"] = tuple(int(x) for x in r)      # (seq_len, N, N_padded)
        return r
    diff_mod._populate_diffusion_cache = _pop

    class Splitter:
        def reset(self):
            self.marks, self.loop, self.n_diff = [], [], 0

        def __init__(self):
            self.reset()

        def _mark(self, label):
            ttnn.synchronize_device(dev)
            self.marks.append((label, time.perf_counter()))

        def __call__(self, stage=None, step=0, total=0, *a, **k):
            if stage == "diffusion":
                self.n_diff += 1
                if self.n_diff == 1:
                    self._mark("diffusion_conditioning")
                elif self.n_diff == 2:
                    self._mark("sampler")
                    self.loop.append((step, self.marks[-1][1]))
                else:
                    self.loop.append((step, time.perf_counter()))
            elif stage == "confidence":
                self._mark("confidence")

        def close(self):
            self._mark("end")
            return {lab: round(t1 - t0, 4)
                    for (lab, t0), (_l, t1) in zip(self.marks, self.marks[1:])}

    def fold(arm: str, target: Path, keep: Path | None = None) -> dict:
        TT._ATOM_AXIS_BUCKET, TT._B2_TOKEN_DIT_SDPA = ARMS[arm]
        try:
            diff_mod.reset_static_cache()
        except Exception:
            pass
        sp = Splitter()
        state.pfn = sp
        state.model.progress_fn = sp
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        seen.pop("shape", None)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
        stages = sp.close()
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        # per-step slope in the sampler, from the model's own progress emits
        ps = None
        if len(sp.loop) > 2:
            ps = round(1e3 * (sp.loop[-1][1] - sp.loop[0][1]) / (sp.loop[-1][0] - sp.loop[0][0]), 4)
        if keep:
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cifs[0], keep / cifs[0].name)
        return {
            "arm": arm, "target": target.stem,
            "atom_bucket": ARMS[arm][0], "dit_sdpa": ARMS[arm][1],
            "diffusion_shape": seen.get("shape"),
            "fold_s": round(wall, 3),
            "prepare_and_trunk_s": round(sp.marks[0][1] - t0, 4),
            "stages_s": stages,
            "sampler_ms_per_step": ps,
            "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
            "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
            "loadavg1": round(os.getloadavg()[0], 2),
        }

    t512, t298 = FIX / "cdk2x2_512.yaml", FIX / "cdk2x2_298.yaml"

    # ---- phase 1: 512 aa timed A/B -----------------------------------------
    if not args.skip_512:
        runs = []
        print("[phase1] warmup: one fold per arm, discarded (program cache, two atom widths)",
              flush=True)
        for arm in ("base", "A", "B", "AB"):
            r = fold(arm, t512); r["warmup"] = True; runs.append(r)
            print(f"  warm {arm:4s} {r['fold_s']:7.3f}s shape={r['diffusion_shape']} "
                  f"cif {r['cif_sha256'][:16]}", flush=True)
            OUT["phase1"] = runs; dump()
        kept: dict[str, int] = {}
        for i in range(args.reps):
            for arm in ORDER:
                keep = None
                if args.keep_512:
                    n = kept[arm] = kept.get(arm, -1) + 1
                    keep = args.cifdir / f"512_{arm}_{n}"
                r = fold(arm, t512, keep=keep); r["warmup"] = False; r["rep"] = i; runs.append(r)
                print(f"  rep{i} {arm:4s} {r['fold_s']:7.3f}s "
                      f"sampler {r['stages_s'].get('sampler')} "
                      f"{r['sampler_ms_per_step']}ms/step plddt {r['plddt']} "
                      f"cif {r['cif_sha256'][:16]} load {r['loadavg1']}", flush=True)
                OUT["phase1"] = runs; dump()

        warm = [r for r in runs if not r["warmup"]]

        def med(vals):
            vals = [v for v in vals if v is not None]
            return round(st.median(vals), 4) if vals else None

        summ = {}
        for arm in ARMS:
            v = [r for r in warm if r["arm"] == arm]
            if not v:
                continue
            summ[arm] = {
                "n": len(v),
                "fold_s": med([r["fold_s"] for r in v]),
                "folds": [r["fold_s"] for r in v],
                "sampler_s": med([r["stages_s"].get("sampler") for r in v]),
                "ms_per_step": med([r["sampler_ms_per_step"] for r in v]),
                "trunk_s": med([r["prepare_and_trunk_s"] for r in v]),
                "confidence_s": med([r["stages_s"].get("confidence") for r in v]),
                "shape": v[0]["diffusion_shape"],
                "cif_sha256": sorted({r["cif_sha256"] for r in v}),
                "plddt": sorted({round(float(r["plddt"]), 6) for r in v}),
            }
        # the A/A floor: split the 10 base folds by their position in the rep
        b = [r for r in warm if r["arm"] == "base"]
        b1 = med([r["fold_s"] for k, r in enumerate(b) if k % 2 == 0])
        b2 = med([r["fold_s"] for k, r in enumerate(b) if k % 2 == 1])
        s1 = med([r["sampler_ms_per_step"] for k, r in enumerate(b) if k % 2 == 0])
        s2 = med([r["sampler_ms_per_step"] for k, r in enumerate(b) if k % 2 == 1])
        floor = {
            "base_first_median_s": b1,
            "base_second_median_s": b2,
            "fold_AA_ratio": round(b1 / b2, 5),
            "step_first_ms": s1,
            "step_second_ms": s2,
            "step_AA_ratio": round(s1 / s2, 5) if s1 and s2 else None,
            "sampler_first_s": med([r["stages_s"].get("sampler")
                                    for k, r in enumerate(b) if k % 2 == 0]),
            "sampler_second_s": med([r["stages_s"].get("sampler")
                                     for k, r in enumerate(b) if k % 2 == 1]),
            "fold_spread_pct": round(100 * (max(r["fold_s"] for r in b) -
                                            min(r["fold_s"] for r in b)) /
                                     st.median([r["fold_s"] for r in b]), 2),
        }
        base = summ["base"]
        for arm in ("A", "B", "AB"):
            if arm in summ:
                summ[arm]["fold_speedup"] = round(base["fold_s"] / summ[arm]["fold_s"], 5)
                if base["ms_per_step"] and summ[arm]["ms_per_step"]:
                    summ[arm]["step_speedup"] = round(
                        base["ms_per_step"] / summ[arm]["ms_per_step"], 5)
                if base["sampler_s"] and summ[arm]["sampler_s"]:
                    summ[arm]["sampler_speedup"] = round(
                        base["sampler_s"] / summ[arm]["sampler_s"], 5)
                    summ[arm]["sampler_saved_s"] = round(
                        base["sampler_s"] - summ[arm]["sampler_s"], 4)
                summ[arm]["fold_saved_s"] = round(base["fold_s"] - summ[arm]["fold_s"], 4)
        if "A" in summ and "B" in summ and "AB" in summ:
            summ["additivity"] = {
                "sum_of_separate_saved_s": round(summ["A"]["fold_saved_s"] +
                                                 summ["B"]["fold_saved_s"], 4),
                "combined_saved_s": summ["AB"]["fold_saved_s"],
            }
        OUT["phase1_summary"] = summ
        OUT["phase1_AA_floor"] = floor
        dump()
        print("\n[phase1] A/A floor:", json.dumps(floor), flush=True)
        print("[phase1] summary:", json.dumps({k: v for k, v in summ.items()
                                               if k != "additivity"}, indent=1), flush=True)

    # ---- phase 2: the 298 aa monomeric control -----------------------------
    ctl = []
    print("\n[phase2] cdk2x2_298 control", flush=True)
    for arm in ("base", "A", "B", "AB", "base"):
        tag = f"{arm}_{sum(1 for c in ctl if c['arm'] == arm)}"
        r = fold(arm, t298, keep=args.cifdir / f"298_{tag}")
        r["tag"] = tag
        ctl.append(r)
        print(f"  {tag:8s} {r['fold_s']:7.3f}s shape={r['diffusion_shape']} "
              f"plddt {r['plddt']} cif {r['cif_sha256'][:16]}", flush=True)
        OUT["phase2"] = ctl; dump()

    # ---- phase 3: the suspected latent defect ------------------------------
    # padded_seq * 14 is a multiple of 32 only because padded_seq is a multiple of 32. With the
    # token bucket off and 298 tokens, 298*14 = 4172 and 4172/32 = 130.375: the window reshape
    # cannot partition the axis. Confirm it breaks BEFORE claiming the atom bucket closes it.
    if not args.skip_defect:
        print("\n[phase3] TT_BIO_TOKEN_BUCKET=0 at 298 aa", flush=True)
        os.environ["TT_BIO_TOKEN_BUCKET"] = "0"
        defect = {"token_bucket": "0", "n_tokens": 298,
                  "padded_seq_times_14": 298 * 14, "is_multiple_of_32": (298 * 14) % 32 == 0}
        for arm in ("base", "B"):
            leg = {"arm": arm}
            try:
                r = fold(arm, t298)
                leg.update(ok=True, fold_s=r["fold_s"], shape=r["diffusion_shape"],
                           cif_sha256=r["cif_sha256"])
            except Exception as e:                                     # noqa: BLE001
                leg.update(ok=False, error=f"{type(e).__name__}: {e}"[:600],
                           tb=traceback.format_exc()[-1500:])
            defect[arm] = leg
            print(f"  {arm}: ok={leg['ok']} {leg.get('error', leg.get('shape'))}", flush=True)
            OUT["phase3"] = defect; dump()
        os.environ["TT_BIO_TOKEN_BUCKET"] = "1"

    OUT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dump()
    print("\nwrote", OUT_PATH, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
