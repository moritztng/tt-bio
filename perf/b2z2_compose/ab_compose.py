#!/usr/bin/env python3
"""Boltz-2 512 aa on Blackhole: what wave 2's landed levers are worth composed.

Four levers landed on branches, every one of them measured on Wormhole or on one op in
isolation. This runs them on a Blackhole fold, alone and together, so the union has an
additivity discount instead of a product of op ratios.

  K2     tt_bio.triatt_qkv._FUSED_ENABLED        [Wq|Wk|Wv|Wg]: one operand pass, not two
  DST    TT_BIO_GATE_DST_RESIDENT                the gated move keeps its product in DST
  MSA    TT_BIO_MSA_DEPTH_LADDER                 35 real rows land on 64, not 1024
  UNION  all three

F1-direct (``trimul_tail.DIRECT_PACK``) is not an arm: ``F1_BLOCK_KEYS = {(8, 8)}`` and
Boltz-2 keys (4, 4) at c_z=128, so the kernel declines every call this fixture makes. The
harness counts its declines rather than asserting the point.

Protocol is wave 1's, unchanged: ``perf/size512/fixtures/cdk2x2_512.yaml`` with its fixed
35-row a3m, 3 recycles, 200 sampling steps, 1 sample, seed 0, templates off, timed at
``predict_one``, cold fold per arm discarded. One process, one device open.

Two base positions per rep, so the session's A/A floor comes out of the same data and
``_L1_OUT_RUNG`` (tenstorrent.py, module-level, only grows) cannot rewrite the baseline
under one arm without rewriting it under the base beside it.
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

#: One arm is the set of levers it turns on. Everything absent is shipped-default off.
#:   k2     triatt_qkv._FUSED_ENABLED          [Wq|Wk|Wv|Wg] in one operand pass
#:   dst    TT_BIO_GATE_DST_RESIDENT           the gated move keeps its product in DST
#:   msa    TT_BIO_MSA_DEPTH_LADDER            35 real rows land on 64, not 1024
#:   host   TT_BIO_DEVICE_CONDITIONING         the diffusion conditioning runs on the device
#:   cbd    _MM_BLOCK M_block 4 -> 8           on the c_z=128 projection keys
#:   trace  Boltz2._diffusion_trace            replay the per-step DiT as one captured trace
LEVERS = ("k2", "dst", "msa", "host", "cbd", "trace")
ARMS = {
    "base":     (),
    "K2":       ("k2",),
    "DST":      ("dst",),
    "MSA":      ("msa",),
    "HOST":     ("host",),
    "CBD":      ("cbd",),
    "TRACE":    ("trace",),
    "K2DST":    ("k2", "dst"),
    "UNION_BE": ("k2", "dst", "cbd"),
    "UNION_ALL": ("k2", "dst", "cbd", "host"),
    "UNION":    ("k2", "dst", "msa"),
}
ORDER = ["base", "K2", "DST", "base", "MSA", "UNION"]

#: `_MM_BLOCK` entries the CBD arm rewrites, and what it rewrites them to. `b2z2-cb-depth-prefetch`
#: measured M_block 4 -> 8 at 1.0138x on a Wormhole block and found ring depth worth nothing, so
#: the arm is the block shape and not the depth. Going through the table rather than
#: `mm_generic.BLOCK_OVERRIDE` is deliberate: `_CACHE`'s key carries `cfg`, which carries the table
#: entry, so a changed entry gets its own compiled program. BLOCK_OVERRIDE is applied INSIDE
#: `build()` after the key is formed, so an A/B over it is handed the other arm's program -- the
#: same instrument failure `_cache_key_gated` exists to prevent.
CBD_KEYS = ((4, 4), (4, 12), (4, 16))
CBD_M_BLOCK = 8

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def _seed_msa(target: Path, a3m_text: str, msa_dir: Path) -> None:
    from tt_bio.main import _read_bio_chains
    chains = _read_bio_chains(target)
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
            diffusion_trace=False,
        ),
    )


def main() -> int:
    global SAMPLING_STEPS, RECYCLING_STEPS, OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--steps", type=int, default=SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=RECYCLING_STEPS)
    ap.add_argument("--skip-298", action="store_true")
    ap.add_argument("--trace-region", action="store_true",
                    help="reserve a 1 GiB ttnn trace region before the device opens. Required by "
                         "the TRACE arm and by nothing else, so it is opt-in: reserving it "
                         "changes what every arm in the session has to work with.")
    ap.add_argument("--order", default=None,
                    help="comma-separated arm order per rep, overriding ORDER. Must contain at "
                         "least two base positions or the A/A floor has nothing to pair.")
    args = ap.parse_args()
    SAMPLING_STEPS, RECYCLING_STEPS, OUT_PATH = args.steps, args.recycles, args.out
    order = ORDER if not args.order else [a.strip() for a in args.order.split(",")]
    assert order.count("base") >= 2, (
        "two base positions per rep, minimum -- one gives no A/A pair at all, which is how wave 1 "
        "got a null fold_AA_ratio")
    assert set(order) <= set(ARMS), f"unknown arm in --order: {set(order) - set(ARMS)}"
    assert args.trace_region or not any("trace" in ARMS[a] for a in order), \
        "the TRACE arm needs --trace-region: the region is reserved before the device opens"

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    import tt_bio.triatt_qkv as QKV
    import tt_bio.reblock_permute as RP
    import tt_bio.trimul_tail as TTAIL
    import tt_bio.mm_generic as MG
    from tt_bio.token_axis import msa_depth_bucket, msa_ladder_enabled
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree "
        "(memory parity-gate-scores-installed-package-not-checkout)")

    # The arms are set in-process. An env pin would silently make every arm the same arm.
    assert not (set(os.environ) & {"TT_BIO_TRIATT_QKV_GATE_FUSED", "TT_BIO_GATE_DST_RESIDENT",
                                  "TT_BIO_MSA_DEPTH_LADDER", "TT_BIO_MSA_PAD_POISON",
                                  "TT_BIO_DEVICE_CONDITIONING", "TT_BIO_HOST_LEVERS"}), \
        "no lever may be pinned in the environment"

    if args.trace_region:
        os.environ.setdefault("TT_BIO_TRACE_REGION_SIZE", str(1 << 30))
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
        "lever_defaults": {"TRIATT_QKV_GATE_FUSED": QKV._FUSED_ENABLED,
                           "TRIATT_HEAD_MAJOR_QKV": QKV._ENABLED,
                           "GATE_DST_RESIDENT": RP.GATE_DST_RESIDENT,
                           "MSA_LADDER": msa_ladder_enabled(),
                           "WORK_CB_DEPTH": RP.WORK_CB_DEPTH,
                           "MM_CB_DEPTH": MG.CB_DEPTH,
                           "MM_BLOCK_OVERRIDE": dict(MG.BLOCK_OVERRIDE),
                           "TRIMUL_TAIL_DIRECT_PACK": TTAIL.DIRECT_PACK,
                           "F1_BLOCK_KEYS": sorted(str(k) for k in TTAIL.F1_BLOCK_KEYS)},
        "msa_bucket_for_35_rows": {"ladder_off": None, "ladder_on": None},
    }
    try:
        import importlib.metadata as _md
        OUT["env"]["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    os.environ.pop("TT_BIO_MSA_DEPTH_LADDER", None)
    OUT["env"]["msa_bucket_for_35_rows"]["ladder_off"] = msa_depth_bucket(35)
    os.environ["TT_BIO_MSA_DEPTH_LADDER"] = "1"
    OUT["env"]["msa_bucket_for_35_rows"]["ladder_on"] = msa_depth_bucket(35)
    os.environ.pop("TT_BIO_MSA_DEPTH_LADDER", None)
    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z2-compose-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for name in ("cdk2x2_512", "cdk2x2_298"):
        _seed_msa(FIX / f"{name}.yaml", (FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)

    t_load = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-bh-compose", cfg)
    OUT["model_load_s"] = round(time.perf_counter() - t_load, 3)
    dump()

    diff_mod = state.model.structure_module.score_model
    seen = {}
    _orig_pop = diff_mod._populate_diffusion_cache

    def _pop(*a, **k):
        r = _orig_pop(*a, **k)
        seen["shape"] = tuple(int(x) for x in r)
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

    mm_block_shipped = {k: TT._MM_BLOCK[k] for k in CBD_KEYS if k in TT._MM_BLOCK}
    OUT["env"]["cbd_keys_present"] = {str(k): list(v) for k, v in mm_block_shipped.items()}

    def set_arm(arm: str):
        on = set(ARMS[arm])
        assert on <= set(LEVERS), f"unknown lever in arm {arm}: {on - set(LEVERS)}"
        QKV._FUSED_ENABLED = "k2" in on
        RP.set_gate_dst_resident("dst" in on)
        for flag, lever in (("TT_BIO_MSA_DEPTH_LADDER", "msa"),
                            ("TT_BIO_DEVICE_CONDITIONING", "host")):
            if lever in on:
                os.environ[flag] = "1"
            else:
                os.environ.pop(flag, None)
        for k, shipped in mm_block_shipped.items():
            TT._MM_BLOCK[k] = ((CBD_M_BLOCK,) + tuple(shipped[1:])) if "cbd" in on else shipped
        state.model._diffusion_trace = ("trace" in on) and state.model.use_tenstorrent

    def counters():
        return {"k2_served": QKV.FUSED_STATS[0], "k2_declined": QKV.FUSED_STATS[1],
                "mm_programs": len(MG._CACHE),
                "headmajor_served": QKV.STATS[0], "headmajor_declined": QKV.STATS[1],
                "gated_served": RP.STATS_GATED[0], "gated_declined": RP.STATS_GATED[1],
                "f1_served": TTAIL.STATS[0], "f1_declined": TTAIL.STATS[1]}

    def fold(arm: str, target: Path, keep: Path | None = None) -> dict:
        set_arm(arm)
        before = counters()
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
        after = counters()
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        ps = None
        if len(sp.loop) > 2:
            ps = round(1e3 * (sp.loop[-1][1] - sp.loop[0][1]) / (sp.loop[-1][0] - sp.loop[0][0]), 4)
        if keep:
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cifs[0], keep / cifs[0].name)
        return {
            "arm": arm, "target": target.stem,
            "levers": {lv: (lv in ARMS[arm]) for lv in LEVERS},
            "diffusion_shape": seen.get("shape"),
            "fold_s": round(wall, 3),
            "prepare_and_trunk_s": round(sp.marks[0][1] - t0, 4),
            "stages_s": stages,
            "sampler_ms_per_step": ps,
            "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
            "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
            "calls": {k: after[k] - before[k] for k in after},
            "mm_programs_total": len(MG._CACHE),
            "mm_block_now": {str(k): list(TT._MM_BLOCK[k]) for k in mm_block_shipped},
            "diffusion_trace": bool(getattr(state.model, "_diffusion_trace", False)),
            "loadavg1": round(os.getloadavg()[0], 2),
        }

    t512, t298 = FIX / "cdk2x2_512.yaml", FIX / "cdk2x2_298.yaml"

    # ---- phase 1: 512 aa timed A/B -----------------------------------------
    runs = []
    print("[phase1] warmup: one fold per arm, discarded", flush=True)
    for arm in dict.fromkeys(order):
        r = fold(arm, t512); r["warmup"] = True; runs.append(r)
        print(f"  warm {arm:5s} {r['fold_s']:7.3f}s shape={r['diffusion_shape']} "
              f"cif {r['cif_sha256'][:16]} calls={r['calls']}", flush=True)
        OUT["phase1"] = runs; dump()
    kept: dict[str, int] = {}
    for i in range(args.reps):
        for arm in order:
            n = kept[arm] = kept.get(arm, -1) + 1
            keep = args.cifdir / f"512_{arm}_{n}"
            r = fold(arm, t512, keep=keep); r["warmup"] = False; r["rep"] = i; runs.append(r)
            print(f"  rep{i} {arm:5s} {r['fold_s']:7.3f}s "
                  f"trunk {r['prepare_and_trunk_s']:6.3f} "
                  f"sampler {r['stages_s'].get('sampler')} "
                  f"cif {r['cif_sha256'][:16]} load {r['loadavg1']}", flush=True)
            OUT["phase1"] = runs; dump()

    timed = [r for r in runs if not r["warmup"]]
    med = {}
    for arm in dict.fromkeys(order):
        v = sorted(r["fold_s"] for r in timed if r["arm"] == arm)
        if v:
            med[arm] = {"n": len(v), "median": round(st.median(v), 3),
                        "min": v[0], "max": v[-1],
                        "spread_pct": round(100 * (v[-1] - v[0]) / st.median(v), 2)}
    # A/A floor: the two base positions of each rep against each other.
    aa = []
    for i in range(args.reps):
        b = [r["fold_s"] for r in timed if r["arm"] == "base" and r["rep"] == i]
        if len(b) == 2:
            aa.append(max(b) / min(b))
    OUT["phase1_summary"] = {
        "median_s": med,
        "fold_AA_ratio": round(st.median(aa), 5) if aa else None,
        "AA_pairs": len(aa),
        "ratio_vs_base": {a: round(med["base"]["median"] / med[a]["median"], 5)
                          for a in med if a != "base"},
        "worst_rep_ratio_vs_base_median": {
            a: round(med["base"]["median"] / max(r["fold_s"] for r in timed if r["arm"] == a), 5)
            for a in med if a != "base"},
        "cif_sha256_by_arm": {a: sorted({r["cif_sha256"] for r in timed if r["arm"] == a})
                              for a in med},
        "plddt_by_arm": {a: sorted({r["plddt"] for r in timed if r["arm"] == a}) for a in med},
    }
    dump()

    # ---- phase 2: 298 aa monomeric control ---------------------------------
    if not args.skip_298:
        c = []
        for arm in list(dict.fromkeys(order)) + ["base"]:
            n = len([x for x in c if x["arm"] == arm])
            r = fold(arm, t298, keep=args.cifdir / f"298_{arm}_{n}")
            c.append(r)
            print(f"  298 {arm:5s} {r['fold_s']:7.3f}s cif {r['cif_sha256'][:16]}", flush=True)
            OUT["phase2_298"] = c; dump()

    OUT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dump()
    print(json.dumps(OUT.get("phase1_summary", {}), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        OUT["error"] = traceback.format_exc()
        dump()
        sys.exit(1)
