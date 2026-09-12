#!/usr/bin/env python3
"""The Boltz-2 512 aa fold on an NVIDIA GPU, decomposed into the same three units
as the Tenstorrent cell, and placed against the GPU's own measured roofs.

This is the control the b2z campaign never ran. On Blackhole a PairformerLayer costs
41.4152 ms of device time where a two-parameter cost model built from the card's own
op-cost curve (t_fixed per op + bytes / BW_eff) says 17.67 ms: a 2.34x deficit that
neither roof explains. The question here is whether a twenty-year-old, heavily tuned
software stack misses its own roofs by the same factor on the same computation.

Same three units (tt_bio/reference.py is the torch module tree the ttnn port replaces
layer for layer, so the boundaries are identical, not analogous):

    PairformerLayer      280 calls/fold on TT, 41.4152 ms/call
    DiffusionModule      200 calls/fold on TT, 32.5179 ms/call (one diffusion step)
    MSALayer              16 calls/fold on TT, 120.2894 ms/call

Three timings per unit, because they answer different questions:

    eager_ms     one call, synchronised either side. Includes launch gaps: what the
                 fold actually pays.
    kernel_ms    sum of CUDA kernel durations in that call, from torch.profiler. Device
                 busy time only, gaps excluded.
    graph_ms     the call captured in a CUDA graph and replayed back to back. The exact
                 analogue of the ttnn trace replay that produced the TT numbers
                 (perf/b2x_op_cost/device_floor.py), and the one the deficit uses.

Op count and bytes come from a TorchDispatchMode census of one call, with the same rule
the TT byte census uses: every distinct storage an op reads plus every distinct storage
it writes, deduped by data pointer within the op (a view is not a second buffer).

Usage on the rented box, after setup.sh:

    python3 gpu_decomp.py --roofs roofs_h100.json --out decomp_h100.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

# The Tenstorrent cell, for the side-by-side. Measured, not asserted: CONTEXT.md section 2
# (perf/b2x_op_cost/device_floor_512_qb2c0.json, qb2 card 0, Blackhole, ttnn trace replay).
TT = {
    "PairformerLayer": {"device_ms": 41.4152, "calls": 280, "ops": 428, "bytes": 6.651e9},
    "DiffusionModule": {"device_ms": 32.5179, "calls": 200},
    "MSALayer": {"device_ms": 120.2894, "calls": 16},
}
TT_MODEL = {"t_fixed_us": 6.36, "BW_eff_GBs": 445.0}     # CONTEXT.md section 3

SAMPLING_STEPS = 200
DIFFUSION_SAMPLES = 1
SEED = 0
OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1, default=str))


def gpu_state() -> dict:
    def smi(q, mode="--format=csv,noheader"):
        try:
            return subprocess.run(["nvidia-smi", q, mode], capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception as e:                                              # noqa: BLE001
            return f"nvidia-smi failed: {e}"
    return {"gpu": smi("--query-gpu=name,driver_version,clocks.sm,power.draw,"
                       "temperature.gpu,utilization.gpu"),
            "compute_apps": smi("--query-compute-apps=pid,used_memory"),
            "own_pid": os.getpid()}


# --------------------------------------------------------------------------------------
# op + byte census
# --------------------------------------------------------------------------------------
class Census(TorchDispatchMode):
    """Count dispatched ops and the bytes they move, one call of one module.

    Same accounting rule as the TT byte census (`ttnn-graph-byte-count-must-dedupe-buffer
    -not-tensor-id`): an op moves every distinct storage it reads plus every distinct
    storage it writes, and two views of one buffer are one buffer. Storages are keyed by
    data pointer, so a slice never counts as a second read.
    """

    def __init__(self):
        super().__init__()
        self.ops = 0
        self.views = 0
        self.bytes = 0
        self.by_op: dict[str, dict] = {}
        self.sizes: list[int] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        ins, _ = tree_flatten((args, kwargs))
        out = func(*args, **kwargs)
        outs, _ = tree_flatten(out)
        name = str(func)

        # A view moves nothing. aten.view / slice / expand / select / permute hand back a
        # different stride on the same buffer and launch no kernel, so counting them reads
        # 56.1 GB for a block whose real traffic is a fifth of that -- the same class of
        # error as the TT-side 214 MB/call diffusion artifact (b2x-diffusion-layer-bytes).
        # The test is structural rather than a name list: an op is a view when every output
        # lands on a buffer an input already occupied AND it does not mutate in place.
        def _ptrs(ts):
            out = set()
            for t in ts:
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    try:
                        out.add(t.data_ptr())
                    except Exception:                                       # noqa: BLE001
                        pass
            return out

        in_ptrs, out_ptrs = _ptrs(ins), _ptrs(outs)
        mutating = name.rstrip(".default").endswith("_") or "out" in kwargs
        if out_ptrs and out_ptrs <= in_ptrs and not mutating:
            self.views += 1
            return out

        seen: dict[int, int] = {}
        for t in list(ins) + list(outs):
            if isinstance(t, torch.Tensor) and t.is_cuda:
                try:
                    ptr = t.data_ptr()
                    nb = t.numel() * t.element_size()
                except Exception:                                           # noqa: BLE001
                    continue
                # Two aliases of one buffer are one buffer, and the larger view is the
                # one that bounds the traffic. Charging the whole STORAGE instead would
                # bill a 512x512x128 allocation once per chunk read out of it: the first
                # cut of this census read 60.2 GB for a block whose tensors add up to a
                # tenth of that.
                if nb > seen.get(ptr, 0):
                    seen[ptr] = nb
        moved = sum(seen.values())
        self.ops += 1
        self.bytes += moved
        self.sizes.append(moved)
        r = self.by_op.setdefault(name, {"n": 0, "bytes": 0})
        r["n"] += 1
        r["bytes"] += moved
        return out


# --------------------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------------------
def eager_ms(call, n=5) -> dict:
    per = []
    for _ in range(n + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        per.append(1e3 * (time.perf_counter() - t0))
    per = per[1:]
    return {"median_ms": round(st.median(per), 4), "all_ms": [round(p, 4) for p in per]}


def kernel_ms(call, n=3) -> dict:
    from torch.profiler import ProfilerActivity, profile
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(n):
            call()
        torch.cuda.synchronize()
    tot_us, launches = 0.0, 0
    for e in prof.key_averages():
        if e.device_type == torch.autograd.DeviceType.CUDA or e.self_device_time_total:
            tot_us += e.self_device_time_total
            launches += e.count if e.self_device_time_total else 0
    return {"ms_per_call": round(1e-3 * tot_us / n, 4), "cuda_events_per_call": round(launches / n, 1)}


def graph_ms(call, replays=20) -> dict:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    torch.cuda.synchronize()
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    per = []
    for _ in range(5):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(replays):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        per.append(e0.elapsed_time(e1) / replays)
    del g
    torch.cuda.synchronize()
    return {"median_ms": round(st.median(per), 4), "all_ms": [round(p, 4) for p in per],
            "replays": replays}


# --------------------------------------------------------------------------------------
# the fold
# --------------------------------------------------------------------------------------
def build_fold(msa_dir: Path, target: Path, a3m: Path, recycles: int, bf16: bool,
               use_kernels: bool = False):
    """Load Boltz-2 on the GPU and return a callable that folds once, plus the state.

    Mirrors scripts/gpu_vs_tt/tt_baseline.py's build_fold with accelerator="gpu": same
    config, same seeded MSA cache, so no search runs in or before a timed fold.
    """
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio.main import _read_bio_chains

    chains = _read_bio_chains(target)
    assert len(chains) == 1, f"{target} is not a monomer: {len(chains)}"
    seq = chains[0][1]
    text = a3m.read_text()
    assert text.split("\n")[1] == seq, "a3m query row does not match the target sequence"
    msa_dir.mkdir(parents=True, exist_ok=True)
    (msa_dir / f"{hashlib.sha256(seq.encode()).hexdigest()[:16]}.a3m").write_text(text)

    work = Path(tempfile.mkdtemp(prefix="b2zgpu-"))
    struct_dir = work / "out"
    struct_dir.mkdir(parents=True, exist_ok=True)
    # conf_kwargs is what tt_bio/main.py's predict builds; the worker takes it as given,
    # so a harness that calls load_model directly has to build the same dict. use_kernels
    # is False here and that is the point of the arm: the pure-torch path is the one whose
    # module tree the ttnn port mirrors op for op. The cuequivariance arm is separate.
    _diffusion = {"step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0,
                  "noise_scale": 1.003, "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0,
                  "sigma_data": 16.0, "P_mean": -1.2, "P_std": 1.5,
                  "coordinate_augmentation": True, "alignment_reverse_diff": True,
                  "synchronize_sigmas": True}
    _pairformer = {"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True}
    _msa = {"subsample_msa": False, "num_subsampled_msa": 1024,
            "use_paired_feature": True, "msa_s": 64, "msa_blocks": 4, "msa_dropout": 0.15,
            "z_dropout": 0.25, "pairwise_head_width": 32, "pairwise_num_heads": 4,
            "activation_checkpointing": True}
    conf_kwargs = dict(
        predict_args={"recycling_steps": recycles, "sampling_steps": SAMPLING_STEPS,
                      "diffusion_samples": DIFFUSION_SAMPLES, "max_parallel_samples": 5},
        diffusion_process_args=_diffusion, pairformer_args=_pairformer, msa_args=_msa,
        steering_args={"fk_steering": False, "physical_guidance_update": False,
                       "contact_guidance_update": True, "num_particles": 3, "fk_lambda": 4.0,
                       "fk_resampling_interval": 3, "num_gd_steps": 20},
        use_kernels=use_kernels, use_tenstorrent=False, trace=False, diffusion_trace=False,
    )
    cfg = dict(model="boltz2", fast=False, output_format="cif", conf_kwargs=conf_kwargs,
               recycling_steps=recycles, sampling_steps=SAMPLING_STEPS,
               diffusion_samples=DIFFUSION_SAMPLES, seed=SEED, trace=False,
               msa_dir=str(msa_dir), struct_dir=str(struct_dir),
               use_msa_server=True, msa_db_path=None, use_envdb=False, msa_endpoint=None,
               single_sequence=False, msa_server_url="https://api.colabfold.com",
               msa_pairing_strategy="greedy", msa_server_username=None,
               msa_server_password=None, api_key_value=None, max_msa_seqs=8192,
               write_pae=False, write_pde=False, write_embeddings=False, method=None)
    _ensure_local_artifacts(cfg)
    state = _WorkerState("gpu")
    if bf16:
        # The device runs bf16, so the control runs bf16: an fp32 GPU column would put the
        # two sides on different arithmetic and different byte counts at once.
        state._maybe_ref_bf16 = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    t0 = time.perf_counter()
    state.load_model(cfg)
    load_s = time.perf_counter() - t0
    state.bind_run("b2zgpu", cfg)
    state.pfn = lambda *a, **k: None

    def one_fold():
        for p in struct_dir.glob("*"):
            p.unlink()
        torch.cuda.synchronize()
        t = time.perf_counter()
        metrics, _best, _feats = state.predict_one(target, dict(cfg))
        torch.cuda.synchronize()
        return time.perf_counter() - t, metrics

    return one_fold, state, dict(load_s=round(load_s, 2), struct_dir=str(struct_dir),
                                 recycling_steps=recycles, n_msa=text.count(">"))


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--roofs", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path("/work/tt-bio"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--warm-folds", type=int, default=2)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--use-kernels", action="store_true",
                    help="route trimul/triatt through cuequivariance fused kernels")
    a = ap.parse_args()
    OUT_PATH = a.out

    torch.set_grad_enabled(False)
    roofs = json.loads(a.roofs.read_text())
    fix = a.repo / "perf" / "size512" / "fixtures"
    tgt, a3m = fix / f"cdk2x2_{a.size}.yaml", fix / f"cdk2x2_{a.size}.a3m"

    OUT.update({"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "device": torch.cuda.get_device_name(0),
                        "size_aa": a.size, "dtype": a.dtype, "use_kernels": a.use_kernels,
                        "recycles": a.recycles, "sampling_steps": SAMPLING_STEPS,
                        "samples": DIFFUSION_SAMPLES, "seed": SEED, **gpu_state()},
                "roofs": {k: roofs[k] for k in ("fits", "compute_roof")}})
    dump()

    one_fold, state, meta = build_fold(Path("/work/msa"), tgt, a3m, a.recycles,
                                       a.dtype == "bf16", a.use_kernels)
    OUT["fold_meta"] = meta
    dump()

    # ---- the headline: cold fold discarded, then warm folds ----------------------------
    print("=== cold fold ===", flush=True)
    t, m = one_fold()
    OUT["cold_fold_s"] = round(t, 3)
    OUT["cold_metrics"] = {k: v for k, v in m.items() if isinstance(v, (int, float, str))}
    print(f"  cold {t:.3f} s  {OUT['cold_metrics']}", flush=True)
    dump()

    warm = []
    for i in range(a.warm_folds):
        t, m = one_fold()
        warm.append(round(t, 3))
        print(f"  warm {i} {t:.3f} s", flush=True)
        OUT["warm_folds_s"] = warm
        OUT["warm_median_s"] = round(st.median(warm), 3)
        OUT["warm_metrics"] = {k: v for k, v in m.items() if isinstance(v, (int, float, str))}
        dump()

    # ---- grab one settled call of each unit, on the next fold --------------------------
    import tt_bio.reference as R

    targets = {"PairformerLayer": R.PairformerLayer,
               "MSALayer": R.MSALayer,
               "DiffusionModule": R.DiffusionModule}
    counts: dict[str, dict[str, int]] = {}
    grabs: dict[str, dict] = {}
    originals = []

    def sig(args, kwargs):
        parts = []
        for t in list(args) + list(kwargs.values()):
            if isinstance(t, torch.Tensor):
                parts.append("x".join(str(int(d)) for d in t.shape))
        return ",".join(parts)

    def vol(args):
        return max((t.numel() for t in args if isinstance(t, torch.Tensor)), default=0)

    def arm(name, cls):
        orig = cls.__call__
        originals.append((cls, orig))

        def w(self_obj, *args, **kw):
            s = sig(args, kw)
            c = counts.setdefault(name, {})
            c[s] = c.get(s, 0) + 1
            g = grabs.get(name)
            if c[s] >= 2 and (g is None or vol(args) > g["vol"]):
                grabs[name] = {"obj": self_obj, "vol": vol(args), "sig": s,
                               "args": tuple(x.clone() if isinstance(x, torch.Tensor) else x
                                             for x in args),
                               "kwargs": {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                                          for k, v in kw.items()}}
            return orig(self_obj, *args, **kw)
        cls.__call__ = w

    print("=== grabbing fold ===", flush=True)
    for n, c in targets.items():
        arm(n, c)
    t, _m = one_fold()
    for cls, orig in originals:
        cls.__call__ = orig
    OUT["grab_fold_s"] = round(t, 3)
    OUT["calls_per_fold"] = {n: {"total": sum(c.values()), "by_shape": c}
                             for n, c in counts.items()}
    OUT["grabbed"] = {n: {"sig": g["sig"]} for n, g in grabs.items()}
    print("  " + json.dumps({n: sum(c.values()) for n, c in counts.items()}), flush=True)
    dump()

    # ---- per-unit: census, then three timings ------------------------------------------
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if a.dtype == "bf16"
           else torch.autocast("cuda", enabled=False))
    OUT["units"] = {}
    for name, g in grabs.items():
        rec = {"sig": g["sig"], "calls_per_fold": sum(counts[name].values()),
               "calls_this_shape": counts[name][g["sig"]]}

        def call():
            with ctx:
                g["obj"](*g["args"], **g["kwargs"])

        try:
            call()
            torch.cuda.synchronize()
            cen = Census()
            with cen:
                call()
            torch.cuda.synchronize()
            top = sorted(cen.by_op.items(), key=lambda kv: -kv[1]["bytes"])[:15]
            rec["census"] = {"ops": cen.ops, "view_ops": cen.views,
                             "dispatches": cen.ops + cen.views, "bytes": cen.bytes,
                             "bytes_GB": round(cen.bytes / 1e9, 4),
                             "mean_op_MB": round(cen.bytes / max(cen.ops, 1) / 1e6, 4),
                             "median_op_MB": round(st.median(cen.sizes) / 1e6, 4),
                             "top_ops": [{"op": k, **v} for k, v in top],
                             "op_MB_sorted": sorted(round(b / 1e6, 4) for b in cen.sizes)}
            print(f"  {name}: {cen.ops} ops, {cen.bytes/1e9:.4f} GB", flush=True)
        except Exception as e:                                              # noqa: BLE001
            rec["census"] = {"error": f"{type(e).__name__}: {e}"}
        dump()

        for label, fn in (("eager", eager_ms), ("kernel", kernel_ms), ("graph", graph_ms)):
            try:
                rec[label] = fn(call)
            except Exception as e:                                          # noqa: BLE001
                rec[label] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {name}: {label} {rec[label]}", flush=True)
            OUT["units"][name] = rec
            dump()
        OUT["units"][name] = rec
        dump()

    OUT["env_after"] = gpu_state()
    dump()
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
