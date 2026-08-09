#!/usr/bin/env python3
"""D4 -- is the diffusion stage's per-call floor recoverable by ttnn trace capture?

One process, one trunk. The trunk conditioning `cond` is built ONCE and the SAME dict is
handed to `edm_sample` with `trace=False` and `trace=True`, so the A/B differs in exactly
one thing: whether the per-step denoise device stream is dispatched op by op from host or
replayed from a captured trace.

RNG: `edm_sample` re-seeds the global torch RNG at sampler entry (`torch.manual_seed(seed)`),
and neither denoise path consumes the host RNG, so both arms draw the identical noise
sequence (initial noise, per-step augmentation, per-step eps). The initial-noise frame is
hashed on both arms and compared, so shared draws are a checked fact, not an assumption
(memory diffusion-port-parity-shared-draws).

Two-gate rule (memory ttnn-trace-interleaved-eager-corruption): the traced arm is compared to
the eager arm at EVERY step of the trajectory, not just at the end, and an eager run is then
repeated after the trace exists to check the eager path was not corrupted by the trace's
buffer pool.

    TT_VISIBLE_DEVICES=2 python3 perf/diff_trace/trace_ab.py \
        --model protenix-v2 --target examples/prot300.yaml \
        --a3m scripts/gpu_vs_tt/fixtures/prot300.a3m \
        --out perf/diff_trace/trace_ab_protenix-v2_298aa.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RECYCLING_STEPS = 10
SEED = 0

# ---------------------------------------------------------------------------------------
# ttnn call counting
# ---------------------------------------------------------------------------------------
_COUNT = {"on": False, "n": 0, "by": {}}


def _wrap_ns(ns, names):
    installed = []
    for nm in names:
        fn = getattr(ns, nm, None)
        if fn is None or not callable(fn):
            continue

        def mk(fn=fn, nm=nm):
            def w(*a, **k):
                if _COUNT["on"]:
                    _COUNT["n"] += 1
                    _COUNT["by"][nm] = _COUNT["by"].get(nm, 0) + 1
                return fn(*a, **k)
            return w
        try:
            setattr(ns, nm, mk())
            installed.append(nm)
        except Exception:                                        # noqa: BLE001
            pass
    return installed


TOP_OPS = ["linear", "matmul", "add", "add_", "multiply", "multiply_", "subtract", "div",
           "layer_norm", "rms_norm", "softmax", "permute", "transpose", "concat", "reshape",
           "typecast", "to_layout", "to_memory_config", "clone", "relu", "sigmoid", "silu",
           "gelu", "sum", "mean", "reciprocal", "pad", "unsqueeze", "squeeze", "repeat",
           "slice", "reallocate", "deallocate", "from_torch", "to_torch",
           "copy_host_to_device_tensor", "execute_trace", "sqrt", "rsqrt", "exp", "tanh",
           "where", "eq", "ne", "gt", "lt", "logical_and", "bcast", "embedding", "arange",
           "zeros", "ones", "full", "argmax", "max", "min", "abs", "neg", "pow"]
EXP_OPS = ["minimal_matmul", "nlp_create_qkv_heads", "nlp_concat_heads"]
TF_OPS = ["scaled_dot_product_attention", "concatenate_heads", "split_query_key_value_and_split_heads"]


def install_counters():
    a = _wrap_ns(ttnn, TOP_OPS)
    b = _wrap_ns(getattr(ttnn, "experimental", None), EXP_OPS) if hasattr(ttnn, "experimental") else []
    c = _wrap_ns(getattr(ttnn, "transformer", None), TF_OPS) if hasattr(ttnn, "transformer") else []
    return {"ttnn": a, "experimental": b, "transformer": c}


# ---------------------------------------------------------------------------------------
def dispatch_floor(dev, reps=4000):
    """Per-call floor: the time it takes to get one trivial ttnn op through the pipe when
    the device cannot be the bottleneck. One 32x32 tile add, issued back to back, one
    synchronise on each side of the whole burst. This is max(host issue, device exec) for
    the smallest real op, i.e. the number a 202k-call fold multiplies by."""
    a = ttnn.from_torch(torch.randn(1, 1, 32, 32), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(torch.randn(1, 1, 32, 32), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = []
    for _ in range(5):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.deallocate(ttnn.add(a, b))
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / reps)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return {"us_per_call": round(st.median(out) * 1e6, 3),
            "spread_us": [round(min(out) * 1e6, 3), round(max(out) * 1e6, 3)], "reps": reps}


def roofs(dev):
    """Compute / DRAM-read / DRAM-write roofs on THIS card, same method as W1's roofs_card.py
    (square bf16 matmul HiFi4; DRAM->L1 clone; L1->DRAM clone). Roofs are per-card."""
    from tt_bio.tenstorrent import CORE_GRID_MAIN
    DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)

    def timed(fn, warm=4, pipe=4, reps=5):
        for _ in range(warm):
            fn()
        ttnn.synchronize_device(dev)
        o = []
        for _ in range(reps):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(pipe):
                fn()
            ttnn.synchronize_device(dev)
            o.append((time.perf_counter() - t0) / pipe)
        return st.median(o)

    res = {}
    best = 0.0
    for n in (2048, 4096):
        a = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        b = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=ckc,
                                                      memory_config=DRAM, core_grid=CORE_GRID_MAIN)))
        tf = 2 * n ** 3 / s / 1e12
        res[f"matmul_bf16_{n}_TFLOPs"] = round(tf, 2)
        best = max(best, tf)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    res["compute_roof_bf16_TFLOPs"] = round(best, 2)
    # read: DRAM-interleaved source -> L1 clone (DRAM sees reads only)
    # write: L1 source -> DRAM clone (DRAM sees writes only)
    for tag, src_mc, dst_mc, mb in (("read", DRAM, L1, 24), ("write", L1, DRAM, 24)):
        rows = mb * 1024 * 1024 // (2 * 4096)
        t = ttnn.from_torch(torch.randn(1, 1, rows, 4096), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=src_mc)
        nbytes = rows * 4096 * 2
        s = timed(lambda: ttnn.deallocate(ttnn.clone(t, memory_config=dst_mc)))
        res[f"dram_{tag}_GBs"] = round(nbytes / s / 1e9, 1)
        res[f"dram_{tag}_MB"] = round(nbytes / 1e6, 1)
        ttnn.deallocate(t)
    res["machine_balance_FLOP_per_byte"] = round(best * 1e12 / (res["dram_read_GBs"] * 1e9), 1)
    return res


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix-v2")
    ap.add_argument("--target", default="examples/prot300.yaml")
    ap.add_argument("--a3m", default="scripts/gpu_vs_tt/fixtures/prot300.a3m")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-roofs", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    from tt_bio.tenstorrent import get_device, arch_name
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    from tt_bio import protenix as P

    _noop = lambda *a, **k: None                                  # noqa: E731
    _E.set_progress(_noop)

    dev = get_device(trace_region_size=1 << 30)
    R = {"model": args.model, "target": args.target, "steps": args.steps,
         "arch": arch_name(), "ttnn": getattr(ttnn, "__version__", "?"),
         "trace_region_bytes": 1 << 30}
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        out_p.write_text(json.dumps(R, indent=2))

    print("=== dispatch floor ===", flush=True)
    R["dispatch_floor"] = dispatch_floor(dev)
    print("  ", R["dispatch_floor"], flush=True)
    flush()

    if not args.skip_roofs:
        print("=== roofs on this card ===", flush=True)
        R["roofs"] = roofs(dev)
        print("  ", R["roofs"], flush=True)
        flush()

    # --- load model, seed MSA cache, featurise once ------------------------------------
    print("=== load model ===", flush=True)
    sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))
    from tt_baseline import seed_msa_cache                        # noqa: E402

    work = Path(tempfile.mkdtemp(prefix=f"d4-{args.model}-"))
    msa_dir = work / "msa"
    struct_dir = work / "out"
    struct_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(model=args.model, fast=False, output_format="cif",
               recycling_steps=RECYCLING_STEPS, sampling_steps=args.steps,
               diffusion_samples=1, seed=SEED, trace=False,
               msa_dir=str(msa_dir), struct_dir=str(struct_dir),
               use_msa_server=True, msa_db_path=None, use_envdb=False, msa_endpoint=None,
               single_sequence=False, msa_server_url="https://api.colabfold.com",
               msa_pairing_strategy="greedy", msa_server_username=None,
               msa_server_password=None, api_key_value=None, max_msa_seqs=8192,
               write_pae=False, write_pde=False, write_embeddings=False, method=None)
    _ensure_local_artifacts(cfg)
    seed_msa_cache(REPO / args.target, REPO / args.a3m, msa_dir)
    state = _WorkerState("tenstorrent")
    t0 = time.perf_counter()
    state.load_model(cfg)
    state.bind_run("d4", cfg)
    state.pfn = _noop
    R["load_s"] = round(time.perf_counter() - t0, 2)
    model = state.model

    from tt_bio.main import _read_bio_chains, _read_bio_constraints, _resolve_a3m_text
    from tt_bio.protenix_data import build_complex_features
    chains = _read_bio_chains(REPO / args.target)
    bonds = _read_bio_constraints(REPO / args.target)
    chain_specs = [(cseq, _resolve_a3m_text(spec, cseq, msa_dir) if mt == "protein" else None, mt)
                   for _cid, cseq, spec, mt in chains]
    feats = build_complex_features(chain_specs, mol_dir=cfg.get("mol_dir"),
                                   chain_ids=[cid for cid, _s, _sp, _mt in chains], bonds=bonds)

    print("=== trunk (once) ===", flush=True)
    t0 = time.perf_counter()
    cond, aux = model._trunk_cond(feats, progress_fn=_noop, n_cycles=RECYCLING_STEPS)
    ttnn.synchronize_device(dev)
    R["trunk_s"] = round(time.perf_counter() - t0, 3)
    N_atoms = cond["c_l"].shape[0]
    R["N_atoms"] = int(N_atoms)
    R["NT_tokens"] = int(aux["NT"])
    R["device_dit"] = bool(model.diffusion.device_dit)
    R["dit_z_present"] = cond.get("dit_z") is not None
    print(f"  atoms={N_atoms} tokens={aux['NT']} device_dit={R['device_dit']} "
          f"dit_z={R['dit_z_present']} trunk={R['trunk_s']}s", flush=True)
    flush()

    # --- capture-cost instrumentation --------------------------------------------------
    dm = model.diffusion
    cap = {"s": 0.0, "n": 0}
    _orig_capture = dm._capture_trace

    def _timed_capture(*a, **k):
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        try:
            return _orig_capture(*a, **k)
        finally:
            ttnn.synchronize_device(dev)
            cap["s"] += time.perf_counter() - t
            cap["n"] += 1
    dm._capture_trace = _timed_capture

    def sample(trace, steps, dump=None):
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        x = P.edm_sample(dm, cond, N_atoms, n_step=steps, seed=SEED, trace=trace, dump_fn=dump)
        ttnn.synchronize_device(dev)
        return time.perf_counter() - t, x

    # --- warm both arms (kernel compile, cond caches, trace capture) --------------------
    print("=== warm ===", flush=True)
    w0, _ = sample(False, 3)
    cap["s"] = 0.0; cap["n"] = 0
    w1, _ = sample(True, 3)
    R["warm"] = {"eager_3step_s": round(w0, 3), "traced_3step_s": round(w1, 3),
                 "capture_s": round(cap["s"], 3), "captures": cap["n"]}
    print("  ", R["warm"], flush=True)
    flush()

    # --- A/B, full trajectories --------------------------------------------------------
    # NOTE: the ttnn call counters are installed LAST, after every timed region. A counting
    # wrapper on 1900 ops/step is a per-op host cost that lands on the eager arm only, so
    # installing it before the A/B would inflate exactly the arm under test.
    print("=== A/B ===", flush=True)

    def collect():
        buf = {}
        return buf, (lambda k, x: buf.__setitem__(k, x.clone()))

    eager_traj, eager_dump = collect()
    eager_t = []
    for r in range(args.reps):
        d = eager_dump if r == 0 else None
        s, x_e = sample(False, args.steps, dump=d)
        eager_t.append(s)
        print(f"  eager rep{r}: {s:.3f} s", flush=True)

    trace_traj, trace_dump = collect()
    trace_t = []
    cap["s"] = 0.0; cap["n"] = 0
    for r in range(args.reps):
        d = trace_dump if r == 0 else None
        s, x_t = sample(True, args.steps, dump=d)
        trace_t.append(s)
        print(f"  traced rep{r}: {s:.3f} s", flush=True)
    R["recapture_during_ab"] = {"s": round(cap["s"], 4), "n": cap["n"]}

    # eager AFTER the trace exists -- the interleaving gate
    s, x_e2 = sample(False, args.steps)
    R["eager_after_trace_s"] = round(s, 3)
    print(f"  eager-after-trace: {s:.3f} s", flush=True)

    R["stage_s"] = {"eager": [round(v, 4) for v in eager_t],
                    "traced": [round(v, 4) for v in trace_t],
                    "eager_median": round(st.median(eager_t), 4),
                    "traced_median": round(st.median(trace_t), 4)}
    e_med, t_med = st.median(eager_t), st.median(trace_t)
    R["stage_s"]["ratio_eager_over_traced"] = round(e_med / t_med, 4)
    R["stage_s"]["delta_ms_per_fold"] = round((e_med - t_med) * 1e3, 1)
    R["capture_cost_s"] = round(R["warm"]["capture_s"], 4)

    # --- parity ------------------------------------------------------------------------
    def h(t):
        return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()[:12]

    par = {"noise_hash_eager": h(eager_traj[-1]), "noise_hash_traced": h(trace_traj[-1])}
    par["shared_draws"] = par["noise_hash_eager"] == par["noise_hash_traced"]
    par["final_bit_exact"] = bool(torch.equal(x_e, x_t))
    par["final_max_abs_delta"] = float((x_e - x_t).abs().max())
    par["eager_repeat_bit_exact"] = bool(torch.equal(x_e, x_e2))
    par["eager_repeat_max_abs_delta"] = float((x_e - x_e2).abs().max())
    per_step = []
    for k in sorted(kk for kk in eager_traj if kk >= 0):
        a, b = eager_traj[k], trace_traj[k]
        d = (a - b).abs().max().item()
        av, bv = a.flatten().double(), b.flatten().double()
        pcc = float(torch.corrcoef(torch.stack([av, bv]))[0, 1]) if d > 0 else 1.0
        per_step.append({"step": k, "max_abs_delta": d, "pcc": pcc,
                         "bit_exact": bool(torch.equal(a, b))})
    par["steps_bit_exact"] = sum(1 for s_ in per_step if s_["bit_exact"])
    par["steps_total"] = len(per_step)
    par["worst_step"] = max(per_step, key=lambda s_: s_["max_abs_delta"]) if per_step else None
    par["min_pcc"] = min((s_["pcc"] for s_ in per_step), default=None)
    par["trajectory"] = per_step[::10]
    R["parity"] = par
    print("  parity:", {k: v for k, v in par.items() if k != "trajectory"}, flush=True)
    flush()

    # --- decompose the step: pure device denoise, vs denoise wall, vs stage -------------
    # A burst of execute_trace with no host work and no sync between replays is the denoise
    # step's DEVICE time with every host cost removed -- the hard floor a perfect host would
    # still have to pay. Comparing it to the traced step wall prices what the per-step host
    # round trip (to_torch of r_update, the EDM update, the two uploads) costs on top.
    print("=== device-only replay burst ===", flush=True)
    tr = dm._trace
    burst = []
    for _ in range(5):
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        for _ in range(20):
            ttnn.execute_trace(dev, tr["tid"], cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        burst.append((time.perf_counter() - t) / 20)
    R["device_replay_ms"] = round(st.median(burst) * 1e3, 4)
    print(f"  pure device denoise replay: {R['device_replay_ms']:.3f} ms/step", flush=True)

    # per-step denoise wall, sync on both sides, both arms (steps 20-24, warm)
    print("=== per-step denoise wall ===", flush=True)
    walls = {}
    for tag, fn_name in (("eager", "denoise"), ("traced", "denoise_traced")):
        rec = []
        orig = getattr(dm, fn_name)

        def timed(*a, _orig=orig, _rec=rec, **k):
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            r = _orig(*a, **k)
            ttnn.synchronize_device(dev)
            _rec.append(time.perf_counter() - t)
            return r
        setattr(dm, fn_name, timed)
        try:
            sample(tag == "traced", 25)
        finally:
            setattr(dm, fn_name, orig)
        walls[tag] = {"ms_median_steps20_24": round(st.median(rec[20:25]) * 1e3, 4),
                      "ms_all_median": round(st.median(rec) * 1e3, 4)}
        print(f"  {tag}: {walls[tag]}", flush=True)
    R["denoise_wall"] = walls
    e_step = R["stage_s"]["eager_median"] / args.steps * 1e3
    t_step = R["stage_s"]["traced_median"] / args.steps * 1e3
    R["step_budget_ms"] = {
        "eager_stage_per_step": round(e_step, 3),
        "traced_stage_per_step": round(t_step, 3),
        "eager_denoise_wall": walls["eager"]["ms_median_steps20_24"],
        "traced_denoise_wall": walls["traced"]["ms_median_steps20_24"],
        "device_only_replay": R["device_replay_ms"],
        "host_sampler_per_step_eager": round(e_step - walls["eager"]["ms_median_steps20_24"], 3),
        "host_roundtrip_per_step_traced": round(
            walls["traced"]["ms_median_steps20_24"] - R["device_replay_ms"], 3),
    }
    print("  ", R["step_budget_ms"], flush=True)
    flush()

    # --- ttnn call counts per step, both arms (LAST: the wrappers cost host time) -------
    print("=== call counts ===", flush=True)
    inst = install_counters()
    counts = {}
    for tag, tr_ in (("eager", False), ("traced", True)):
        _COUNT["on"] = True; _COUNT["n"] = 0; _COUNT["by"] = {}
        sample(tr_, 4)
        _COUNT["on"] = False
        counts[tag] = {"per_step": round(_COUNT["n"] / 4, 1),
                       "per_fold": round(_COUNT["n"] / 4 * args.steps, 1),
                       "top": dict(sorted(_COUNT["by"].items(), key=lambda kv: -kv[1])[:12])}
    R["wrapped_ops"] = {k: len(v) for k, v in inst.items()}
    R["calls"] = counts
    for tag in counts:
        print(f"  {tag}: {counts[tag]['per_step']} ttnn calls/step -> "
              f"{counts[tag]['per_fold']:.0f} per fold", flush=True)
    fl = R["dispatch_floor"]["us_per_call"]
    R["dispatch_floor_share"] = {
        "eager_floor_ms_per_step": round(counts["eager"]["per_step"] * fl / 1e3, 3),
        "eager_step_wall_ms": R["step_budget_ms"]["eager_denoise_wall"],
        "pct_of_step_wall": round(counts["eager"]["per_step"] * fl / 1e3
                                  / R["step_budget_ms"]["eager_denoise_wall"] * 100, 1),
    }
    print("  ", R["dispatch_floor_share"], flush=True)
    flush()
    print("wrote", out_p, flush=True)


if __name__ == "__main__":
    main()
