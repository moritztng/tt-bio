#!/usr/bin/env python3
"""GPU leg of the GPU-vs-Tenstorrent Protenix-v2 / OpenDDE head-to-head.

Runs the UPSTREAM reference implementations (bytedance/Protenix, aurekaresearch
OpenDDE) on an NVIDIA GPU with the model loaded once in-process: one cold fold
(CUDA init / autotune / first-kernel compile, reported separately), then N warm
timed folds (min/median/max). Same target, same precomputed MSA, same config as
the TT leg (scripts/gpu_vs_tt/tt_baseline.py): 10 cycles / 200 diffusion steps /
1 sample / seed 0, over whichever of the two committed targets is selected --
prot117 (117 aa) or prot300 (CDK2, 298 aa), both with a 35-row alignment so that
token count is the only thing that changes between them.

The optimization ladder is vendor-sanctioned only: dtype (fp32 -> bf16), the
upstream triangle-kernel selectors (triattention / cuequivariance), Protenix's
built-in fusion/cache flags, TF32. Each rung is correctness-checked against the
fp32-eager rung's coordinates (CA Kabsch RMSD) -- a fast but wrong rung is
reported as INVALID, not as a speedup.

Usage (on the GPU box, after gpu_setup.sh):

    python3 gpu_bench.py --model protenix-v2 --repeat 3 --out protenix_gpu.json
    python3 gpu_bench.py --model opendde --repeat 3 --out opendde_gpu.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"

CYCLES = 10
STEPS = 200
SAMPLES = 1
SEED = 0

# Optimization ladder, per model. Each rung sets EVERY performance knob
# explicitly: both vendors default fusion/cache/tf32 to ON for inference, so an
# unset flag is not a clean eager reference. "fp32+torch kernels, no fusion" is
# the eager reference point; the later rungs are the vendor's own published
# performance knobs. Kernel selector names differ between the two codebases
# (protenix: triattention/cuequivariance; opendde: cuequivariance/auto).
#
# For protenix 2.0.0 the "defaults to ON for inference" claim is exact, and the
# citation matters because configs_base.py says the opposite. configs_base.py is
# the TRAINING config (fusion/cache/tf32 all False, l.131-133); inference merges
# configs_inference.py on top of it, which sets all three True (l.32-34), and the
# shipped `protenix predict` CLI defaults match (runner/batch_inference.py
# l.296-299, l.643-660). LD-shipped-default below leaves those three keys OUT of
# the rung dict entirely, so _build_configs never passes them and the resolved
# value comes from upstream; the resolved config is recorded per rung so this is
# auditable rather than asserted.
LADDERS = {
    "protenix-v2": [
        dict(name="L0-eager-fp32", dtype="fp32", triangle_attention="torch",
             triangle_multiplicative="torch", enable_efficient_fusion=False,
             enable_diffusion_shared_vars_cache=False, enable_tf32=False),
        dict(name="L1-bf16-vendor-kernels", dtype="bf16",
             triangle_attention="triattention", triangle_multiplicative="cuequivariance",
             enable_efficient_fusion=False, enable_diffusion_shared_vars_cache=False,
             enable_tf32=False),
        dict(name="L2-bf16-fusion-cache", dtype="bf16",
             triangle_attention="triattention", triangle_multiplicative="cuequivariance",
             enable_efficient_fusion=True, enable_diffusion_shared_vars_cache=True,
             enable_tf32=True),
        # What `protenix predict --model_name protenix-v2 --use_default_params true`
        # runs. The only knob it moves against L2 is the triangle-attention kernel,
        # which upstream defaults to cuequivariance and L2 sets to triattention.
        dict(name="LD-shipped-default", dtype="bf16",
             triangle_attention="cuequivariance",
             triangle_multiplicative="cuequivariance"),
        # configs_base.py's values for the three flags, i.e. the training config,
        # with the default kernels. Not a shipped inference default -- it exists
        # only to price what fusion/cache/tf32 are worth at these sizes.
        dict(name="LB-basecfg-flags-off", dtype="bf16",
             triangle_attention="cuequivariance",
             triangle_multiplicative="cuequivariance",
             enable_efficient_fusion=False, enable_diffusion_shared_vars_cache=False,
             enable_tf32=False),
    ],
    "opendde": [
        dict(name="L0-eager-fp32", dtype="fp32", triatt_kernel="torch",
             trimul_kernel="torch", enable_fusion=False, enable_cache=False,
             enable_tf32=False),
        dict(name="L1-bf16-vendor-kernels", dtype="bf16",
             triatt_kernel="cuequivariance", trimul_kernel="cuequivariance",
             enable_fusion=False, enable_cache=False, enable_tf32=False),
        dict(name="L2-bf16-fusion-cache", dtype="bf16", triatt_kernel="auto",
             trimul_kernel="auto", enable_fusion=True, enable_cache=True,
             enable_tf32=True),
    ],
}


def _import_any(names: list[str]):
    for n in names:
        try:
            return importlib.import_module(n)
        except ImportError:
            continue
    raise ImportError(f"none of {names} importable")


def _build_configs(model_name: str, rung: dict, input_json: str, dump_dir: str,
                   checkpoint: str | None, samples: int = SAMPLES):
    """Replicate runner/inference.py::run()'s 3-pass config construction."""
    cb = _import_any(["configs.configs_base", "protenix.configs.configs_base"])
    cd = _import_any(["configs.configs_data", "protenix.configs.configs_data"])
    ci = _import_any(["configs.configs_inference", "protenix.configs.configs_inference"])
    cm = _import_any(["configs.configs_model_type", "protenix.configs.configs_model_type"])
    cc = _import_any(["protenix.config.config", "opendde.config.config"])
    from typing import Mapping

    arg_str = [
        "--model_name", model_name,
        "--input_json_path", input_json,
        "--dump_dir", dump_dir,
        "--seeds", str(SEED),
        "--model.N_cycle", str(CYCLES),
        "--sample_diffusion.N_step", str(STEPS),
        "--sample_diffusion.N_sample", str(samples),
        "--use_msa", "true", "--use_template", "false", "--use_rna_msa", "false",
        # The v2 checkpoint ships template_embedder weights; with templates off
        # the module is not built, so the sanctioned load is non-strict.
        "--load_strict", "false",
        "--dtype", rung["dtype"],
        "--triangle_attention", rung["triangle_attention"],
        "--triangle_multiplicative", rung["triangle_multiplicative"],
    ]
    if checkpoint:
        arg_str += ["--load_checkpoint_path", checkpoint]
        # protenix 2.0.0's runner loads load_checkpoint_dir/<model_name>.pt and
        # download_inference_cache 403s on the gated official URL if it is
        # absent there; point the dir at the local copy when the name matches.
        ck = Path(checkpoint)
        if ck.name == f"{model_name}.pt":
            arg_str += ["--load_checkpoint_dir", str(ck.parent)]
    for k in ("enable_efficient_fusion", "enable_diffusion_shared_vars_cache",
              "enable_tf32"):
        if k in rung:
            arg_str += [f"--{k}", str(rung[k]).lower()]

    # protenix 2.0.0 parse_configs does arg_str.split() -- it wants one string.
    arg_str = " ".join(arg_str)
    configs = {**cb.configs, **{"data": cd.data_configs}, **ci.inference_configs}
    configs = cc.parse_configs(configs=configs, arg_str=arg_str,
                               fill_required_with_null=True)
    base = {**cb.configs, **{"data": cd.data_configs}, **ci.inference_configs}

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base, cm.model_configs[configs.model_name])
    configs = cc.parse_configs(configs=base, arg_str=arg_str,
                               fill_required_with_null=True)
    return configs


def _install_vendor_cueq_counters() -> dict:
    """Count entries into cuequivariance_ops_torch's triangle kernels directly.

    Model-agnostic, because it wraps the vendor package instead of a model's call
    site. Re-entrant: run_model calls this once per rung, and the _bench_orig_
    stash means a second call re-wraps the ORIGINAL rather than nesting another
    layer on top of the previous rung's closure (which would keep the old rung's
    dict counting). Returns {} when the package is absent, so a torch-only model
    reads as "no cueq counters" rather than a crash.
    """
    counts: dict[str, int] = {}
    try:
        mod = importlib.import_module("cuequivariance_ops_torch")
    except ImportError:
        return counts
    for name in dir(mod):
        if "triangle" not in name.lower():
            continue
        orig_attr = f"_bench_orig_{name}"
        fn = getattr(mod, orig_attr, None) or getattr(mod, name, None)
        if not callable(fn):
            continue
        counts[f"cueqops_{name}"] = 0

        def make(key, orig):
            def counted(*a, **kw):
                counts[key] += 1
                return orig(*a, **kw)
            return counted
        try:
            setattr(mod, orig_attr, fn)
            setattr(mod, name, make(f"cueqops_{name}", fn))
        except Exception:
            counts.pop(f"cueqops_{name}", None)
    return counts


def _install_kernel_counters(model: str) -> dict | None:
    """Count the triangle kernels each fold actually calls.

    protenix falls back to torch silently in two places -- cuequivariance trimul
    needs c_hidden == c_z (triangular.py:494) and the cuequivariance triangle
    attention kernel drops to torch for unsupported head dims (layers.py:456-458)
    -- so a rung that ASKS for a kernel is not evidence the kernel ran. Wrapping
    the four call sites turns "we set the flag" into "the kernel was entered N
    times". Returns None for a codebase whose module layout does not match.

    OpenDDE runs on the protenix-v2 stack, so it gets the same call-site wrap; it
    was previously excluded by a literal model-name check and reported None. On top
    of that, every model gets the vendor-package counters, which wrap
    cuequivariance_ops_torch itself rather than one codebase's call site, so a model
    whose layout is unknown still produces evidence instead of an assertion.
    """
    counts = _install_vendor_cueq_counters()
    if model not in ("protenix-v2", "opendde"):
        return counts or None
    try:
        lay = importlib.import_module("protenix.model.triangular.layers")
        tri = importlib.import_module("protenix.model.triangular.triangular")
    except ImportError:
        return counts or None
    counts.update({"cueq_triatt": 0, "cueq_trimul": 0, "triattention": 0, "torch_attn": 0})

    def wrap(mod, attr, key):
        # Restore first: run_model calls this once per rung, and re-wrapping an
        # already-wrapped function would nest one layer per rung and keep the
        # previous rung's dict counting.
        orig_attr = f"_bench_orig_{attr}"
        fn = getattr(mod, orig_attr, None) or getattr(mod, attr, None)
        if fn is None:
            return
        setattr(mod, orig_attr, fn)
        def counted(*a, **kw):
            counts[key] += 1
            return fn(*a, **kw)
        setattr(mod, attr, counted)

    wrap(lay, "cuequivariance_triangular_attn", "cueq_triatt")
    wrap(lay, "_tri_attention", "triattention")
    wrap(lay, "_attention", "torch_attn")
    wrap(tri, "kernel_triangular_mult", "cueq_trimul")
    return counts


def _resolved_knobs(configs) -> dict:
    """The knobs as upstream's config machinery actually resolved them."""
    out = {}
    for k in ("model_name", "dtype", "triangle_attention", "triangle_multiplicative",
              "enable_efficient_fusion", "enable_diffusion_shared_vars_cache",
              "enable_tf32"):
        try:
            v = getattr(configs, k)
        except AttributeError:
            v = "<absent>"
        # protenix wraps some config leaves (ValueMaybeNone, GlobalConfigValue);
        # this dict is written straight to JSON, so keep it primitive.
        out[k] = v if isinstance(v, (str, bool, int, float, type(None))) else repr(v)
    try:
        out["skip_amp"] = {k: bool(v) for k, v in dict(configs.skip_amp).items()}
    except Exception:
        out["skip_amp"] = "<absent>"
    return out


def _ca_indices(atom_array, n_atoms: int):
    """CA row indices into the predicted coordinate tensor, or None if the atom
    array does not line up with it (never worth crashing a paid session over)."""
    try:
        names = list(atom_array.atom_name)
    except Exception:
        return None
    if len(names) != n_atoms:
        return None
    idx = [i for i, n in enumerate(names) if n == "CA"]
    return idx or None


def _confidence(pred) -> dict:
    """Whatever confidence the vendor reports, as plain numbers.

    Rule 6 of the five-model benchmark: a rung that fails its own accuracy check is not a
    speed data point, so every run has to carry a confidence read. protenix returns these
    in the prediction dict, so they cost nothing to record -- the alternative (reading
    plDDT back off the B-factor column of a written file) is a lossier route to the same
    number.
    """
    import numpy as np
    out = {}
    for k, v in (pred or {}).items():
        if not any(t in k.lower() for t in ("plddt", "ptm", "pae", "confidence", "resolved")):
            continue
        try:
            a = np.asarray(v.cpu() if hasattr(v, "cpu") else v, dtype=np.float64)
        except Exception:
            continue
        if not a.size:
            continue
        out[f"{k}_shape"] = list(a.shape)
        out[k] = round(float(a.mean()), 6)
    # protenix reports plDDT as per-atom BIN LOGITS, not as a score: 8234 atoms x 25 bins
    # on this fixture. Averaging those raw gives a negative "plDDT", which is how the
    # first run of this benchmark read 0.0 through the B-factor column. Softmax over the
    # bin axis against bin centres is the score every folding tool prints.
    pl = _plddt_from_logits(pred)
    if pl is not None:
        out["plddt_score_mean"] = round(float(pl.mean()), 6)
        out["plddt_score_n"] = int(pl.size)
    return out


def _plddt_from_logits(pred):
    """Per-atom plDDT in 0-1 from whatever plddt tensor the vendor returned, or None.

    Handles both conventions: an already-scored [N] vector in 0-1 (or 0-100), and the
    [N, n_bins] logits protenix and OpenDDE return. Bin centres are (i+0.5)/n_bins, the
    AF2/AF3 convention these heads are trained against.
    """
    import numpy as np
    v = (pred or {}).get("plddt")
    if v is None:
        return None
    try:
        a = np.asarray(v.cpu() if hasattr(v, "cpu") else v, dtype=np.float64)
    except Exception:
        return None
    a = a.reshape(-1, a.shape[-1]) if a.ndim > 2 else a
    if a.ndim == 1:
        return a / 100.0 if a.max() > 1.5 else a
    if a.ndim != 2 or a.shape[-1] < 2:
        return None
    nb = a.shape[-1]
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    centres = (np.arange(nb) + 0.5) / nb
    return p @ centres


def _save_structure(atom_array, coords, pred, path: Path) -> str:
    """Write the predicted structure so the shared accuracy gate can read it.

    gpu_bench.py keeps coordinates in memory and never wrote a file, which was fine while
    the only correctness check was a rung-vs-rung RMSD inside one process. The five-model
    benchmark needs a structure per cell, checked by the same gate for all five models.
    """
    import numpy as np
    c = np.asarray(coords.cpu() if hasattr(coords, "cpu") else coords, dtype=np.float32)
    while c.ndim > 2:
        c = c[0]
    try:
        aa = atom_array.copy()
    except Exception as e:
        return f"no atom array: {e}"
    if len(aa) != c.shape[0]:
        return f"atom count mismatch: array {len(aa)} vs coords {c.shape[0]}"
    aa.coord = c
    # Per-atom plDDT into the B-factor column, which is where the gate looks and where
    # every folding tool puts it. Skip silently if the model reports it per-token instead.
    pl = _plddt_from_logits(pred)
    if pl is not None:
        b = np.asarray(pl, dtype=np.float32).reshape(-1)
        if b.size == len(aa):
            aa.set_annotation("b_factor", b)
    path.parent.mkdir(parents=True, exist_ok=True)
    import biotite.structure.io.pdb as pdb
    f = pdb.PDBFile()
    f.set_structure(aa)
    f.write(str(path))
    return str(path)


def _kabsch_rmsd(c1, c2, idx=None) -> float:
    import numpy as np
    a = np.asarray(c1, dtype=np.float64)
    b = np.asarray(c2, dtype=np.float64)
    # Vendors disagree on leading dims ([N,3] vs [N_sample,N,3]); flatten to
    # (N,3) and compare the first sample's atoms.
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1])
    if idx is not None:
        a = a[idx]
        b = b[idx]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a = a - a.mean(0)
    b = b - b.mean(0)
    H = a.T @ b
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    rot = Vt.T @ D @ U.T
    return float(np.sqrt(((a @ rot.T - b) ** 2).sum(-1).mean()))


def _build_opendde_runner(rung: dict, input_json: str, dump_dir: str,
                          checkpoint: str | None, samples: int = SAMPLES):
    """OpenDDE's public runner factory (runner.batch_inference.get_default_runner)
    with the vendor's own knob names. Returns (runner, configs, inf_module,
    dataloader_module, seed_module)."""
    bi = importlib.import_module("runner.batch_inference")
    inf = importlib.import_module("runner.inference")
    dl = importlib.import_module("opendde.data.inference.infer_dataloader")
    sd = importlib.import_module("opendde.utils.seed")
    runner = bi.get_default_runner(
        seeds=[SEED], dump_dir=dump_dir, n_cycle=CYCLES, n_step=STEPS,
        n_sample=samples, dtype=rung["dtype"], model_name="opendde_v1",
        load_checkpoint_path=checkpoint or "", use_msa=True,
        trimul_kernel=rung["trimul_kernel"], triatt_kernel=rung["triatt_kernel"],
        enable_cache=rung["enable_cache"], enable_fusion=rung["enable_fusion"],
        enable_tf32=rung["enable_tf32"], deterministic=False, use_template=False,
        use_rna_msa=False, need_atom_confidence=True,
    )
    configs = runner.configs
    configs.input_json_path = input_json
    return runner, configs, inf, dl, sd


def build_fold(model: str, model_name: str, rung: dict, input_json: str,
               dump_dir: Path, checkpoint: str | None, n_msa_expected: int,
               samples: int = SAMPLES):
    """Load one ladder rung and return ``(one_fold, meta, runner)``.

    Split out of ``run_model`` so the concurrency launcher (``gpu_concurrency.py``)
    folds through exactly the same code path as the latency benchmark: an aggregate
    throughput number is only comparable to the committed per-fold latency if the fold
    itself is identical.
    """
    torch = importlib.import_module("torch")
    dump_dir.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    if model == "opendde":
        runner, configs, inf, dl, sd = _build_opendde_runner(
            rung, input_json, str(dump_dir), checkpoint, samples=samples)
    else:
        inf = _import_any(["runner.inference", "protenix.runner.inference"])
        dl = _import_any(["protenix.data.inference.infer_dataloader"])
        sd = _import_any(["protenix.utils.seed"])
        configs = _build_configs(model_name, rung, input_json, str(dump_dir),
                                 checkpoint, samples=samples)
        if hasattr(inf, "update_gpu_compatible_configs"):
            configs = inf.update_gpu_compatible_configs(configs)
        inf.download_inference_cache(configs)
        runner = inf.InferenceRunner(configs)
    load_s = time.perf_counter() - t_load

    # Host featurization, timed. It sits OUTSIDE the timed fold below and outside the
    # published cell's scope, which is exactly why it has to be recorded: the TT side's
    # cell is a whole fold and pays this every target, this side hoists it out of the
    # loop and pays it once. Timing it changes nothing about the fold.
    t_feat = time.perf_counter()
    dataloader = dl.get_inference_dataloader(configs=configs)
    data, atom_array, err = next(iter(dataloader))[0]
    featurize_s = time.perf_counter() - t_feat
    assert not err, f"featurization failed: {err}"
    # Featurize a second time and keep both. The first call is cold -- it loads the CCD
    # dictionary and every other process-level cache -- and the TT side's featurization
    # is measured on warm folds, so comparing the two would compare a cold number to a
    # warm one. The second call is the per-target steady-state cost and is what goes
    # against the TT figure; the first is what a one-shot fold from a cold process pays.
    t_feat2 = time.perf_counter()
    _d2, _a2, err2 = next(iter(dl.get_inference_dataloader(configs=configs)))[0]
    featurize_warm_s = time.perf_counter() - t_feat2
    assert not err2, f"second featurization failed: {err2}"
    del _d2, _a2
    new_configs = inf.update_inference_configs(configs, data["N_token"].item())
    runner.update_model_configs(new_configs)
    # AFTER update_inference_configs, because that is where skip_amp is decided
    # from the token count (inference.py:396-414) -- reading it earlier would
    # record a value the fold does not run.
    resolved = _resolved_knobs(new_configs)
    counters = _install_kernel_counters(model)
    n_msa = int(data["N_msa"].item())
    n_token = int(data["N_token"].item())
    # Fairness is only real if both sides consume the SAME alignment rows. The
    # TT side reads all 35; if this side cropped or padded to something else the
    # head-to-head is invalid, so fail loudly rather than publish the number.
    assert n_msa == n_msa_expected, \
        f"GPU consumed {n_msa} MSA rows, TT side uses {n_msa_expected}"

    def one_fold():
        sd.seed_everything(seed=SEED, deterministic=False)
        t0 = time.perf_counter()
        # protenix's forward deletes the MSA keys from input_feature_dict
        # in place (protenix.py:524); hand each fold a fresh shallow copy.
        pred = runner.predict(
            {**data, "input_feature_dict": dict(data["input_feature_dict"])})
        torch.cuda.synchronize()
        return time.perf_counter() - t0, pred

    return one_fold, dict(load_s=round(load_s, 2), featurize_s=round(featurize_s, 4),
                          featurize_warm_s=round(featurize_warm_s, 4),
                          n_msa=n_msa, n_token=n_token,
                          diffusion_samples=samples, resolved_config=resolved,
                          kernel_counts=counters, atom_array=atom_array), runner


def run_model(model: str, model_name: str, repeat: int, input_json: str,
              dump_root: Path, checkpoint: str | None, out_path: Path,
              rungs: list[dict], label: str, n_msa_expected: int,
              save_structure: str | None = None) -> dict:
    torch = importlib.import_module("torch")

    results = []
    ref_coords = None
    ref_ca_idx = None
    for rung in rungs:
        one_fold, meta, runner = build_fold(
            model, model_name, rung, input_json, dump_root / rung["name"],
            checkpoint, n_msa_expected)
        load_s, n_msa, n_token = meta["load_s"], meta["n_msa"], meta["n_token"]

        cold_s, pred = one_fold()
        # Reset after the cold fold so the counts are per-warm-fold, and the
        # cold fold's own compile/autotune path cannot inflate them.
        counters = meta["kernel_counts"]
        if counters is not None:
            for k in counters:
                counters[k] = 0
        times = []
        for _ in range(repeat):
            t, pred = one_fold()
            times.append(t)
        per_fold_kernels = ({k: v // max(1, repeat) for k, v in counters.items()}
                            if counters is not None else None)

        coords = pred["coordinate"]
        if hasattr(coords, "cpu"):
            coords = coords.cpu().numpy()
        rmsd_all, rmsd_ca = None, None
        if ref_coords is None:
            ref_coords = coords
            ref_ca_idx = _ca_indices(meta["atom_array"], coords.reshape(-1, 3).shape[0])
        else:
            try:
                rmsd_all = _kabsch_rmsd(ref_coords, coords)
                if ref_ca_idx is not None:
                    rmsd_ca = _kabsch_rmsd(ref_coords, coords, idx=ref_ca_idx)
            except Exception as e:
                rmsd_all = f"rmsd-error: {e}"

        ts = sorted(times)
        struct, conf = None, _confidence(pred)
        write_s = None
        if save_structure is not None:
            try:
                t_write = time.perf_counter()
                struct = _save_structure(meta["atom_array"], pred["coordinate"], pred,
                                         Path(save_structure) / f"{rung['name']}.pdb")
                write_s = round(time.perf_counter() - t_write, 4)
            except Exception as e:
                struct = f"save-error: {e}"
        results.append(dict(
            rung=rung["name"], rung_config=rung, load_s=load_s,
            # The per-target host cost this harness keeps outside its timed region:
            # featurization is built once before the fold loop and the structure is
            # written once after it, so neither is in warm_median_s.
            host_phases=dict(featurize_s=meta["featurize_s"],
                             featurize_warm_s=meta["featurize_warm_s"], write_s=write_s),
            resolved_config=meta["resolved_config"],
            kernel_calls_per_fold=per_fold_kernels,
            cold_s=round(cold_s, 3), warm_times_s=[round(t, 3) for t in times],
            warm_min_s=round(ts[0], 3), warm_median_s=round(ts[len(ts) // 2], 3),
            warm_max_s=round(ts[-1], 3), n_msa=n_msa, n_token=n_token,
            # The committed 2026-08-07 numbers called this "ca_kabsch_rmsd" but
            # passed no CA index, so it was always all-atom. Kept under its true
            # name, with CA reported separately when the atom array lines up.
            all_atom_kabsch_rmsd_vs_L0=rmsd_all,
            ca_kabsch_rmsd_vs_L0=rmsd_ca,
            confidence=conf, structure=struct,
        ))
        print(f"[{model} {rung['name']}] warm median {ts[len(ts)//2]:.2f}s "
              f"(min {ts[0]:.2f}/max {ts[-1]:.2f}) cold {cold_s:.1f}s "
              f"rmsd_all={rmsd_all} rmsd_ca={rmsd_ca}\n"
              f"    resolved={meta['resolved_config']}\n"
              f"    kernels/fold={per_fold_kernels}", file=sys.stderr, flush=True)
        del runner
        torch.cuda.empty_cache()

    # Host CPU matters: at these token counts a good share of the wall clock is
    # kernel-launch dispatch, so a scaling ratio taken across two different rented
    # hosts is not clean. Recorded here so the 117-vs-300 ratio can be shown to be
    # a within-host measurement (the 117-aa run of 2026-08-06 did not record it).
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass

    summary = dict(
        model=model, side="gpu", model_name=model_name,
        gpu=torch.cuda.get_device_name(0), host_cpu=cpu,
        cpu_count=os.cpu_count(),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        input=f"{label}, precomputed MSA (same a3m as TT side)",
        recycling_steps=CYCLES, sampling_steps=STEPS, diffusion_samples=SAMPLES,
        seed=SEED, rungs=results, date=time.strftime("%Y-%m-%d"),
    )
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_input_json(path: Path, msa_a3m: Path, seq: str, name: str) -> None:
    job = {
        "name": name,
        "modelSeeds": [SEED],
        "sequences": [{
            "proteinChain": {
                "sequence": seq,
                "count": 1,
                "msa": {
                    "precomputed_msa_dir": str(msa_a3m.parent),
                    "pairing_db": "uniref100",
                },
                "unpairedMsaPath": str(msa_a3m),
                "pairedMsaPath": "",
                "templatesPath": "",
            }
        }],
    }
    path.write_text(json.dumps([job], indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["protenix-v2", "opendde"])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--msa-a3m", type=Path, default=FIXTURES / "prot117.a3m")
    ap.add_argument("--seq-file", type=Path, default=FIXTURES / "prot117.seq",
                    help="one-line target sequence; must match the a3m's query row")
    ap.add_argument("--label", default="prot.yaml sequence (117 aa)")
    ap.add_argument("--name", default="prot117", help="job name in the input JSON")
    ap.add_argument("--checkpoint", default=None,
                    help="local checkpoint path (skips the gated official download)")
    ap.add_argument("--rungs", default=None,
                    help="comma-separated subset of rung names (default: all)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-structure", default=None,
                    help="directory for one PDB per rung, for the shared accuracy gate")
    args = ap.parse_args()

    model_name = {"protenix-v2": "protenix-v2", "opendde": "opendde_v1"}[args.model]
    work = Path(tempfile.mkdtemp(prefix=f"gpubench-{args.model}-")) \
        if (tmp := os.environ.get("TMPDIR")) else Path(f"/tmp/gpubench-{args.model}")
    work.mkdir(parents=True, exist_ok=True)
    input_json = work / "input.json"
    seq = args.seq_file.read_text().strip()
    a3m_rows = args.msa_a3m.read_text().split("\n")
    # Same identical-bytes check the TT harness makes, from the other side.
    assert a3m_rows[1] == seq, f"{args.msa_a3m} query row does not match {args.seq_file}"
    n_msa_expected = args.msa_a3m.read_text().count(">")
    print(f"target {args.label}: {len(seq)} aa, {n_msa_expected} MSA rows", file=sys.stderr)
    write_input_json(input_json, args.msa_a3m.resolve(), seq, args.name)

    ladder = LADDERS[args.model]
    selected = set(args.rungs.split(",")) if args.rungs else None
    rungs = [r for r in ladder if selected is None or r["name"] in selected]
    if not rungs:
        ap.error("no rungs selected")
    run_model(args.model, model_name, args.repeat, str(input_json), work / "dump",
              args.checkpoint, args.out, rungs, args.label, n_msa_expected,
              save_structure=args.save_structure)
    return 0


if __name__ == "__main__":
    import tempfile
    sys.exit(main())
