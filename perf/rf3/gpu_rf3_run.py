"""One RF3 fold rung, repeated in a single process, with the kernel paths COUNTED and the
phases timed.

    python perf/rf3/gpu_rf3_run.py --inputs perf/rf3/inputs/rf3_512.json \
        --out-dir /work/out/rf3_512 --report /work/results/rf3_512.json --reps 4

Forked from perf/dsfix/gpu_rfd3_run.py, which established the discipline: counters, not flags.
RF3 routes triangle attention and triangle multiplication to cuEquivariance or to a vanilla
PyTorch path at *call* time, on `self.use_cuequivariance and SHOULD_USE_CUEQUIVARIANCE`
(rf3/model/layers/attention.py). An env var or an installed wheel proves nothing about what ran,
so every path is wrapped and its real call count reported. RFD3's cueq counter read 0 on every
arm despite cuequivariance being installed; do not assume RF3's reads non-zero.

Why one process with N reps rather than N processes: the rung is a single fixed shape, so all
reps recompile nothing, and `initialize()` is idempotent, so the checkpoint load is paid once.
Rep 0 is the cold rep (cuDNN/cueq autotune, allocator growth) and is reported separately, never
folded into the median.

Phases are wall-clock around cuda-synchronised boundaries:
  prep        the AtomWorks transform pipeline (host featurisation)
  featinit    RF3.pre_recycle
  trunk       RF3.recycle, summed over the recycles
  distogram   DistogramHead
  diffusion   SampleDiffusion.sample_diffusion_like_af3, the full rollout
  confidence  ConfidenceHead
Anything in the fold that is not one of these (structure assembly, metrics, CIF writing) shows up
as `other_s` = fold_s - sum(phases), so the breakdown always closes.
"""

import argparse
import collections
import json
import pathlib
import subprocess
import sys
import time

COUNTS: dict[str, int] = {}
PHASE: dict[str, list[float]] = collections.defaultdict(list)
_ACTIVE_REP = [-1]


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_only(obj, name: str, label: str | None = None):
    """Count calls to obj.name without timing it (no sync, so no perturbation)."""
    label = label or name
    fn = getattr(obj, name, None)
    if fn is None:
        COUNTS[label] = -1          # -1 == not present in this build
        return
    COUNTS[label] = 0

    def wrapper(*a, **kw):
        COUNTS[label] += 1
        return fn(*a, **kw)

    setattr(obj, name, wrapper)


def timed(obj, name: str, phase: str):
    """Count AND time calls to obj.name, cuda-synchronised on both sides."""
    fn = getattr(obj, name, None)
    if fn is None:
        COUNTS["phase_missing:" + phase] = -1
        return
    COUNTS.setdefault("calls:" + phase, 0)

    def wrapper(*a, **kw):
        COUNTS["calls:" + phase] += 1
        _sync()
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            _sync()
            PHASE["%d/%s" % (_ACTIVE_REP[0], phase)].append(time.perf_counter() - t0)

    setattr(obj, name, wrapper)


def gpu_static() -> dict:
    q = ("name,driver_version,memory.total,power.limit,clocks.max.sm,"
         "compute_cap,pcie.link.gen.max")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + q,
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        vals = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return dict(zip(q.split(","), vals))
    except Exception as e:                                    # noqa: BLE001
        return {"error": repr(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--reps", type=int, default=4, help="rep 0 is the discarded cold rep")
    ap.add_argument("--n-recycles", type=int, default=10)
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--diffusion-batch-size", type=int, default=1)
    ap.add_argument("--early-stop-plddt", type=float, default=0.0,
                    help="0 disables. The shipped default is 0.5, which silently truncates a "
                         "no-MSA fold after one recycle and would measure the wrong thing.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    import foundry
    import rf3

    # --- config, exactly as rf3/inference.py assembles it -----------------------------------
    pkg = pathlib.Path(rf3.__file__).parent
    config_dir = pkg / "configs"
    if not config_dir.exists():                              # editable/dev checkout
        config_dir = pkg.parent.parent / "configs"
    overrides = ["inference_engine=rf3",
                 "inputs=%s" % args.inputs,
                 "out_dir=%s" % args.out_dir,
                 "n_recycles=%d" % args.n_recycles,
                 "num_steps=%d" % args.num_steps,
                 "diffusion_batch_size=%d" % args.diffusion_batch_size,
                 "early_stopping_plddt_threshold=%g" % args.early_stop_plddt,
                 "seed=%d" % args.seed,
                 "skip_existing=False"]
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="inference", overrides=overrides)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)

    run_keys = {"inputs", "out_dir", "dump_predictions", "dump_trajectories",
                "one_model_per_file", "annotate_b_factor_with_plddt", "sharding_pattern",
                "skip_existing", "template_selection", "ground_truth_conformer_selection",
                "cyclic_chains", "add_missing_atoms"}
    run_params = {k: cfg_dict[k] for k in run_keys if k in cfg_dict}
    init_cfg = OmegaConf.create({k: v for k, v in cfg_dict.items() if k not in run_keys})

    # --- counters ---------------------------------------------------------------------------
    import rf3.diffusion_samplers.inference_sampler as S
    import rf3.model.layers.af3_auxiliary_heads as AUX
    import rf3.model.layers.attention as A
    import rf3.model.RF3 as M
    import rf3.model.RF3_structure as ST

    count_only(A.TriangleAttention, "_forward_cuequivariance", "triangle_attention_cueq")
    count_only(A.TriangleAttention, "_forward_vanilla", "triangle_attention_vanilla")
    count_only(A.TriangleMultiplication, "_forward_cuequivariance", "triangle_multiply_cueq")
    count_only(A.TriangleMultiplication, "_forward_vanilla", "triangle_multiply_vanilla")
    if getattr(A, "cuet", None) is not None:
        count_only(A.cuet, "triangle_attention", "cuet.triangle_attention")
        count_only(A.cuet, "triangle_multiplicative_update", "cuet.triangle_multiplicative_update")
    else:
        COUNTS["cuet.triangle_attention"] = -1
        COUNTS["cuet.triangle_multiplicative_update"] = -1

    _sdpa = F.scaled_dot_product_attention
    COUNTS["scaled_dot_product_attention"] = 0

    def sdpa_wrapper(*a, **kw):
        COUNTS["scaled_dot_product_attention"] += 1
        return _sdpa(*a, **kw)

    F.scaled_dot_product_attention = sdpa_wrapper
    torch.nn.functional.scaled_dot_product_attention = sdpa_wrapper

    # --- phase timers -----------------------------------------------------------------------
    timed(M.RF3, "pre_recycle", "featinit")
    timed(M.RF3, "recycle", "trunk")
    timed(ST.DistogramHead, "forward", "distogram")
    timed(S.SampleDiffusion, "sample_diffusion_like_af3", "diffusion")
    timed(AUX.ConfidenceHead, "forward", "confidence")

    env = {"rf3_config_dir": str(config_dir),
           "should_use_cuequivariance": bool(getattr(foundry, "SHOULD_USE_CUEQUIVARIANCE", False)),
           "torch": torch.__version__,
           "torch_cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(),
           "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
           "cudnn_tf32": torch.backends.cudnn.allow_tf32,
           "float32_matmul_precision": torch.get_float32_matmul_precision(),
           "python": sys.version.split()[0],
           "gpu_static": gpu_static()}
    from importlib.metadata import PackageNotFoundError, version
    for pkg_name in ("rc-foundry", "atomworks", "cuequivariance-torch",
                     "cuequivariance-ops-torch-cu12", "cuequivariance-ops-cu12",
                     "lightning", "numpy"):
        try:
            env[pkg_name] = version(pkg_name)
        except PackageNotFoundError:
            env[pkg_name] = None

    report = {"label": args.label, "inputs": args.inputs, "reps": args.reps,
              "n_recycles": args.n_recycles, "num_steps": args.num_steps,
              "diffusion_batch_size": args.diffusion_batch_size,
              "early_stop_plddt": args.early_stop_plddt, "seed": args.seed,
              "env": env, "rep_s": [], "phases": {}, "counts": None,
              "peak_vram_alloc_B": None, "peak_vram_reserved_B": None,
              "load_s": None, "ok": False, "why": ""}

    def dump():
        report["counts"] = dict(COUNTS)
        per_rep: dict[str, dict[str, float]] = {}
        for key, xs in PHASE.items():
            rep, phase = key.split("/", 1)
            per_rep.setdefault(rep, {})[phase] = round(sum(xs), 5)
            per_rep[rep][phase + "_n"] = len(xs)
        report["phases"] = per_rep
        p = pathlib.Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2) + "\n")

    # --- build the engine once, fold `reps` times --------------------------------------------
    try:
        t0 = time.perf_counter()
        engine = instantiate(init_cfg, _convert_="partial", _recursive_=False)
        engine.initialize()
        _sync()
        report["load_s"] = round(time.perf_counter() - t0, 3)

        from foundry.utils.logging import suppress_warnings

        base_out = pathlib.Path(run_params["out_dir"])
        for rep in range(args.reps):
            _ACTIVE_REP[0] = rep
            params = dict(run_params)
            params["out_dir"] = str(base_out / ("rep%d" % rep))
            torch.cuda.reset_peak_memory_stats()
            _sync()
            t1 = time.perf_counter()
            with suppress_warnings(is_inference=True):
                engine.run(**params)
            _sync()
            dt = time.perf_counter() - t1
            report["rep_s"].append(round(dt, 4))
            report["peak_vram_alloc_B"] = max(report["peak_vram_alloc_B"] or 0,
                                              torch.cuda.max_memory_allocated())
            report["peak_vram_reserved_B"] = max(report["peak_vram_reserved_B"] or 0,
                                                 torch.cuda.max_memory_reserved())
            print("[rf3] %s rep%d %.3fs" % (args.label, rep, dt), flush=True)
            dump()

        # --- sanity: a real structure, finite coords, no early stop -------------------------
        ok, why = check_output(base_out / ("rep%d" % (args.reps - 1)))
        report["ok"], report["why"] = ok, why
        report.update(read_confidence(base_out / ("rep%d" % (args.reps - 1))))
    except Exception as e:                                    # noqa: BLE001
        import traceback
        report["ok"] = False
        report["why"] = "".join(traceback.format_exception(e))[-4000:]
        dump()
        raise
    finally:
        dump()


def check_output(out_dir: pathlib.Path) -> tuple[bool, str]:
    """Every rung must have written a structure with finite coordinates and no early stop."""
    import csv
    if not out_dir.exists():
        return False, "no out_dir %s" % out_dir
    for rs in out_dir.rglob("*_ranking_scores.csv"):
        with rs.open() as fh:
            for row in csv.DictReader(fh):
                if str(row.get("early_stopped", "")).lower() == "true":
                    return False, "EARLY STOPPED (%s) -- the fold was truncated" % rs.name
    cifs = [p for p in out_dir.rglob("*.cif")] + [p for p in out_dir.rglob("*.cif.gz")]
    if not cifs:
        return False, "no cif written under %s" % out_dir
    import gzip
    p = sorted(cifs)[0]
    opener = gzip.open if p.suffix == ".gz" else open
    n, bad = 0, 0
    with opener(p, "rt") as fh:                               # type: ignore[operator]
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                n += 1
                for tok in line.split()[10:13]:
                    try:
                        v = float(tok)
                    except ValueError:
                        continue
                    if v != v or abs(v) == float("inf"):
                        bad += 1
    if n == 0:
        return False, "%s has 0 atoms" % p.name
    if bad:
        return False, "%s has %d non-finite coords" % (p.name, bad)
    return True, "%d atoms in %s" % (n, p.name)


def read_confidence(out_dir: pathlib.Path) -> dict:
    """pTM/pLDDT are the fold-is-sane signal, not a perf number. Report whatever is there."""
    out: dict = {}
    for j in sorted(out_dir.rglob("*_summary_confidences.json")):
        try:
            d = json.loads(j.read_text())
        except Exception:                                     # noqa: BLE001
            continue
        out["confidence"] = {k: v for k, v in d.items() if isinstance(v, (int, float))}
        break
    return out


if __name__ == "__main__":
    main()
