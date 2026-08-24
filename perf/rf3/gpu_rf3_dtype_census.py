"""What precision does RF3 actually execute in, read out of a running fold.

    python perf/rf3/gpu_rf3_dtype_census.py --inputs perf/rf3/inputs/rf3_512.json \
        --out-dir /work/out/census --report /work/results/census.json

This exists because reading the source got the answer wrong twice. `models/rf3/configs/trainer/
rf3.yaml` says `precision: bf16-mixed`; the perf page's B200 note says "RF3 ships fp32"; both are
claims about a config, and neither is a tensor. So: forward-hook every module, wrap the functional
ops that no module owns, and record the dtypes that actually arrive and leave, plus the autocast and
matmul-precision state read from INSIDE the forward rather than from the config.

Nothing here sets a precision. `torch.set_float32_matmul_precision`, `torch.autocast` and any dtype
cast are deliberately absent: the whole point is what the shipped defaults do. The one deviation
from stock inference is `early_stopping_plddt_threshold=0`, inherited from `gpu_rf3_run.py`, because
the shipped 0.5 truncates a no-MSA fold after one recycle and would census a tenth of the network.

Instrumentation, and what each part is for:

  module hooks   `register_module_forward_hook` is global, so every nn.Module in the tree is
                 covered without walking it. The key carries the module's qualified name, its
                 class, its weight dtype, the input and output dtype signatures, and whether
                 autocast was enabled at that moment. Weight dtype is load-bearing: RF3's cueq path
                 asserts bf16 inputs, and bf16 weights with autocast OFF would satisfy that assert
                 just as well as fp32 weights with autocast ON. Only the weight dtype separates the
                 two.
  fn wrappers    softmax, einsum, matmul, layer_norm and sigmoid are called as functions, so no
                 module hook sees them. Under autocast, softmax and layer_norm are on torch's
                 float32 list and matmul/linear are on the lower-precision list, so these four are
                 exactly where a mixed forward shows its seams.
  cueq wrappers  `cuet.triangle_attention` and `cuet.triangle_multiplicative_update` are custom
                 CUDA extensions and are NOT on autocast's registry, so they receive whatever dtype
                 reaches them. Their operands are the answer to the question this task asks.
  pre-cast       `TriangleAttention._forward_cuequivariance` is wrapped separately from
                 `cuet.triangle_attention` because RF3 casts to the autocast dtype between them.
                 Wrapping only the cueq call would show bf16 and hide who produced fp32.

Counters are kept identical in name to `gpu_rf3_run.py` so a census is cross-checkable against the
published GPU cells: `triangle_attention_cueq`, `triangle_attention_vanilla`,
`triangle_multiply_cueq`, `triangle_multiply_vanilla`, `scaled_dot_product_attention`.
"""

import argparse
import collections
import json
import pathlib
import subprocess
import sys
import time

CENSUS: collections.Counter = collections.Counter()
FN: collections.Counter = collections.Counter()
COUNTS: dict = {}
CTX_SNAPSHOTS: dict = {}
FLOPS: collections.Counter = collections.Counter()
NAMES: dict = {}
_MAX_KEYS = 40000


def _ac() -> int:
    import torch
    try:
        return int(torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"))
    except TypeError:
        return int(torch.is_autocast_enabled())


def dts(x, depth: int = 0) -> str:
    """Compact dtype signature of an arbitrary nested argument structure."""
    import torch
    if isinstance(x, torch.Tensor):
        return str(x.dtype).replace("torch.", "")
    if isinstance(x, (list, tuple)):
        if depth > 2 or len(x) > 12:
            return "..."
        return "(" + ",".join(dts(v, depth + 1) for v in x) + ")"
    if isinstance(x, dict):
        if depth > 2 or len(x) > 20:
            return "..."
        return "{" + ",".join("%s=%s" % (k, dts(v, depth + 1)) for k, v in sorted(x.items())) + "}"
    if isinstance(x, (int, float, bool, str, type(None))):
        return "-"
    return type(x).__name__


def probe() -> dict:
    """Precision state as the running process sees it, not as the config declares it."""
    import torch
    d: dict = {}
    for dev in ("cuda", "cpu"):
        try:
            d["autocast_enabled_" + dev] = bool(torch.is_autocast_enabled(dev))
        except TypeError:
            d["autocast_enabled_" + dev] = bool(torch.is_autocast_enabled()) if dev == "cuda" else None
        try:
            d["autocast_dtype_" + dev] = str(torch.get_autocast_dtype(dev))
        except Exception as e:                                    # noqa: BLE001
            d["autocast_dtype_" + dev] = "err:%r" % e
    for name, get in (
        ("autocast_cache_enabled", lambda: torch.is_autocast_cache_enabled()),
        ("float32_matmul_precision", lambda: torch.get_float32_matmul_precision()),
        ("cuda_matmul_allow_tf32", lambda: torch.backends.cuda.matmul.allow_tf32),
        ("cudnn_allow_tf32", lambda: torch.backends.cudnn.allow_tf32),
        ("cuda_matmul_fp32_precision", lambda: torch.backends.cuda.matmul.fp32_precision),
        ("cudnn_conv_fp32_precision", lambda: torch.backends.cudnn.conv.fp32_precision),
        ("allow_bf16_reduced_precision_reduction",
         lambda: torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        ("allow_fp16_reduced_precision_reduction",
         lambda: torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
        ("grad_enabled", lambda: torch.is_grad_enabled()),
        ("inference_mode", lambda: torch.is_inference_mode_enabled()),
        ("default_dtype", lambda: str(torch.get_default_dtype())),
    ):
        try:
            v = get()
        except Exception as e:                                    # noqa: BLE001
            v = "err:%s" % type(e).__name__
        d[name] = v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
    return d


def snap(label: str) -> None:
    if label not in CTX_SNAPSHOTS:
        CTX_SNAPSHOTS[label] = probe()


# --------------------------------------------------------------------------------------------
# module census
# --------------------------------------------------------------------------------------------
def index_modules(root, prefix: str = "") -> int:
    n = 0
    for name, m in root.named_modules(prefix=prefix):
        NAMES.setdefault(id(m), name or "<root>")
        n += 1
    return n


def lazy_index(mod, args) -> None:
    """Name a module subtree the first time anything in it is called.

    A forward PRE-hook fires top-down, so the outermost module reached gets indexed first and
    everything under it inherits a real qualified name. Finding the root by attribute lookup on the
    engine is not reliable across foundry versions; this does not need to.
    """
    if id(mod) not in NAMES:
        index_modules(mod, prefix=type(mod).__name__)


def _wdtype(mod) -> str:
    import torch
    w = getattr(mod, "weight", None)
    return str(w.dtype).replace("torch.", "") if isinstance(w, torch.Tensor) else "-"


def module_hook(mod, args, out) -> None:
    import torch
    cls = type(mod).__name__
    key = (NAMES.get(id(mod), "?" + cls), cls, _wdtype(mod), dts(args), dts(out), _ac())
    if key in CENSUS or len(CENSUS) < _MAX_KEYS:
        CENSUS[key] += 1
    if cls == "Linear" and isinstance(out, torch.Tensor) and args and isinstance(args[0], torch.Tensor):
        # 2*N*K*M, the only weighting that is free to compute at hook time.
        FLOPS[str(out.dtype).replace("torch.", "")] += 2 * args[0].numel() * out.shape[-1]


# --------------------------------------------------------------------------------------------
# functional wrappers
# --------------------------------------------------------------------------------------------
def wrap_fn(owner, name: str, label: str) -> None:
    fn = getattr(owner, name, None)
    if fn is None:
        COUNTS[label] = -1
        return
    COUNTS[label] = 0

    def w(*a, **kw):
        COUNTS[label] += 1
        r = fn(*a, **kw)
        key = (label, dts(a), dts(kw), dts(r), _ac())
        if key in FN or len(FN) < _MAX_KEYS:
            FN[key] += 1
        snap(label)
        return r

    setattr(owner, name, w)


def count_only(obj, name: str, label: str) -> None:
    """Name-compatible with gpu_rf3_run.py so the counters cross-check against published cells."""
    fn = getattr(obj, name, None)
    if fn is None:
        COUNTS[label] = -1
        return
    COUNTS[label] = 0

    def w(*a, **kw):
        COUNTS[label] += 1
        r = fn(*a, **kw)
        key = (label, dts(a[1:] if a and hasattr(a[0], "_parameters") else a), dts(kw), dts(r), _ac())
        if key in FN or len(FN) < _MAX_KEYS:
            FN[key] += 1
        snap(label)
        return r

    setattr(obj, name, w)


def install(torch, F) -> None:
    """Every wrapper, in one place, so the report can say exactly what was instrumented."""
    wrap_fn(F, "softmax", "F.softmax")
    wrap_fn(torch, "softmax", "torch.softmax")
    wrap_fn(F, "layer_norm", "F.layer_norm")
    wrap_fn(torch, "einsum", "torch.einsum")
    wrap_fn(torch, "matmul", "torch.matmul")
    wrap_fn(torch, "bmm", "torch.bmm")
    wrap_fn(torch, "sigmoid", "torch.sigmoid")
    wrap_fn(F, "scaled_dot_product_attention", "scaled_dot_product_attention")
    torch.nn.functional.scaled_dot_product_attention = F.scaled_dot_product_attention


def instrument_rf3(torch, F) -> None:
    import rf3.model.layers.attention as A

    # Pre-cast: the args as they ARRIVE, before RF3's own `.to(autocast_dtype)`.
    count_only(A.TriangleAttention, "_forward_cuequivariance", "triangle_attention_cueq")
    count_only(A.TriangleAttention, "_forward_vanilla", "triangle_attention_vanilla")
    count_only(A.TriangleMultiplication, "_forward_cuequivariance", "triangle_multiply_cueq")
    count_only(A.TriangleMultiplication, "_forward_vanilla", "triangle_multiply_vanilla")

    # Post-cast: what the custom CUDA extension actually receives. Autocast has no registry
    # entry for these, so nothing casts them implicitly.
    cuet = getattr(A, "cuet", None)
    if cuet is not None:
        wrap_fn(cuet, "triangle_attention", "cuet.triangle_attention")
        wrap_fn(cuet, "triangle_multiplicative_update", "cuet.triangle_multiplicative_update")
    else:
        COUNTS["cuet.triangle_attention"] = -1
        COUNTS["cuet.triangle_multiplicative_update"] = -1

    # opt_einsum is bound at import time as a module global, so patching the package is not enough.
    wrap_fn(A, "einsum", "attention.einsum")


# --------------------------------------------------------------------------------------------
# card hygiene, verbatim from gpu_rf3_run.py
# --------------------------------------------------------------------------------------------
def compute_apps() -> list:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        apps = []
        for line in out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit():
                apps.append({"pid": int(parts[0]), "used_MiB": int(parts[1])})
        return apps
    except Exception as e:                                        # noqa: BLE001
        return [{"error": repr(e)}]


def gpu_static() -> dict:
    q = "name,driver_version,memory.total,power.limit,power.draw,compute_cap"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        vals = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return dict(zip(q.split(","), vals))
    except Exception as e:                                        # noqa: BLE001
        return {"error": repr(e)}


# --------------------------------------------------------------------------------------------
def serialise() -> dict:
    census = [{"site": k[0], "cls": k[1], "wdtype": k[2], "in": k[3], "out": k[4],
               "autocast": bool(k[5]), "n": n} for k, n in CENSUS.most_common()]
    fn = [{"op": k[0], "args": k[1], "kwargs": k[2], "out": k[3], "autocast": bool(k[4]), "n": n}
          for k, n in FN.most_common()]
    by_out: collections.Counter = collections.Counter()
    for row in census:
        by_out[row["out"]] += row["n"]
    return {"census": census, "fn": fn, "counts": dict(COUNTS),
            "counter_caveat": (
                "torch.matmul / torch.bmm / torch.einsum / torch.softmax are counters on the "
                "PYTHON symbol. nn.Linear reaches ATen through F.linear and `@` reaches it through "
                "__matmul__, so neither passes the wrapped symbol: a 0 here means nothing called "
                "that name, NOT that no matrix multiply ran. The nn.Linear rows in `census` are the "
                "matmul evidence; these counters exist to catch the explicit call sites."),
            "ctx_snapshots": CTX_SNAPSHOTS,
            "calls_by_output_dtype": dict(by_out.most_common()),
            "linear_flops_by_dtype": dict(FLOPS.most_common())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--n-recycles", type=int, default=10)
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--diffusion-batch-size", type=int, default=1)
    ap.add_argument("--early-stop-plddt", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="census")
    ap.add_argument("--no-cueq", action="store_true",
                    help="SEPARATE LABELLED ARM. Forces the vanilla triangle path so the census "
                         "can read the dtypes of the PyTorch implementation RF3 does not ship. "
                         "Never the primary arm.")
    ap.add_argument("--profile", action="store_true",
                    help="SEPARATE LABELLED ARM. torch.profiler over the fold, kernel names kept, "
                         "so 'which cuEquivariance kernel ran' is answered by the kernel's own "
                         "name instead of by a wheel A/B.")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    import foundry
    import rf3

    pkg = pathlib.Path(rf3.__file__).parent
    config_dir = pkg / "configs"
    if not config_dir.exists():
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

    install(torch, F)
    instrument_rf3(torch, F)

    env = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(),
           "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "should_use_cuequivariance": bool(getattr(foundry, "SHOULD_USE_CUEQUIVARIANCE", False)),
           "python": sys.version.split()[0],
           "gpu_static": gpu_static(),
           "compute_apps_before": compute_apps(),
           "precision_at_import": probe()}
    from importlib.metadata import PackageNotFoundError, version
    for p in ("rc-foundry", "atomworks", "cuequivariance-torch", "cuequivariance-ops-torch-cu12",
              "cuequivariance-ops-torch-cu13", "cuequivariance-ops-cu12", "cuequivariance-ops-cu13",
              "lightning", "triton", "numpy"):
        try:
            env[p] = version(p)
        except PackageNotFoundError:
            env[p] = None

    report = {"label": args.label, "inputs": args.inputs, "arm": "shipped-defaults",
              "n_recycles": args.n_recycles, "num_steps": args.num_steps,
              "diffusion_batch_size": args.diffusion_batch_size, "seed": args.seed,
              "early_stop_plddt": args.early_stop_plddt,
              "no_cueq_arm": args.no_cueq, "profile_arm": args.profile,
              "env": env, "fold_s": None, "n_modules": None,
              "compute_apps_after": None, "gpu_exclusive": None,
              "kernels": None, "ok": False, "why": ""}
    if args.no_cueq:
        report["arm"] = "CONTRAST-no-cueq"
    if args.profile:
        report["arm"] = "CONTRAST-profiler"

    def dump():
        report.update(serialise())
        p = pathlib.Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str) + "\n")

    try:
        engine = instantiate(init_cfg, _convert_="partial", _recursive_=False)
        engine.initialize()

        model = getattr(engine, "model", None) or getattr(engine, "module", None)
        roots = [m for m in (model, getattr(engine, "trainer", None)) if m is not None]
        n_mod = 0
        for r in roots:
            if hasattr(r, "named_modules"):
                n_mod += index_modules(r)
        report["n_modules"] = n_mod

        if args.no_cueq:
            import rf3.model.layers.attention as A
            n = 0
            for r in roots:
                if not hasattr(r, "modules"):
                    continue
                for m in r.modules():
                    if isinstance(m, (A.TriangleAttention, A.TriangleMultiplication)):
                        m.use_cuequivariance = False
                        n += 1
            report["no_cueq_modules_flipped"] = n

        torch.nn.modules.module.register_module_forward_pre_hook(lazy_index)
        if not args.profile:
            # Hooks on every module would put ~10^5 Python frames into the trace and bury the
            # kernel names the profiler arm exists to read.
            torch.nn.modules.module.register_module_forward_hook(module_hook)

        from foundry.utils.logging import suppress_warnings
        t0 = time.perf_counter()
        if args.profile:
            from torch.profiler import ProfilerActivity, profile
            acts = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                acts.append(ProfilerActivity.CUDA)
            with profile(activities=acts, record_shapes=False, with_stack=False) as prof:
                with suppress_warnings(is_inference=True):
                    engine.run(**run_params)
            rows = []
            for e in prof.key_averages():
                dev = str(getattr(e, "device_type", ""))
                if "CUDA" in dev or "DeviceType.CUDA" in dev or getattr(e, "self_device_time_total", 0):
                    rows.append({"name": e.key, "count": e.count,
                                 "self_device_us": round(getattr(e, "self_device_time_total", 0.0), 1)})
            rows.sort(key=lambda r: -r["self_device_us"])
            report["kernels"] = rows[:200]
        else:
            with suppress_warnings(is_inference=True):
                engine.run(**run_params)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        report["fold_s"] = round(time.perf_counter() - t0, 3)

        after = compute_apps()
        before = env["compute_apps_before"] or []
        report["compute_apps_after"] = after
        report["gpu_exclusive"] = (len([a for a in after if "pid" in a]) <= 1
                                   and len([a for a in before if "pid" in a]) <= 1)
        report["gpu_after_static"] = gpu_static()

        ok, why = check_output(pathlib.Path(run_params["out_dir"]))
        if ok and not report["gpu_exclusive"]:
            ok, why = False, ("GPU NOT EXCLUSIVE: before=%r after=%r -- a census taken beside a "
                              "co-tenant is still a valid dtype readout, but the run is flagged "
                              "because the same box condition voided an earlier RF3 campaign"
                              % (before, after))
        report["ok"], report["why"] = ok, why
    except Exception as e:                                        # noqa: BLE001
        import traceback
        report["ok"] = False
        report["why"] = "".join(traceback.format_exception(e))[-4000:]
        dump()
        raise
    finally:
        dump()

    summarise(report)


def check_output(out_dir: pathlib.Path):
    import csv
    if not out_dir.exists():
        return False, "no out_dir %s" % out_dir
    for rs in out_dir.rglob("*_ranking_scores.csv"):
        with rs.open() as fh:
            for row in csv.DictReader(fh):
                if str(row.get("early_stopped", "")).lower() == "true":
                    return False, "EARLY STOPPED (%s)" % rs.name
    cifs = list(out_dir.rglob("*.cif")) + list(out_dir.rglob("*.cif.gz"))
    if not cifs:
        return False, "no cif written under %s" % out_dir
    return True, "%d cif(s), first %s" % (len(cifs), sorted(cifs)[0].name)


TRI = ("TriangleAttention", "TriangleMultiplication")


def summarise(report: dict) -> None:
    print("\n================ RF3 dtype census: %s ================" % report["arm"])
    p = report["env"]["precision_at_import"]
    print("at import : autocast_cuda=%s dtype=%s  f32_matmul=%s tf32_matmul=%s tf32_cudnn=%s"
          % (p.get("autocast_enabled_cuda"), p.get("autocast_dtype_cuda"),
             p.get("float32_matmul_precision"), p.get("cuda_matmul_allow_tf32"),
             p.get("cudnn_allow_tf32")))
    for label, s in sorted(report.get("ctx_snapshots", {}).items()):
        print("inside %-38s autocast_cuda=%s dtype=%s f32_matmul=%s"
              % (label, s.get("autocast_enabled_cuda"), s.get("autocast_dtype_cuda"),
                 s.get("float32_matmul_precision")))
    print("\ncounters:", json.dumps(report.get("counts", {}), sort_keys=True))
    print("\ncalls by output dtype:", report.get("calls_by_output_dtype"))
    print("Linear FLOPs by dtype :", report.get("linear_flops_by_dtype"))
    print("\n-- the triangle chain -------------------------------------------------")
    for row in report.get("census", []):
        if any(t in row["site"] or t == row["cls"] for t in TRI) or "norm" in row["site"][-12:]:
            print("  %-58s %-24s w=%-9s in=%-28s out=%-10s ac=%s n=%d"
                  % (row["site"][-58:], row["cls"], row["wdtype"], row["in"][:28], row["out"],
                     row["autocast"], row["n"]))
    print("\n-- functional ops ------------------------------------------------------")
    for row in report.get("fn", [])[:40]:
        print("  %-42s args=%-34s out=%-10s ac=%s n=%d"
              % (row["op"], row["args"][:34], row["out"], row["autocast"], row["n"]))
    print("\nok=%s exclusive=%s fold_s=%s\nwhy=%s"
          % (report["ok"], report["gpu_exclusive"], report["fold_s"], report["why"]))


if __name__ == "__main__":
    main()
