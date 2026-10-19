"""BC4 deliverable B on the box: where a B200's BindCraft 2 design iteration goes.

Runs the frozen workload (PD-L1, binder 80, seed 101) through bc2_gpu.py's own hook, takes a bounded JAX profiler
trace inside the first WARM trajectory, and measures the box's own roofs (HBM copy bandwidth and bf16 matmul) in the
same process, so the achieved fraction is against this card, not a datasheet.

    python bc4_census.py --bc2 /root/bc2 --out DIR [--trace-after 10] [--trace-s 6]

Writes DIR/census.json (roofs, trajectory timings), DIR/trace/ (the profiler trace, parsed on pc by
bc4_census_parse.py) and DIR/runs.jsonl (the same rep records as every other arm).
"""
import argparse, json, os, sys, threading, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--bc2", required=True)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--trace-after", type=float, default=10, help="seconds into the warm trajectory before tracing")
ap.add_argument("--trace-s", type=float, default=6, help="length of the trace window")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, a.bc2)

import jax, jax.numpy as jnp  # noqa: E402
import bindcraft.campaign as campaign  # noqa: E402
from bindcraft.model_weights import model_weights  # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings  # noqa: E402
from bc2_traj import cpu_model, hook, s_per_iter  # noqa: E402

LOG = open(a.out / "runs.jsonl", "a")
census = dict(host=os.uname().nodename, cpu=cpu_model(), ncpu=os.cpu_count(), jax=jax.__version__,
              trace_dir=str(a.out / "trace"))


def log(**kw):
    LOG.write(json.dumps(dict(t=time.time(), **kw), default=str) + "\n"); LOG.flush()


def roofs():
    """This card's measured roofs: HBM read+write bandwidth (a copy over 8 GiB) and bf16 matmul throughput."""
    out = {}
    n = 1 << 30  # 1 Gi float32 = 4 GiB, copied = 8 GiB touched
    x = jnp.ones(n, jnp.float32).block_until_ready()
    cp = jax.jit(lambda v: v * 2.0)
    for _ in range(3):
        cp(x).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(10):
        y = cp(x)
    y.block_until_ready()
    dt = (time.perf_counter() - t0) / 10
    out["hbm_copy_gbs"] = round(2 * n * 4 / dt / 1e9, 1)
    del x, y
    m = 8192
    A = jnp.ones((m, m), jnp.bfloat16)
    mm = jax.jit(lambda u, v: u @ v)
    for _ in range(3):
        mm(A, A).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(10):
        C = mm(A, A)
    C.block_until_ready()
    dt = (time.perf_counter() - t0) / 10
    out["bf16_matmul_tflops"] = round(2 * m ** 3 / dt / 1e12, 1)
    del A, C
    return out


def trace_window():
    """Trace a bounded window inside the warm trajectory, once it is already in steady state."""
    time.sleep(a.trace_after)
    if len(traj) < 2 or traj[1].get("design_s"):
        return  # the warm trajectory never started, or finished before the window opened
    try:
        jax.profiler.start_trace(str(a.out / "trace"))
        time.sleep(a.trace_s)
        jax.profiler.stop_trace()
        census["traced"] = [time.time() - a.trace_s, time.time()]
    except Exception:
        census["trace_err"] = traceback.format_exc()[-1000:]


project = a.out / f"campaign_s{a.seed}"
settings = cleaned_campaign_settings(read_settings(str(Path(a.bc2) / "examples" / "pdl1.json"), parse_setting_overrides(
    [f"campaign_seed={a.seed}", "max_trajectories=2", f"project_folder={project}", "auto_multi_gpu=false",
     "binder_lengths=[80]"])))

census["roofs"] = roofs()
log(ev="start", kind="census", roofs=census["roofs"], jax=jax.__version__, cpu=census["cpu"], ncpu=census["ncpu"],
    xla_flags=os.environ.get("XLA_FLAGS"), devices=[str(d) for d in jax.devices()])


def emit(i):
    tr = traj[i]
    log(ev="rep", trajectory=tr["n"], kind="cold" if i == 0 else "warm", design_s=tr.get("design_s"),
        iterations=tr.get("iterations"), s_per_iter=s_per_iter(tr), validate_s=tr.get("validate_s"),
        loadavg=tr.get("loadavg"), err=tr.get("err"))
    if i == 0:  # the warm trajectory starts in the next instant
        threading.Thread(target=trace_window, daemon=True).start()


traj = hook(campaign, project, emit)
af2, mpnn = model_weights()
try:
    campaign.run_campaign(settings, str(project), af2_weights=af2, mpnn_weights=mpnn)
except Exception:
    census["err"] = traceback.format_exc()[-3000:]
if traj:
    emit(len(traj) - 1)
census["trajectories"] = [{k: v for k, v in t.items() if k not in ("t0", "t0w", "rows0", "la0")} for t in traj]
(a.out / "census.json").write_text(json.dumps(census, indent=1, default=str) + "\n")
log(ev="end")
