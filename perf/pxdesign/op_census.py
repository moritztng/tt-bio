"""Per-op census of one warm PXDesign denoise step: every ttnn call synced and timed, grouped by stage and signature.

    python perf/pxdesign/op_census.py OUT.json [--levers normal] [--bf16] [--fixture laczc_512_nomsa] [--steps 3]

One cold design and one warm design run untouched (programs compiled, caches filled). Then a design runs with
`--steps` sampler steps wrapped: each ttnn call is bracketed by synchronize_device, so its time is device time plus
one dispatch plus one sync round trip; the empty round trip is measured and reported as `sync_s` so it can be
taken off. Calls nested inside a wrapped call are not counted twice. Stage is the innermost of atom_encoder /
token_dit / atom_decoder that is running, else `glue`. The unwrapped step wall from the warm design is beside it,
so `wall - sum(op device time)` is the host + dispatch share a trace could take back.
"""
import argparse, collections, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--levers", default="normal")
ap.add_argument("--bf16", action="store_true")
ap.add_argument("--fixture", default="laczc_512_nomsa")
ap.add_argument("--steps", type=int, default=3)
a = ap.parse_args()
if a.bf16:
    os.environ["PROTENIX_DIFFUSION_FP32_DEVICE"] = "0"
os.environ["TT_BIO_LEVERS"] = a.levers
TREE = Path.cwd()
sys.path.insert(0, str(TREE))

import torch, ttnn, yaml  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor, ensure_pxdesign_weights  # noqa: E402
from tt_bio.pxdesign.inputs import design_inputs_from_yaml  # noqa: E402
from tt_bio.pxdesign.model import ProtenixDesign  # noqa: E402

torch.set_grad_enabled(False)
spec = yaml.safe_load((TREE / f"perf/pxdesign/targets/{a.fixture}.yaml").read_text())
spec["target"]["file"] = str(TREE / "perf/pxdesign/targets" / Path(spec["target"]["file"]).name)
y = a.out.with_suffix(".yaml")
a.out.parent.mkdir(parents=True, exist_ok=True)
y.write_text(yaml.safe_dump(spec))
ensure_p300_mesh_descriptor()
model = ProtenixDesign.load_from_checkpoint(str(ensure_pxdesign_weights(Path.home() / ".boltz")))
D, dev = model.diffusion, model.dev

state = {"on": False, "depth": 0, "stage": [], "steps_left": 0}
ops = collections.defaultdict(lambda: [0, 0.0, 0.0])   # (stage, op, sig) -> [calls, synced s, flops]


def desc(v):
    if isinstance(v, ttnn.Tensor):
        try:
            buf = "L1" if v.memory_config().buffer_type == ttnn.BufferType.L1 else "D"
        except Exception:
            buf = "?"
        return f"{list(int(d) for d in v.shape)}{str(v.dtype).split('.')[-1][:4]}{buf}"
    if isinstance(v, (list, tuple)):
        return "(" + ",".join(desc(e) for e in v if isinstance(e, (ttnn.Tensor, list, tuple))) + ")"
    return ""


def flops(name, args, out):
    if name not in ("matmul", "linear") or not isinstance(out, ttnn.Tensor):
        return 0.0
    n = 1
    for d in out.shape:
        n *= int(d)
    return 2.0 * n * int(args[0].shape[-1])


def wrap(name, fn):
    def run(*x, **k):
        if not state["on"] or state["depth"]:
            return fn(*x, **k)
        state["depth"] += 1
        try:
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            r = fn(*x, **k)
            ttnn.synchronize_device(dev)
            dt = time.perf_counter() - t
        finally:
            state["depth"] -= 1
        key = (state["stage"][-1] if state["stage"] else "glue", name,
               ",".join(s for s in (desc(v) for v in x) if s))
        e = ops[key]
        e[0] += 1; e[1] += dt; e[2] += flops(name, x, r)
        return r
    return run


for name in dir(ttnn):
    f = getattr(ttnn, name)
    if (name.startswith("_") or not callable(f) or isinstance(f, type) or type(f).__name__ == "module"
            or name in ("synchronize_device", "to_torch", "from_torch", "deallocate")):
        continue
    try:
        setattr(ttnn, name, wrap(name, f))
    except Exception:
        pass


def staged(name, fn):
    def run(*x, **k):
        state["stage"].append(name)
        try:
            return fn(*x, **k)
        finally:
            state["stage"].pop()
    return run


class Staged:
    def __init__(self, name, m):
        self._m, self._call = m, staged(name, m)

    def __call__(self, *x, **k):
        return self._call(*x, **k)

    def __getattr__(self, k):
        return getattr(self._m, k)


D.atxE = Staged("atom_encoder", D.atxE)
D.atxD = Staged("atom_decoder", D.atxD)
D._token_dit_device = staged("token_dit", D._token_dit_device)
_denoise = D.denoise
step_walls = []


def denoise(*x, **k):
    if state["steps_left"] > 0:
        state["on"] = True
        state["steps_left"] -= 1
    t = time.perf_counter()
    r = _denoise(*x, **k)
    step_walls.append(time.perf_counter() - t)
    state["on"] = False
    return r


D.denoise = denoise


def design():
    feats = design_inputs_from_yaml(y)
    feats = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float64 else v) for k, v in feats.items()}
    t = time.perf_counter()
    model.design(feats, n_step=400, n_sample=1, seed=0)
    return time.perf_counter() - t


design()
step_walls.clear()
warm = design()
warm_step = sorted(step_walls)[len(step_walls) // 2]
ttnn.synchronize_device(dev)
t = time.perf_counter()
for _ in range(50):
    ttnn.synchronize_device(dev)
sync_s = (time.perf_counter() - t) / 50
state["steps_left"] = a.steps
step_walls.clear()
design()
S = a.steps
rows = [{"stage": s, "op": o, "sig": g, "calls": c / S, "synced_ms": 1e3 * t / S,
         "dev_ms": 1e3 * max(0.0, t - c * sync_s) / S, "gflop": f / S / 1e9}
        for (s, o, g), (c, t, f) in ops.items()]
rows.sort(key=lambda r: -r["synced_ms"])
by_stage = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
for r in rows:
    b = by_stage[r["stage"]]
    b[0] += r["calls"]; b[1] += r["dev_ms"]; b[2] += r["gflop"]
res = {"levers": sorted(model._levers), "bf16": a.bf16, "arch": str(dev.arch()), "warm_design_s": round(warm, 3),
       "warm_step_wall_ms": round(1e3 * warm_step, 3), "sync_s": sync_s,
       "ops_per_step": round(sum(r["calls"] for r in rows), 1),
       "dev_ms_per_step": round(sum(r["dev_ms"] for r in rows), 3),
       "gflop_per_step": round(sum(r["gflop"] for r in rows), 2),
       "by_stage": {k: {"ops": round(v[0], 1), "dev_ms": round(v[1], 3), "gflop": round(v[2], 2)}
                    for k, v in by_stage.items()},
       "top": rows[:60]}
a.out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "top"}))
T.cleanup()
