#!/usr/bin/env python3
"""Where the wall of one warm ESMFold2 fold goes, on the GAP-table workload (cdk2x2_512, shipped steps, 1 sample).

Builds the fold through tt_baseline.build_fold (as ab.py), runs one cold fold, then:
  plain    one fold, nothing wrapped (the reference wall);
  stages   one fold with every stage below wrapped in synchronize_device on both sides, so each stage's
           wall is its own device time plus its own host dispatch, and the sum against the plain wall
           says how much of the fold is in none of them (host-side glue between stages); host_s is the time
           until the stage's Python returns, before the closing sync (host_s ~ s: dispatch-bound);
  profile  one fold under cProfile, no syncs, top functions by tottime / cumtime into --out.prof.txt.
  xfer     one fold with ttnn.from_torch / to_torch timed per call site (first tt_bio frame), no syncs: which
           uploads and downloads the host blocks on (cProfile only sees ttnn's decorator as their caller).
  --draw   instead of stages/profile/xfer: one fold with the diffusion step's modules (conditioning, atom encoder
           and its blocks, token DiT and its blocks, atom decoder) synchronized on both sides, nested, so each
           module's wall is its device time when the step is device-bound (host_s well below s); `self_s` is a
           module's time outside its wrapped children. ~36 syncs a step, so read it against the plain fold.
  --ops    instead of stages/profile/xfer: one fold with every top-level ttnn op synchronized on both sides,
           summed by (module region, op) and (region, op, first-input shape) into --out. Needs no Tracy build,
           so it is the census on a Galaxy chip; each op carries one sync round trip (sync_us, reported).
The census (census.sh) gives device kernel seconds per region; stage wall minus that is dispatch.

    python perf/spd_gap_esmf/stages.py --model esmfold2 --out r.json
"""
import argparse, cProfile, collections, functools, io, json, os, pstats, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="esmfold2", choices=("esmfold2", "esmfold2-fast"))
ap.add_argument("--fast", action="store_true")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--ops", action="store_true", help="per-op synchronized census instead of stages/profile/xfer")
ap.add_argument("--draw", action="store_true", help="split the diffusion step by module (nested syncs)")
ap.add_argument("--plain", action="store_true", help="cold + plain fold only; also counts the einsums bmm_res served")
ap.add_argument("--cold-stages", action="store_true", help="split the cold fold by stage too (WH: warm reads slower than cold)")
a = ap.parse_args()

import ttnn
import tt_bio.tenstorrent as T
import tt_baseline as B
from tt_bio.main import (_resolve_recycling_steps, _resolve_sampling_steps,
                         _detect_p300_devices, _find_ttnn_mesh_graph_descriptor)
assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"
if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
    if mgd: os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
from tt_bio import esmfold2 as E, esmfold2_runtime as R

fix = ROOT / "perf" / "size512" / "fixtures"
one_fold, meta = B.build_fold(a.model, ROOT / f".msa_ab_{a.model}_{a.size}", fix / f"cdk2x2_{a.size}.yaml",
                              fix / f"cdk2x2_{a.size}.a3m", fast=a.fast)[:2]
res = {"model": a.model, "fast": a.fast, "card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename}

# (owner, attribute, label). FoldingTrunkModel serves both the folding trunk and the LM encoder's pair
# blocks; the label splits them by block count.
STAGES = [
    (E.FoldingTrunkModel, "__call__", "trunk"),
    (E.StructureHead, "prepare", "diffusion_prepare"),
    (E.StructureHead, "draw", "diffusion_draw"),
    (E.MSAEncoder, "forward", "msa_encoder"),
    (E.LanguageModelShim, "forward", "lm_shim"),
    (R._ESMCAdapter, "__call__", "esmc"),
    (E.InputsEmbedder, "forward", "inputs_embedder"),
    (E.DiffusionConditioning, "forward", "diffusion_conditioning"),
    (E.DistogramHeadModel, "forward", "distogram"),
    (E.RelPosEncoding, "forward", "relpos"),
]
acc = collections.defaultdict(float); host = collections.defaultdict(float); cnt = collections.Counter(); depth = [0]
dev = T.get_device()


def wrap(owner, attr, label):
    fn = getattr(owner, attr)

    @functools.wraps(fn)
    def timed(self, *x, **kw):
        if depth[0]:  # nested stage: its time belongs to the outer one
            return fn(self, *x, **kw)
        lab = label
        if owner is E.FoldingTrunkModel:
            lab = f"trunk[{len(getattr(self, 'blocks', []) or [])}]"
        ttnn.synchronize_device(dev); t0 = time.perf_counter(); depth[0] += 1
        try:
            return fn(self, *x, **kw)
        finally:
            host[lab] += time.perf_counter() - t0  # enqueue done: near the stage wall = host-bound
            ttnn.synchronize_device(dev); depth[0] -= 1
            acc[lab] += time.perf_counter() - t0; cnt[lab] += 1
    setattr(owner, attr, timed)
    return fn


def staged(tag):
    acc.clear(); host.clear(); cnt.clear()
    orig = [(o, at, wrap(o, at, lab)) for o, at, lab in STAGES]
    t0 = time.perf_counter(); s, _ = one_fold(); res[f"{tag}_s"] = round(time.perf_counter() - t0, 3)
    res[tag] = {k: {"s": round(v, 3), "host_s": round(host[k], 3), "n": cnt[k]}
                for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}
    res[f"un{tag}_s"] = round(res[f"{tag}_s"] - sum(acc.values()), 3)
    for o, at, fn in orig:
        setattr(o, at, fn)
    print(json.dumps({k: res[k] for k in (f"{tag}_s", tag, f"un{tag}_s")}, indent=1), flush=True)
    a.out.write_text(json.dumps(res, indent=1))


if a.cold_stages:
    staged("cold_stages")
else:
    one_fold()  # cold
if a.plain:
    from tt_bio import bmm_res
    bres, bres_mm = collections.Counter(), bmm_res.matmul

    def counted(*x, **k):
        r = bres_mm(*x, **k); bres["served" if r is not None else "declined"] += 1
        return r
    bmm_res.matmul = counted
t0 = time.perf_counter(); s, m = one_fold(); res["plain_s"] = round(time.perf_counter() - t0, 3)
res["plain_fold_s"] = round(s, 3); res["plddt"] = m.get("plddt"); res["ptm"] = m.get("ptm")
print("plain", res["plain_s"], flush=True)
if a.plain:
    res["bmm_res"] = dict(bres); a.out.write_text(json.dumps(res, indent=1)); print(res["bmm_res"], flush=True)
    T.cleanup(); sys.exit(0)


def op_census():
    import ttnn.decorators as D
    from tt_bio import esmc as C
    reg, st = [], {"depth": 0, "capture": 0}
    by_op = collections.defaultdict(lambda: [0, 0.0]); by_shape = collections.defaultdict(lambda: [0, 0.0])
    undo = []

    def region(owner, attr, label):
        fn = getattr(owner, attr)

        def w(self, *x, **kw):
            reg.append(f"trunk[{len(getattr(self, 'blocks', []) or [])}]" if owner is E.FoldingTrunkModel else label)
            try:
                return fn(self, *x, **kw)
            finally:
                reg.pop()
        setattr(owner, attr, w); undo.append((owner, attr, fn))

    for o, at, lab in STAGES + [(T.TriangleMultiplication, "__call__", "trimul"), (C.SwiGLUFFN, "__call__", "swiglu"),
                                (C.SwiGLUFFN, "residual", "swiglu"), (E.TransitionLayer, "__call__", "transition"),
                                (E.PairUpdateBlock, "__call__", "pub")]:
        region(o, at, lab)

    def record(name, x, fn, *a, **k):
        if st["depth"] or st["capture"]:  # nested op, or recording a trace (nothing runs, sync is illegal)
            return fn(*a, **k)
        st["depth"] += 1
        try:
            ttnn.synchronize_device(dev); t0 = time.perf_counter()
            out = fn(*a, **k)
            ttnn.synchronize_device(dev); dt = time.perf_counter() - t0
        finally:
            st["depth"] -= 1
        r = "/".join(reg) or "-"
        if name.endswith("generic_op"):   # one name for every custom kernel: key it by the tt_bio function that launched it
            f = sys._getframe(1)
            while f and "/tt_bio/" not in f.f_code.co_filename:
                f = f.f_back
            name = f"{name}@{f.f_code.co_name if f else '?'}"
        shp = tuple(x.shape) if hasattr(x, "shape") else None
        for d, key in ((by_op, (r, name)), (by_shape, (r, name, str(shp)))):
            d[key][0] += 1; d[key][1] += dt
        return out

    def hook(cls):
        fn = cls.__call__
        cls.__call__ = lambda self, *a, **k: record(self.python_fully_qualified_name, a[0] if a else None,
                                                    fn, self, *a, **k)
        undo.append((cls, "__call__", fn))
    hook(D.FastOperation); hook(D.Operation)
    ex = ttnn.execute_trace
    ttnn.execute_trace = lambda *a, **k: record("execute_trace", None, ex, *a, **k)
    undo.append((ttnn, "execute_trace", ex))
    for m in (T, E):
        cap = m.capture_trace

        def captured(*a, _cap=cap, **k):
            st["capture"] += 1
            try:
                return _cap(*a, **k)
            finally:
                st["capture"] -= 1
        m.capture_trace = captured; undo.append((m, "capture_trace", cap))

    n, t0 = 200, time.perf_counter()
    for _ in range(n):
        ttnn.synchronize_device(dev)
    res["sync_us"] = round((time.perf_counter() - t0) / n * 1e6, 1)
    t0 = time.perf_counter(); one_fold(); res["ops_fold_s"] = round(time.perf_counter() - t0, 3)
    for o, at, fn in reversed(undo):
        setattr(o, at, fn)
    rows = lambda d: [{"key": " | ".join(k), "n": v[0], "s": round(v[1], 4)} for k, v in sorted(d.items(), key=lambda kv: -kv[1][1])]
    res["ops_n"] = sum(v[0] for v in by_op.values()); res["ops_sum_s"] = round(sum(v[1] for v in by_op.values()), 3)
    res["by_op"] = rows(by_op); res["by_shape"] = rows(by_shape)[:400]
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("plain_s", "ops_fold_s", "ops_n", "ops_sum_s", "sync_us")}), flush=True)
    for r in res["by_op"][:40]:
        print(r, flush=True)


def draw_split():
    mods = [(E.DiffusionModuleModel, "step", "step"), (E.DiffusionConditioningModel, "cond_single", "cond"),
            (E.AtomEncoder, "__call__", "enc"), (E.AtomDecoder, "__call__", "dec"), (E.SWAAtomBlock, "__call__", "swa"),
            (E.DiffusionTransformerModel, "__call__", "dit"), (E.DiffusionTransformerModel, "_shoist", "shoist"),
            (E.AttentionPairBias, "__call__", "attn"), (E.ConditionedTransitionBlock, "__call__", "trans")]
    stack, tot, hst, kid, n, undo = [], collections.defaultdict(float), collections.defaultdict(float), \
        collections.defaultdict(float), collections.Counter(), []

    def wrap_n(owner, attr, label):
        fn = getattr(owner, attr)

        def timed(self, *x, **kw):
            lab = "/".join([*stack, label])
            ttnn.synchronize_device(dev); t0 = time.perf_counter(); stack.append(label)
            try:
                return fn(self, *x, **kw)
            finally:
                hst[lab] += time.perf_counter() - t0
                ttnn.synchronize_device(dev); stack.pop(); dt = time.perf_counter() - t0
                tot[lab] += dt; n[lab] += 1
                if stack:
                    kid["/".join(stack)] += dt
        setattr(owner, attr, timed); undo.append((owner, attr, fn))

    for o, at, lab in mods:
        wrap_n(o, at, lab)
    t0 = time.perf_counter(); one_fold(); res["draw_fold_s"] = round(time.perf_counter() - t0, 3)
    for o, at, fn in reversed(undo):
        setattr(o, at, fn)
    res["draw"] = {k: {"n": n[k], "s": round(v, 4), "self_s": round(v - kid[k], 4), "host_s": round(hst[k], 4)}
                   for k, v in sorted(tot.items())}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("plain_s", "draw_fold_s", "draw")}, indent=1), flush=True)


if a.ops:
    op_census(); T.cleanup(); sys.exit(0)
if a.draw:
    draw_split(); T.cleanup(); sys.exit(0)
staged("stages")

pr = cProfile.Profile()
t0 = time.perf_counter(); pr.enable(); one_fold(); pr.disable(); res["profile_s"] = round(time.perf_counter() - t0, 3)
buf = io.StringIO()
st = pstats.Stats(pr, stream=buf)
st.sort_stats("tottime").print_stats(60); st.sort_stats("cumulative").print_stats(80)
st.print_callers(r"core.py.*\((from_torch|to_torch)\)")
Path(str(a.out) + ".prof.txt").write_text(buf.getvalue())
a.out.write_text(json.dumps(res, indent=1))

xfer = collections.defaultdict(lambda: [0, 0.0, 0])  # site -> [calls, s, bytes]


_WRAPPERS = ("_from_torch", "_to_torch", "upload", "download", "_to_host")


def _site():
    """First tt_bio frame, followed through the generic upload/download wrappers (download <- _to_torch <- ...)
    to the first frame that is not one, so a site names the code that asked for the transfer."""
    f, out = sys._getframe(2), []
    while f is not None:
        fn = f.f_code.co_filename
        if "/tt_bio/" in fn and "/ttnn/" not in fn:
            out.append(f"{fn.split('/tt_bio/')[-1]}:{f.f_lineno}({f.f_code.co_name})")
            if f.f_code.co_name not in _WRAPPERS or len(out) == 4:
                return " <- ".join(out)
        f = f.f_back
    return " <- ".join(out) or "?"


def _timed(name, fn):
    def wrap(t, *args, **kw):
        t0 = time.perf_counter(); out = fn(t, *args, **kw); dt = time.perf_counter() - t0
        host = t if name == "from_torch" else out
        r = xfer[f"{name} {_site()}"]
        r[0] += 1; r[1] += dt; r[2] += host.numel() * host.element_size() if hasattr(host, "numel") else 0
        return out
    return wrap


orig = {n: getattr(ttnn, n) for n in ("from_torch", "to_torch")}
for n, fn in orig.items():
    setattr(ttnn, n, _timed(n, fn))
t0 = time.perf_counter(); one_fold(); res["xfer_fold_s"] = round(time.perf_counter() - t0, 3)
for n, fn in orig.items():
    setattr(ttnn, n, fn)
res["xfer"] = {k: {"n": v[0], "s": round(v[1], 3), "MB": round(v[2] / 2**20, 1)}
               for k, v in sorted(xfer.items(), key=lambda kv: -kv[1][1])}
res["xfer_total_s"] = round(sum(v[1] for v in xfer.values()), 3)
a.out.write_text(json.dumps(res, indent=1))
T.cleanup()
