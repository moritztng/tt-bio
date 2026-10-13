#!/usr/bin/env python3
"""Where the wall of one warm ESMFold2 fold goes, on the GAP-table workload (cdk2x2_512, shipped steps, 1 sample).

Builds the fold through tt_baseline.build_fold (as ab.py), runs one cold fold, then:
  plain    one fold, nothing wrapped (the reference wall);
  stages   one fold with every stage below wrapped in synchronize_device on both sides, so each stage's
           wall is its own device time plus its own host dispatch, and the sum against the plain wall
           says how much of the fold is in none of them (host-side glue between stages);
  profile  one fold under cProfile, no syncs, top functions by tottime / cumtime into --out.prof.txt.
  xfer     one fold with ttnn.from_torch / to_torch timed per call site (first tt_bio frame), no syncs: which
           uploads and downloads the host blocks on (cProfile only sees ttnn's decorator as their caller).
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
acc = collections.defaultdict(float); cnt = collections.Counter(); depth = [0]
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
            ttnn.synchronize_device(dev); depth[0] -= 1
            acc[lab] += time.perf_counter() - t0; cnt[lab] += 1
    setattr(owner, attr, timed)
    return fn


one_fold()  # cold
t0 = time.perf_counter(); s, m = one_fold(); res["plain_s"] = round(time.perf_counter() - t0, 3)
res["plain_fold_s"] = round(s, 3); res["plddt"] = m.get("plddt")
print("plain", res["plain_s"], flush=True)

orig = [(o, at, wrap(o, at, lab)) for o, at, lab in STAGES]
t0 = time.perf_counter(); s, _ = one_fold(); res["stages_s"] = round(time.perf_counter() - t0, 3)
res["stages"] = {k: {"s": round(v, 3), "n": cnt[k]} for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}
res["unstaged_s"] = round(res["stages_s"] - sum(acc.values()), 3)
for o, at, fn in orig:
    setattr(o, at, fn)
print(json.dumps(res, indent=1), flush=True)
a.out.write_text(json.dumps(res, indent=1))

pr = cProfile.Profile()
t0 = time.perf_counter(); pr.enable(); one_fold(); pr.disable(); res["profile_s"] = round(time.perf_counter() - t0, 3)
buf = io.StringIO()
st = pstats.Stats(pr, stream=buf)
st.sort_stats("tottime").print_stats(60); st.sort_stats("cumulative").print_stats(80)
st.print_callers(r"core.py.*\((from_torch|to_torch)\)")
Path(str(a.out) + ".prof.txt").write_text(buf.getvalue())
a.out.write_text(json.dumps(res, indent=1))

xfer = collections.defaultdict(lambda: [0, 0.0, 0])  # site -> [calls, s, bytes]


def _site():
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if "/tt_bio/" in fn and "/ttnn/" not in fn:
            return f"{fn.split('/tt_bio/')[-1]}:{f.f_lineno}({f.f_code.co_name})"
        f = f.f_back
    return "?"


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
