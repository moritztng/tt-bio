#!/usr/bin/env python3
"""gap_itemise: per-op, per-call-site device cost of every RFD3 step region, for one gap_screen arm.

p49_region_itemise's method (synchronize on both sides of every ttnn op while the stack is inside a region) on
today's regions and lever set, so it runs on any arch the pool hands out:
  token_encoder  DiffusionTokenEncoder.run_device
  dit            LocalTokenTransformer.run_device
  decoder        CompactStreamingDecoder.run_full_device
  encoder        LocalAtomTransformer.__call__
An op is labelled op@file:line of its caller, plus the caller's caller when the caller is a helper (_tuned_linear,
pad_axis, ...), so a shared helper still splits by site. The synced sum overshoots the region's real wall by the
per-op dispatch the syncs expose; read the ranking and the ratios, and set them beside the unsynced region wall that
is measured on the warmup steps of the same run.

    gap_itemise.py ARM OUT.json [STEPS=8] [--fast=1] [--fixture=perf/dsfix/fixtures/rfd3_R4.json]
"""
import collections, json, os, pathlib, shutil, statistics, sys, time

ENV_KNOBS = {"gsm": "RFD3_GATHERED_SOFTMAX", "bs": "RFD3_BLOCK_SPARSE",
             "pzc": "RFD3_PROCESS_Z_COLLAPSE", "fgrid": "RFD3_FAST_GRID",
             "asdpa": "RFD3_ATOM_SDPA", "tb8": "RFD3_TRANSITION_B8",
             "dsdpa": "RFD3_DIT_SDPA", "hifi2": "RFD3_HIFI2", "sf": "RFD3_SIGMOID_FOLD",
             "rs": "RFD3_RESIDENT_STEP", "pws": "RFD3_PAIR_WS", "kd": "RFD3_ATTN_KDTREE"}
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
ARM, OUT = args[0], pathlib.Path(args[1])
STEPS = int(args[2]) if len(args) > 2 else 8
WARM = 3
for _k, _env in ENV_KNOBS.items():
    os.environ[_env] = "1" if _k in set(ARM.split("+")) else "0"   # read at import

sys.path.insert(0, os.getcwd())
from tt_bio.main import ensure_p300_mesh_descriptor                      # noqa: E402

ensure_p300_mesh_descriptor()
import ttnn                                                              # noqa: E402
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402

WEIGHTS = pathlib.Path(os.environ.get("BOLTZ_CACHE", pathlib.Path.home() / ".boltz")) / "rfd3" / "weights"
FIXTURE = opts.get("fixture", "perf/dsfix/fixtures/rfd3_R4.json")
OPS = ("linear", "matmul", "add", "multiply", "subtract", "softmax", "typecast", "scatter", "embedding", "reshape",
       "permute", "pad", "slice", "to_layout", "concat", "rms_norm", "layer_norm", "sigmoid", "clone", "full",
       "from_torch", "to_torch", "transpose", "repeat", "sum", "mean", "exp", "where", "zeros", "silu", "generic_op",
       "gather", "deallocate", "copy", "fill", "untilize", "tilize")
HELPERS = {"_tuned_linear", "pad_axis", "_gated", "_sdpa_dense_mask", "_dev", "_tt", "_tt_idx", "fused_sdpa",
           "attn_value_matmul", "_adaln", "heads", "split", "run"}
REGION, STEP = [None], [0]
ACC = collections.defaultdict(lambda: [0.0, 0])
WALL, SYNCED = collections.defaultdict(list), collections.defaultdict(list)


def _sync():
    ttnn.synchronize_device(M.get_device())


def _site(depth):
    fr = sys._getframe(depth)
    s = "%s:%d" % (os.path.basename(fr.f_code.co_filename), fr.f_lineno)
    if fr.f_code.co_name in HELPERS and fr.f_back is not None:
        b = fr.f_back
        s += "<%s:%d" % (os.path.basename(b.f_code.co_filename), b.f_lineno)
    return s


def _wrap(owner, name, label):
    fn = getattr(owner, name, None)
    if fn is None or not callable(fn):
        return

    def w(*a, **k):
        if REGION[0] is None or name == "deallocate":
            return fn(*a, **k)
        key = (REGION[0], "%s@%s" % (label, _site(2)))
        _sync()
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _sync()
            e = ACC[key]
            e[0] += time.perf_counter() - t0
            e[1] += 1
    setattr(owner, name, w)


def _region(cls, meth, region):
    fn = getattr(cls, meth)

    def w(self, *a, **k):
        if REGION[0] is not None:                   # nested region: the outer one owns it
            return fn(self, *a, **k)
        counted = STEP[0] >= WARM
        _sync()
        t0 = time.perf_counter()
        REGION[0] = region if counted else None
        try:
            return fn(self, *a, **k)
        finally:
            REGION[0] = None
            _sync()
            (SYNCED if counted else WALL)[region].append(time.perf_counter() - t0)
    setattr(cls, meth, w)


def main():
    for n in OPS:
        _wrap(ttnn, n, n)
    _wrap(ttnn.transformer, "scaled_dot_product_attention", "sdpa")
    for n in ("nlp_create_qkv_heads", "nlp_concat_heads"):
        _wrap(ttnn.experimental, n, n)
    regions = (("token_encoder", M.DiffusionTokenEncoder, "run_device"),
               ("dit", M.LocalTokenTransformer, "run_device"),
               ("decoder", M.CompactStreamingDecoder, "run_full_device"),
               ("encoder", M.LocalAtomTransformer, "__call__"))
    for region, cls, meth in regions:
        _region(cls, meth, region)
    call = M.RFD3DiffusionModule.__call__

    def stepped(self, *a, **k):
        try:
            return call(self, *a, **k)
        finally:
            STEP[0] += 1
    M.RFD3DiffusionModule.__call__ = stepped

    specs = json.loads(pathlib.Path(FIXTURE).read_text())
    out_dir = "/tmp/gap_itemise_%s" % ARM
    shutil.rmtree(out_dir, ignore_errors=True)
    t0 = time.perf_counter()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=str(WEIGHTS), from_pdb=True, num_timesteps=STEPS, seed=42,
                           num_designs=1, batch_size=1, fast=opts.get("fast") == "1", verbose=False)
    counted = STEP[0] - WARM
    rec = {"arm": ARM, "fast": opts.get("fast") == "1", "fixture": FIXTURE, "steps": STEPS, "counted_steps": counted,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename, "arch": str(M.get_device().arch()),
           "wall_s": round(time.perf_counter() - t0, 1), "regions": {}}
    for region, _, _ in regions:
        rows = sorted(({"op": lab, "calls_per_step": round(n / counted, 2), "ms_per_step": round(1e3 * s / counted, 3)}
                       for (r, lab), (s, n) in ACC.items() if r == region), key=lambda r: -r["ms_per_step"])
        per_call = statistics.median(WALL[region]) * 1e3 if WALL[region] else 0.0
        calls = len(SYNCED[region]) / max(counted, 1)
        rec["regions"][region] = {"synced_sum_ms_per_step": round(sum(r["ms_per_step"] for r in rows), 2),
                                  "unsynced_wall_ms_per_step": round(per_call * calls, 2),
                                  "calls_per_step": calls, "rows": rows}
        print("\n=== %s: synced sum %.1f, unsynced wall %.1f ms/step ===" % (
            region, rec["regions"][region]["synced_sum_ms_per_step"], rec["regions"][region]["unsynced_wall_ms_per_step"]))
        for r in rows[:25]:
            print("%-70s %8.2f %7.1f" % (r["op"], r["ms_per_step"], r["calls_per_step"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
