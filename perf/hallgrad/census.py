#!/usr/bin/env python3
"""BCX census: where one fwd+bwd pair unit's `b` goes, per ttnn op, on real silicon.

`b` is the slope of step time in the number of pair units (blocksweep.py). This script splits it
by op without a device profiler, which the tt-bio wheel does not ship:

  1. Time warm steps at K=1 and K=2, interleaved, same process and clock. b = t(K=2) - t(K=1),
     reported as the distribution of paired differences.
  2. Run ONE more step at each K with every `ttnn.<op>` call wrapped. Each outermost call is
     timed twice: once between two synchronize_device calls ("synced", which is device time
     plus one dispatch plus one sync round trip), then replayed R times back to back on the
     same inputs with one sync at the end ("replay", the pipelined per-call time, which hides
     dispatch whenever the kernel is longer than its enqueue). In-place ops and deallocation
     are never replayed.
  3. The unit's ops are the K=2 calls minus the K=1 calls, per signature. Everything outside
     the unit (pair init, head, loss seed) cancels, the same way it cancels out of `b`.
  4. Roofs are measured in this process, on this card, at this clock: a HiFi4/fp32-acc matmul
     (the config every autograd op uses), the default-config matmul p2 quoted, and DRAM
     streaming rates for copy (clone), eltwise (add), permute, layernorm, softmax and sum at
     the pair tensor's own size. An op's roof time is max(FLOPs / matmul roof, bytes / stream
     roof) at its own shape.

`b - sum(replay)` is what the unit costs beyond its kernels: dispatch gaps the queue does not
hide, allocation, host Python in the tape, and syncs. It is reported as its own line.
"""
import argparse
import ast
import collections
import json
import os
import pathlib
import socket
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.hallgrad.e2e_distogram import ClockTrace, make_weights, tt_chain  # noqa: E402

C_Z, C_S, HEADS, HEAD_DIM, BINS, CONTACT_BINS, MIN_SEP = 128, 384, 4, 32, 64, 20, 6

MATMUL = {"matmul", "linear"}
MOVE = {"permute", "transpose", "reshape", "concat", "slice", "getitem", "repeat",
        "to_memory_config", "typecast", "clone", "pad"}
CLASS = {**{k: "matmul" for k in MATMUL}, **{k: "data-movement" for k in MOVE},
         "layer_norm": "layernorm", "softmax": "softmax", "sum": "reduction",
         "mean": "reduction", "deallocate": "dealloc"}
NO_REPLAY = {"deallocate"}
WRAP = ["matmul", "linear", "add", "multiply", "subtract", "mean", "reshape", "concat",
        "layer_norm", "deallocate", "sum", "typecast", "softmax", "sigmoid", "zeros",
        "to_memory_config", "slice", "rsub", "rsqrt", "permute", "transpose", "relu",
        "ones_like", "gtz", "divide", "clone", "repeat", "pad", "moreh_linear_backward"]


def elem_bytes(ttnn, dt):
    return {ttnn.bfloat16: 2.0, ttnn.float32: 4.0, ttnn.bfloat8_b: 1088 / 1024,
            ttnn.uint32: 4.0, ttnn.int32: 4.0}.get(dt, 2.0)


def is_tensor(ttnn, v):
    return isinstance(v, ttnn.Tensor)


def tbytes(ttnn, t):
    try:
        shape = [int(d) for d in t.padded_shape]
    except Exception:
        shape = [int(d) for d in t.shape]
    return float(np.prod(shape)) * elem_bytes(ttnn, t.dtype)


def tdesc(ttnn, t):
    try:
        buf = "L1" if t.memory_config().buffer_type == ttnn.BufferType.L1 else "DRAM"
    except Exception:
        buf = "?"
    return f"{list(int(d) for d in t.shape)}:{str(t.dtype).split('.')[-1]}:{buf}"


def flat_tensors(ttnn, v):
    if is_tensor(ttnn, v):
        return [v]
    if isinstance(v, (list, tuple)):
        return [x for e in v for x in flat_tensors(ttnn, e)]
    return []


def arg_desc(ttnn, v):
    if is_tensor(ttnn, v):
        return tdesc(ttnn, v)
    if isinstance(v, (list, tuple)):
        return "(" + ",".join(arg_desc(ttnn, e) for e in v) + ")"
    if isinstance(v, slice):
        return f"{v.start}:{v.stop}"
    if isinstance(v, (int, float, bool, str)) or v is None:
        return repr(v)
    return type(v).__name__


def op_flops(name, args, kwargs, out):
    if name not in MATMUL or out is None:
        return 0.0
    a = args[0]
    oshape = [int(d) for d in out.shape]
    if name == "linear":
        k = int(a.shape[-1])
    else:
        k = int(a.shape[-2] if kwargs.get("transpose_a") else a.shape[-1])
    return 2.0 * float(np.prod(oshape)) * k


class Recorder:
    """Wraps ttnn.<op> and ttnn.Tensor.__getitem__; records only the outermost call."""

    def __init__(self, ttnn, device, replay):
        self.ttnn, self.device, self.replay = ttnn, device, replay
        self.rec, self.active, self.depth, self.phase = None, False, 0, None
        self.orig = {}
        self.ag_ranges = self._ag_ranges()

    @staticmethod
    def _ag_ranges():
        src = (ROOT / "tt_bio" / "autograd.py").read_text()
        out = []
        for node in ast.parse(src).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                out.append((node.lineno, node.end_lineno, node.name))
        return out

    def _origin(self):
        """The autograd verb that issued this call: outermost autograd frame, not the driver."""
        f, names = sys._getframe(2), []
        while f is not None:
            co = f.f_code
            if co.co_filename.endswith("tt_bio/autograd.py") and co.co_name != "backward":
                ln = co.co_firstlineno
                top = next((n for a, b, n in self.ag_ranges if a <= ln <= b), co.co_name)
                names.append(top)
            f = f.f_back
        return names[-1] if names else "harness"

    def _wrap(self, name, fn):
        ttnn, rec = self.ttnn, self

        def w(*args, **kwargs):
            if not rec.active or rec.depth:
                return fn(*args, **kwargs)
            rec.depth += 1
            try:
                origin = rec._origin()
                ttnn.synchronize_device(rec.device)
                t0 = time.perf_counter()
                out = fn(*args, **kwargs)
                ttnn.synchronize_device(rec.device)
                t1 = time.perf_counter()
                rep = None
                if name not in NO_REPLAY and not name.endswith("_"):
                    t2 = time.perf_counter()
                    for _ in range(rec.replay):
                        o = fn(*args, **kwargs)
                        del o
                    ttnn.synchronize_device(rec.device)
                    rep = (time.perf_counter() - t2) / rec.replay
                ins = flat_tensors(ttnn, list(args) + list(kwargs.values()))
                outs = flat_tensors(ttnn, out)
                sig = ",".join(arg_desc(ttnn, a) for a in args)
                kw = ",".join(f"{k}={arg_desc(ttnn, v)}" for k, v in sorted(kwargs.items())
                              if k in ("transpose_a", "transpose_b", "dim", "dims", "keepdim",
                                       "dtype"))
                rec.rec.append({
                    "name": name, "origin": origin, "phase": rec.phase,
                    "sig": sig + (f" |{kw}" if kw else ""),
                    "out": ",".join(tdesc(ttnn, t) for t in outs),
                    "synced_s": t1 - t0, "replay_s": rep,
                    "flops": op_flops(name, args, kwargs, outs[0] if outs else None),
                    "bytes": sum(tbytes(ttnn, t) for t in ins + outs),
                })
                return out
            finally:
                rec.depth -= 1
        return w

    def install(self):
        for n in WRAP:
            if hasattr(self.ttnn, n):
                self.orig[n] = getattr(self.ttnn, n)
                setattr(self.ttnn, n, self._wrap(n, self.orig[n]))
        gi = self.ttnn.Tensor.__getitem__
        self.orig["__getitem__"] = gi
        wrapped = self._wrap("getitem", gi)
        self.ttnn.Tensor.__getitem__ = lambda self_, key: wrapped(self_, key)
        return self


def one_step(ag, ttnn, device, Wtt, cfg, logits_np, mask_t, M, inC, rec=None):
    """Forward, host loss seed, backward, and both host reads -- halloop's step without Adam."""
    t0 = time.perf_counter()
    if rec:
        rec.phase = "fwd"
    lt = ag.Tensor(ttnn.from_torch(torch.from_numpy(logits_np).to(torch.bfloat16),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                   requires_grad=True)
    d = tt_chain(ag, ttnn, lt, Wtt, cfg)
    z64 = ttnn.to_torch(d.value).to(torch.float64)
    t_fwd = time.perf_counter() - t0
    p = torch.softmax(z64, dim=-1)
    pc = p[..., :CONTACT_BINS].sum(-1).clamp_min(1e-12)
    seed_t = -(p * (inC / pc.unsqueeze(-1) - 1.0)) * (mask_t / M).unsqueeze(-1)
    reached = len(ag._reverse_topo([d]))
    if rec:
        rec.phase = "bwd"
    t1 = time.perf_counter()
    d.backward(seed=ttnn.from_torch(seed_t.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=device))
    g = ttnn.to_torch(lt.grad).to(torch.float32)
    t_bwd = time.perf_counter() - t1
    if not bool(torch.isfinite(g).all()) or not float(g.norm()) > 0:
        raise RuntimeError("logit gradient is not finite or is zero")
    lt.grad = lt.node = d.grad = d.node = None
    del lt, d, z64, p, pc, seed_t, g
    return {"step_s": time.perf_counter() - t0, "fwd_s": t_fwd, "bwd_s": t_bwd, "reached": reached}


def timed(ttnn, device, fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / iters


def measure_roofs(ttnn, device, n, reps=3):
    """Same-process references per op class. Each is taken `reps` times; median and spread kept."""
    import tt_bio.autograd as ag
    hifi4 = ag.precise_config()

    def T(shape, dt=torch.bfloat16, tdt=None):
        return ttnn.from_torch(torch.randn(*shape).to(dt), dtype=tdt or ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=device)
    pair = T((n, n, C_Z))
    pair2 = T((n, n, C_Z))
    flat = T((n * n, C_Z))
    w = T((C_Z, C_Z))
    sq = T((4096, 4096))
    sq2 = T((4096, 4096))
    per_ch_a = T((C_Z, n, n))
    per_ch_b = T((C_Z, n, n))
    scores = T((128, HEADS, 128, n))
    gamma = T((1, C_Z))
    pair_bytes = n * n * C_Z * 2.0
    items = {
        "matmul_default_65536x128x128": (lambda: ttnn.matmul(flat, w), 2.0 * n * n * C_Z * C_Z, None),
        "matmul_hifi4_65536x128x128": (lambda: ttnn.matmul(flat, w, compute_kernel_config=hifi4),
                                       2.0 * n * n * C_Z * C_Z, None),
        "matmul_hifi4_4096cube": (lambda: ttnn.matmul(sq, sq2, compute_kernel_config=hifi4),
                                  2.0 * 4096 ** 3, None),
        "matmul_default_4096cube": (lambda: ttnn.matmul(sq, sq2), 2.0 * 4096 ** 3, None),
        "matmul_hifi4_trimul_128x256x256x256": (
            lambda: ttnn.matmul(per_ch_a, per_ch_b, transpose_b=True, compute_kernel_config=hifi4),
            2.0 * C_Z * n ** 3, None),
        "clone_pair": (lambda: ttnn.clone(pair), None, 2 * pair_bytes),
        "add_pair": (lambda: ttnn.add(pair, pair2), None, 3 * pair_bytes),
        "multiply_pair": (lambda: ttnn.multiply(pair, pair2), None, 3 * pair_bytes),
        "sigmoid_pair": (lambda: ttnn.sigmoid(pair), None, 2 * pair_bytes),
        "permute_pair_201": (lambda: ttnn.permute(pair, (2, 0, 1)), None, 2 * pair_bytes),
        "permute_pair_102": (lambda: ttnn.permute(pair, (1, 0, 2)), None, 2 * pair_bytes),
        "layer_norm_pair": (lambda: ttnn.layer_norm(pair, weight=gamma, epsilon=1e-6,
                                                    compute_kernel_config=hifi4),
                            None, 2 * pair_bytes),
        "softmax_scores_128x4x128x256": (lambda: ttnn.softmax(scores, dim=-1,
                                                             compute_kernel_config=hifi4),
                                         None, 2 * 128 * HEADS * 128 * n * 2.0),
        "sum_pair_dim0": (lambda: ttnn.sum(pair, dim=0, keepdim=True), None, pair_bytes),
        "clone_4096sq": (lambda: ttnn.clone(sq), None, 2 * 4096 * 4096 * 2.0),
        "add_4096sq": (lambda: ttnn.add(sq, sq2), None, 3 * 4096 * 4096 * 2.0),
    }
    out = {}
    for name, (fn, flops, byts) in items.items():
        ts = [timed(ttnn, device, fn) for _ in range(reps)]
        t = statistics.median(ts)
        out[name] = {"s": t, "s_all": ts,
                     "tflops": flops / t / 1e12 if flops else None,
                     "gbs": byts / t / 1e9 if byts else None}
        print(f"  roof {name:40s} {t * 1e3:8.4f} ms  "
              + (f"{out[name]['tflops']:7.2f} TFLOP/s" if flops else f"{out[name]['gbs']:7.1f} GB/s"),
              flush=True)
    return out


def stamp(clocks=None):
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except Exception:
            return None
    vis = os.environ.get("TT_VISIBLE_DEVICES", "")
    return {
        "host": socket.gethostname(),
        "board_subsystem_device": {c: sh(f"cat '/sys/class/tenstorrent/tenstorrent!{c}/device/subsystem_device'")
                                   for c in vis.split(",") if c},
        "board_subsystem_vendor": {c: sh(f"cat '/sys/class/tenstorrent/tenstorrent!{c}/device/subsystem_vendor'")
                                   for c in vis.split(",") if c},
        "git_sha": sh(f"git -C {ROOT} rev-parse HEAD"),
        "git_dirty": sh(f"git -C {ROOT} status --porcelain -- tt_bio perf | wc -l"),
        "tt_visible_devices": vis,
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "loadavg": os.getloadavg(), "nproc": os.cpu_count(),
        "top_cpu": sh("ps -eo pcpu,comm --sort=-pcpu | head -4"),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--pairs", type=int, default=20, help="interleaved K=1/K=2 timed step pairs")
    ap.add_argument("--replay", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag

    device = tt.get_device()
    clocks = ClockTrace(period=1.0).start()
    blob = {"argv": " ".join(sys.argv), "stamp_start": stamp(), "n": args.n, "replay": args.replay,
            "chunk": args.chunk, "c_z": C_Z, "heads": HEADS, "head_dim": HEAD_DIM,
            "block_transition": True, "checkpoint": False}

    def flush():
        blob["clock_meta"] = clocks.summary()
        blob["clock_samples"] = clocks.samples
        json.dump(blob, open(args.out, "w"), indent=1)

    n = args.n
    idx = np.arange(n)
    mask = np.abs(idx[:, None] - idx[None, :]) >= MIN_SEP
    M = int(mask.sum())
    mask_t = torch.from_numpy(mask).to(torch.float64)
    inC = torch.zeros(BINS, dtype=torch.float64)
    inC[:CONTACT_BINS] = 1.0
    logits = (np.random.default_rng(args.seed).standard_normal((n, 21)) * 0.1).astype(np.float32)

    arms = {}
    for K in (1, 2):
        Wnp = make_weights(np.random.default_rng(args.seed), C_S, C_Z, HEADS, HEAD_DIM, C_Z, BINS,
                           blocks=K, block_transition=True)
        Wtt = {k: ag.Tensor(ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(v)).to(torch.bfloat16),
                                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device))
               for k, v in Wnp.items()}
        cfg = dict(heads=HEADS, head_dim=HEAD_DIM, hidden=C_Z, chunk=args.chunk, checkpoint=False,
                   blocks=K, block_transition=True)
        arms[K] = (Wtt, cfg)

    step = lambda K, rec=None: one_step(ag, ttnn, device, *arms[K], logits, mask_t, M, inC, rec)

    print("# warm-up", flush=True)
    for _ in range(args.warm):
        for K in (1, 2):
            step(K)

    print(f"# {args.pairs} interleaved K=1/K=2 pairs", flush=True)
    t_win0 = time.time()
    s1, s2 = [], []
    for i in range(args.pairs):
        a, b = (1, 2) if i % 2 == 0 else (2, 1)
        ra, rb = step(a), step(b)
        (s1 if a == 1 else s2).append(ra)
        (s1 if b == 1 else s2).append(rb)
    t_win1 = time.time()
    diffs = [y["step_s"] - x["step_s"] for x, y in zip(s1, s2)]
    blob["timed"] = {
        "t_K1": [r["step_s"] for r in s1], "t_K2": [r["step_s"] for r in s2],
        "fwd_K1": [r["fwd_s"] for r in s1], "fwd_K2": [r["fwd_s"] for r in s2],
        "bwd_K1": [r["bwd_s"] for r in s1], "bwd_K2": [r["bwd_s"] for r in s2],
        "reached": [s1[0]["reached"], s2[0]["reached"]],
        "b_paired": diffs,
        "b_median": statistics.median(diffs), "b_mean": statistics.mean(diffs),
        "b_p10": float(np.percentile(diffs, 10)), "b_p90": float(np.percentile(diffs, 90)),
        "b_from_medians": statistics.median([r["step_s"] for r in s2])
                          - statistics.median([r["step_s"] for r in s1]),
        "clock_window": clocks.window(t_win0, t_win1),
        "window": [t_win0, t_win1],
    }
    tb = blob["timed"]
    print(f"  b median {tb['b_median']:.5f} s  mean {tb['b_mean']:.5f}  p10 {tb['b_p10']:.5f}  "
          f"p90 {tb['b_p90']:.5f}  | AICLK {tb['clock_window']}", flush=True)
    flush()

    print("# instrumented steps", flush=True)
    rec = Recorder(ttnn, device, args.replay).install()
    blob["records"] = {}
    for K in (1, 2):
        rec.rec = []
        rec.active = True
        t0 = time.time()
        r = step(K, rec)
        rec.active = False
        blob["records"][str(K)] = {"calls": rec.rec, "step": r,
                                   "clock_window": clocks.window(t0, time.time())}
        print(f"  K={K}: {len(rec.rec)} outermost ttnn calls, instrumented step {r['step_s']:.2f} s "
              f"| AICLK {blob['records'][str(K)]['clock_window']}", flush=True)
        flush()

    print("# roofs", flush=True)
    t0 = time.time()
    blob["roofs"] = measure_roofs(ttnn, device, n)
    blob["roofs_clock_window"] = clocks.window(t0, time.time())
    print(f"  roofs AICLK {blob['roofs_clock_window']}", flush=True)

    # re-take b after everything, to see whether the box drifted
    t_win0 = time.time()
    s1b, s2b = [], []
    for i in range(6):
        a, b = (1, 2) if i % 2 == 0 else (2, 1)
        ra, rb = step(a), step(b)
        (s1b if a == 1 else s2b).append(ra)
        (s1b if b == 1 else s2b).append(rb)
    d2 = [y["step_s"] - x["step_s"] for x, y in zip(s1b, s2b)]
    blob["timed_after"] = {"b_paired": d2, "b_median": statistics.median(d2),
                           "clock_window": clocks.window(t_win0, time.time())}
    print(f"  b after: median {statistics.median(d2):.5f} s | AICLK {blob['timed_after']['clock_window']}",
          flush=True)

    clocks.stop()
    blob["stamp_end"] = stamp()
    flush()
    print(f"# wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
