#!/usr/bin/env python3
"""Where one ESMFold2 diffusion step goes on one chip: eager vs trace-replayed, step vs token DiT vs sub-block vs op.

spd-bet-ditpersist's falsifier needs to know how much of a step is per-op cost (program launch, CB setup, the
DRAM round trip between ops) that a persistent program removes, and how much is work it keeps. The fold is
built as stages.py builds it; inside the first draw(), on the tensors prepare() left resident, each item runs

  eager   N calls back to back, one sync at the end: device time + host dispatch, whichever binds;
  traced  the same call captured once and replayed N times (no host dispatch): device time alone;

and its ttnn op count (deallocate excluded). Items: the whole denoiser step, the token DiT (12 blocks), one
AttentionPairBias and one ConditionedTransitionBlock, the atom encoder, and chains of 20 single ops at the
token shape ([1, L, 768] bf16: add, multiply, sigmoid, layer_norm, 768x768 linear, 768x3072 linear), whose
traced per-op time is the floor a fused program does not pay between ops, and the same chains with the
activation in L1 (`x20_l1_*`). Then the draw runs normally.

    python perf/spd/dit_step_probe.py --model esmfold2-fast --fast --size 512 --out r.json
"""
import argparse, json, math, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="esmfold2-fast", choices=("esmfold2", "esmfold2-fast"))
ap.add_argument("--fast", action="store_true")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch
import ttnn
import ttnn.decorators as D
import tt_bio.tenstorrent as T
import tt_baseline as B
from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
from tt_bio import esmfold2 as E

B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
fix = ROOT / "perf" / "size512" / "fixtures"
one_fold, meta = B.build_fold(a.model, ROOT / f".msa_ab_{a.model}_{a.size}", fix / f"cdk2x2_{a.size}.yaml",
                              fix / f"cdk2x2_{a.size}.a3m", fast=a.fast)[:2]
res = {"model": a.model, "fast": a.fast, "size": a.size, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "host": os.uname().nodename, "trace_region": T.trace_region_size(), "items": {}}

count = [0, False]


def counting(cls):
    fn = cls.__call__

    def w(self, *x, **k):
        if count[1] and not self.python_fully_qualified_name.endswith("deallocate"):
            count[0] += 1
        return fn(self, *x, **k)
    cls.__call__ = w


counting(D.FastOperation); counting(D.Operation)


def free(o):
    for t in (o if isinstance(o, (tuple, list)) else (o,)):
        if isinstance(t, ttnn.Tensor) and t.is_allocated():
            ttnn.deallocate(t)


def bench(dev, name, fn, n):
    r = {}
    for _ in range(2):
        free(fn()); ttnn.synchronize_device(dev)
    count[0], count[1] = 0, True
    free(fn()); ttnn.synchronize_device(dev)
    count[1] = False; r["ops"] = count[0]
    ts = []
    for _ in range(3):
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        for _ in range(n):
            free(fn())
        ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) / n * 1e3)
    r["eager_ms"] = round(statistics.median(ts), 4)
    if T.trace_region_size() > 0:
        try:
            tid, out = T.capture_trace(dev, fn)
            for _ in range(2):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ts = []
            for _ in range(3):
                ttnn.synchronize_device(dev); t0 = time.perf_counter()
                for _ in range(n):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) / n * 1e3)
            r["traced_ms"] = round(statistics.median(ts), 4)
            ttnn.release_trace(dev, tid); free(out)
        except Exception as e:  # region too small for this item, or an op that refuses capture
            r["trace_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    if r["ops"]:
        r["eager_us_per_op"] = round(r["eager_ms"] * 1e3 / r["ops"], 2)
        if "traced_ms" in r:
            r["traced_us_per_op"] = round(r["traced_ms"] * 1e3 / r["ops"], 2)
    res["items"][name] = r
    print(name, json.dumps(r), flush=True)
    a.out.write_text(json.dumps(res, indent=1))


def chain(op, x, k=20):
    def run():
        y = x
        for _ in range(k):
            z = op(y)
            if y is not x:
                ttnn.deallocate(y)
            y = z
        return y
    return run


def probe(head):
    dmm, dev, c = head.module, head.tt_device, head.module._ctx
    torch.manual_seed(0)
    L, N = c["s_inputs"].shape[1], head._n_atoms
    t = torch.tensor([head.sigma_data * 2.0])
    t_noise = 0.25 * torch.log(t / head.sigma_data)
    n_raw = torch.cos(2.0 * math.pi * (t_noise[:, None] * head._fw[None, :] + head._fb[None, :])).float()
    r = torch.randn(1, N, 3)
    r_input = torch.cat([r, torch.zeros_like(r)], dim=-1).float()
    ft = head._from_torch
    nt, rt = ft(n_raw), ft(r_input)
    s = dmm.conditioning.cond_single(c["s_inputs"], nt)
    x = ttnn.from_torch(torch.randn(1, L, 768), layout=ttnn.TILE_LAYOUT, device=dev, dtype=E._DTYPE)
    tf, z = dmm.token_transformer, c["z"]
    at, tr = tf.attn[0], tf.trans[0]
    res["L"], res["N_atoms"] = L, N
    res["t0"] = time.time()
    n = a.reps
    bench(dev, "step", lambda: dmm.step(nt, rt), n)
    bench(dev, "token_dit", lambda: tf(x, s, z), n)
    bench(dev, "attn_block", lambda: at(x, s, z), 4 * n)
    bench(dev, "trans_block", lambda: tr(x, s), 4 * n)
    enc = lambda: dmm.atom_encoder(c["atom_feats"], c["cos"], c["sin"], c["band"], c["scatter_m"], r_input=rt,
                                   valid=c["valid"])[:2]  # [2] is the encoder's cached c: not ours to free
    bench(dev, "atom_encoder", enc, n)
    ck, lin = at.compute_kernel_config, at._lin
    ops = {
        "add": lambda y: ttnn.add(y, x),
        "multiply": lambda y: ttnn.multiply(y, x),
        "sigmoid": lambda y: ttnn.sigmoid(y),
        "layer_norm": lambda y: ttnn.layer_norm(y, epsilon=1e-5, compute_kernel_config=ck),
        "linear768": lambda y: lin(y, at.q_w, at.q_b),
    }
    for k, op in ops.items():
        bench(dev, f"x20_{k}", chain(op, x), 2 * n)
    bench(dev, "x20_linear768x3072", chain(lambda y: lin(x, tr.swish_w), x), 2 * n)
    # the same chains with the activation in L1 (interleaved): what an op costs once it stops round-tripping DRAM
    L1 = ttnn.L1_MEMORY_CONFIG
    xl = ttnn.to_memory_config(x, L1)
    ops_l1 = {
        "add": lambda y: ttnn.add(y, xl, memory_config=L1),
        "sigmoid": lambda y: ttnn.sigmoid(y, memory_config=L1),
        "layer_norm": lambda y: ttnn.layer_norm(y, epsilon=1e-5, compute_kernel_config=ck, memory_config=L1),
        "linear768": lambda y: lin(y, at.q_w, at.q_b, memory_config=L1),
    }
    for k, op in ops_l1.items():
        bench(dev, f"x20_l1_{k}", chain(op, xl), 2 * n)
    free(xl)
    res["t1"] = time.time()
    free([s, x, nt, rt])
    a.out.write_text(json.dumps(res, indent=1))


orig = E.StructureHead.draw
done = [False]


def draw(self, *x, **k):
    if not done[0]:
        done[0] = True
        probe(self)
    return orig(self, *x, **k)


E.StructureHead.draw = draw
t0 = time.perf_counter(); one_fold(); res["fold_s"] = round(time.perf_counter() - t0, 3)
a.out.write_text(json.dumps(res, indent=1))
print("done", res["fold_s"], flush=True)
T.cleanup()
