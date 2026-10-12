#!/usr/bin/env python3
"""One OpenFold3 token-DiT block (`_DiTBlock`) under the `--fast` lever set: block time, op table, numerics.

    python perf/spd_gap_of3/dit_ops.py --out OUT.json [--n 512] [--reps 20] [--arms off,dit_fuse]
        [--ckpt of3-p2-155k.pt] [--ops] [--block 0]

The block is built the way the fold builds it: `OF3DiffusionTransformer(n_blocks=1)` from the real checkpoint's
`diffusion_module.diffusion_transformer` (block `--block` renamed to 0), inside `lever_scope(openfold3_fold.FAST_LEVERS
+ arm)`, at the act dtype `diffusion_bf16` gives (bf16, no device_dtype_override), the worker's HiFi4 / fp32-acc
config. Inputs are random (a, s ~ N(0, 1), z ~ N(0, 1), mask all ones) at N tokens, uploaded bf16 as the stack
receives them; the pair bias is cached as the rollout caches it (one warm call first). Per arm:

* `eager_ms`: median of `--reps` calls, `synchronize_device` on both sides.
* `trace_ms`: the same call captured once and replayed `--reps` times, each replay synchronised -- the device time,
  which is what the traced rollout pays (4800 block calls a fold at 24 blocks x 200 steps).
* `shared_ms`: the per-stack-call work every block shares (`OF3DiffusionTransformer.shared_s`), paid once per
  24 blocks; reported apart, and `per_block_ms = trace_ms + shared_ms / 24`.
* `ops` (with --ops): every ttnn op the block issues, synchronised before and after, binned by op and call site
  (`op  <file:line of the caller>  <- <openfold3_diffusion_transformer.py line>`), with `sync_floor_us` (the same
  bracket around an empty op) to subtract.
* numerics: the device output against a float64 torch reference of the same block on the same bf16-rounded inputs
  (`rel_rms`, `max_abs` of the output, and of the residual update `out - a`).
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

import torch

C_A, C_S, C_Z, H, DH = 768, 384, 128, 16, 48


def ref_block(sd, a, s, z, tok):
    """float64 OF3 DiffusionTransformerBlock (non-cross path) on [N, C] inputs."""
    d = {k: v.double() for k, v in sd.items()}
    ln = lambda x, w=None: torch.nn.functional.layer_norm(x, x.shape[-1:], w, None, 1e-5)  # noqa: E731

    def adaln(x, p):
        sn = ln(s, d[p + "layer_norm_s.weight"])
        g = torch.sigmoid(sn @ d[p + "linear_g.weight"].T + d[p + "linear_g.bias"])
        return g * ln(x) + sn @ d[p + "linear_s.weight"].T

    apb, ct = "attention_pair_bias.", "conditioned_transition."
    N = a.shape[0]
    a_ln = adaln(a, apb + "layer_norm_a.")
    zb = ln(z, d[apb + "layer_norm_z.weight"]) @ d[apb + "linear_z.weight"].T      # [N, N, 16]
    bias = zb.permute(2, 0, 1) + ((tok - 1.0) * 1e9)[None, None, :]
    q = (a_ln @ d[apb + "mha.linear_q.weight"].T + d[apb + "mha.linear_q.bias"]).view(N, H, DH).transpose(0, 1)
    k = (a_ln @ d[apb + "mha.linear_k.weight"].T).view(N, H, DH).transpose(0, 1)
    v = (a_ln @ d[apb + "mha.linear_v.weight"].T).view(N, H, DH).transpose(0, 1)
    att = torch.softmax(q @ k.transpose(-1, -2) * DH ** -0.5 + bias, dim=-1)
    o = (att @ v).transpose(0, 1).reshape(N, H * DH)
    o = o * torch.sigmoid(a_ln @ d[apb + "mha.linear_g.weight"].T)
    o = o @ d[apb + "mha.linear_o.weight"].T
    o = o * torch.sigmoid(s @ d[apb + "linear_ada_out.weight"].T + d[apb + "linear_ada_out.bias"])
    a = a + o
    a_t = adaln(a, ct + "layer_norm.")
    tr = torch.nn.functional.silu(a_t @ d[ct + "swiglu.linear_a.weight"].T) * (a_t @ d[ct + "swiglu.linear_b.weight"].T)
    tr = (tr @ d[ct + "linear_out.weight"].T) * torch.sigmoid(s @ d[ct + "linear_g.weight"].T + d[ct + "linear_g.bias"])
    return a + tr * tok[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--arms", default="off", help="comma list; `off` = OpenFold3's FAST_LEVERS, a name = that + it")
    ap.add_argument("--ckpt", default=os.path.expanduser("~/spd/spd-shipflow/weights/of3-p2-155k.pt"))
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--ops", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio import openfold3_fold as OF
    from tt_bio import openfold3_diffusion_transformer as MDT
    dev = T.get_device(trace="openfold3")
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    pre = f"diffusion_module.diffusion_transformer.blocks.{a.block}."
    blk_sd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    del sd
    dt_sd = {"blocks.0." + k: v for k, v in blk_sd.items()}

    g = torch.Generator().manual_seed(a.seed)
    N = a.n
    a_h = torch.randn(N, C_A, generator=g)
    s_h = torch.randn(N, C_S, generator=g)
    z_h = torch.randn(N, N, C_Z, generator=g)
    tok = torch.ones(N)
    up = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)  # noqa: E731
    a_d, s_d, z_d = up(a_h[None]), up(s_h[None]), up(z_h[None])
    tok_d, tmc_d = up(tok[None]), up(tok[None, :, None])
    rb = lambda x: x.to(torch.bfloat16).double()  # noqa: E731
    ref = ref_block(blk_sd, rb(a_h), rb(s_h), rb(z_h), tok.double())
    a_in = rb(a_h)

    def sync():
        ttnn.synchronize_device(dev)

    res = {"n": N, "reps": a.reps, "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "arch": str(dev.arch()), "ckpt": a.ckpt, "block": a.block, "arms": {}}
    try:
        res["aiclk"] = [Path(p).read_text().strip() for p in
                        sorted(Path("/sys/class/tenstorrent").glob("*/tt_aiclk"))][:1]
    except Exception:                                                     # noqa: BLE001
        pass

    def flush():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))

    for arm in a.arms.split(","):
        spec = OF.FAST_LEVERS | (T.parse_levers(arm) if arm != "off" else frozenset())
        row = {"levers": sorted(spec)}
        res["arms"][arm] = row
        with T.lever_scope(spec):
            dit = MDT.OF3DiffusionTransformer(dt_sd, T.lpx_compute_kernel_config(ckc), n_blocks=1)
            blk = dit.blocks[0]
            row["act_dtype"] = str(blk._act_dtype)
            row["paths"] = {"sdpa32": bool(blk.sdpa32), "lp": blk.lp is not None, "s16": blk.s16 is not None,
                            "fuse": bool(getattr(blk, "fuse", None))}
            cache = {}
            rows = T.sdpa32_rows(N) if blk.sdpa32 else N
            assert rows == N, ("bench assumes no sdpa32 padding", N, rows)
            mask_bias, tmc = dit._masks(tok_d, N, N, rows)
            tmc = tmc_d if isinstance(tmc, tuple) else tmc

            def shared():
                return dit.shared_s(s_d)

            def call(sh):
                return blk(a_d, s_d, z_d, mask_bias, tmc, cache=cache, s16=sh, tok_ones=dit._tok_ones.get(rows, False))

            def free(sh):
                for t in sh or ():
                    ttnn.deallocate(t)

            sh = shared()
            out = call(sh)                                   # warm: programs + the cached pair bias
            sync()
            o_h = ttnn.to_torch(out).double().reshape(N, C_A)
            ttnn.deallocate(out)
            fin = bool(torch.isfinite(o_h).all())
            err = o_h - ref
            upd_ref, upd = ref - a_in, o_h - a_in
            row["numerics"] = {
                "finite": fin,
                "rel_rms": float(err.norm() / ref.norm()), "max_abs": float(err.abs().max()),
                "upd_rel_rms": float((upd - upd_ref).norm() / upd_ref.norm()),
                "upd_max_abs": float((upd - upd_ref).abs().max()),
                "ref_rms": float(ref.pow(2).mean().sqrt()), "upd_ref_rms": float(upd_ref.pow(2).mean().sqrt())}
            print(arm, json.dumps(row["numerics"]), flush=True)
            for _ in range(2):
                ttnn.deallocate(call(sh))
            sync()

            ts = []
            for _ in range(a.reps):
                sync(); t0 = time.perf_counter()
                o = call(sh)
                sync(); ts.append(time.perf_counter() - t0)
                ttnn.deallocate(o)
            row["eager_ms"] = round(statistics.median(ts) * 1e3, 4)
            free(sh)
            ts = []
            for _ in range(a.reps):
                sync(); t0 = time.perf_counter()
                x = shared()
                sync(); ts.append(time.perf_counter() - t0)
                free(x)
            row["shared_ms"] = round(statistics.median(ts) * 1e3, 4)
            sh = shared()

            # Traced: what the rollout replays.
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            o_tr = call(sh)
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            sync()
            ts = []
            for _ in range(a.reps):
                sync(); t0 = time.perf_counter()
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                sync(); ts.append(time.perf_counter() - t0)
            row["trace_ms"] = round(statistics.median(ts) * 1e3, 4)
            sync(); t0 = time.perf_counter()
            for _ in range(a.reps):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            sync()
            row["trace_mean_back_to_back_ms"] = round((time.perf_counter() - t0) / a.reps * 1e3, 4)
            # The replayed output equals the eager one (same programs, same inputs).
            o2 = ttnn.to_torch(o_tr).double().reshape(N, C_A)
            row["trace_matches_eager"] = bool(torch.equal(o2, o_h))
            ttnn.release_trace(dev, tid)
            ttnn.deallocate(o_tr)
            row["per_block_ms"] = round(row["trace_mean_back_to_back_ms"] + row["shared_ms"] / 24, 4)
            print(arm, {k: row[k] for k in ("eager_ms", "trace_ms", "trace_mean_back_to_back_ms", "shared_ms",
                                            "per_block_ms", "trace_matches_eager")}, flush=True)
            flush()

            if a.ops:
                row["ops"] = op_table(ttnn, MDT, lambda: call(sh), sync, dev, min(a.reps, 5))
                flush()
            free(sh)
            ttnn.deallocate(mask_bias)
            for v in cache.values():
                for t in (v if isinstance(v, tuple) else (v,)):
                    if isinstance(t, ttnn.Tensor):
                        ttnn.deallocate(t)
            del dit, blk
    flush()


def op_table(ttnn, MDT, call, sync, dev, reps):
    """Every ttnn op one block call issues, synchronised on both sides, binned by op and call site."""
    acc, inside, do_trace, seen = {}, {"op": 0}, {"v": True}, set()
    HOST_ONLY = {"unsqueeze", "squeeze", "reshape", "allocate_tensor_on_device"}

    def addrs(v):
        ts = v if isinstance(v, (tuple, list)) else (v,)
        return {t.buffer_address() for t in ts if isinstance(t, ttnn.Tensor) and t.is_allocated()
                and t.storage_type() == ttnn.StorageType.DEVICE}

    def traced(fn, x, k, out):
        """The op alone, captured into a trace and replayed: its device time without host issue.
        An in-place op re-mutates its input on every replay; the census pass reads no values."""
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        o2 = fn(*x, **k)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        sync()
        R = 10
        t0 = time.perf_counter()
        for _ in range(R):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        sync()
        dt = (time.perf_counter() - t0) / R
        ttnn.release_trace(dev, tid)
        keep = addrs(list(x) + list(k.values())) | addrs(out)
        for t in (o2 if isinstance(o2, (tuple, list)) else (o2,)):
            if isinstance(t, ttnn.Tensor) and t.storage_type() == ttnn.StorageType.DEVICE and t.is_allocated() \
                    and t.buffer_address() not in keep:
                ttnn.deallocate(t)
        return dt
    here = Path(MDT.__file__).name
    names = ["matmul", "linear", "layer_norm", "rms_norm", "permute", "transpose", "reshape", "to_layout",
             "typecast", "multiply", "multiply_", "add", "add_", "subtract", "sigmoid", "concat", "slice", "pad",
             "clone", "reallocate", "to_memory_config", "softmax", "generic_op", "addcmul", "addalpha", "where",
             "unsqueeze", "squeeze", "copy", "move", "allocate_tensor_on_device", "silu", "mul", "fill",
             "sharded_to_interleaved", "interleaved_to_sharded", "sum", "mean", "rsqrt", "sub", "exp"]
    saved = []

    def wrap(owner, name, label):
        fn = getattr(owner, name, None)
        if fn is None or not callable(fn):
            return

        def inner(*x, **k):
            if inside["op"]:
                return fn(*x, **k)
            f = sys._getframe(1)
            caller = f"{Path(f.f_code.co_filename).name}:{f.f_lineno}"
            site = ""
            while f is not None:
                if Path(f.f_code.co_filename).name == here:
                    site = f"{here}:{f.f_lineno}"
                    break
                f = f.f_back
            key = f"{label} {caller}" + (f" <- {site}" if site and site != caller else "")
            sync()
            inside["op"] = 1
            t0 = t1 = time.perf_counter()
            dev_s = 0.0
            try:
                out = fn(*x, **k)
                sync()
                t1 = time.perf_counter()
                if do_trace["v"] and label not in HOST_ONLY:
                    if key not in seen:
                        print("trace", key, flush=True)
                    dev_s = traced(fn, x, k, out)
                    if key not in seen:
                        seen.add(key)
                        print("  ok", round(dev_s * 1e6, 1), flush=True)
                return out
            finally:
                inside["op"] = 0
                r = acc.setdefault(key, [0.0, 0, 0.0])
                r[0] += t1 - t0
                r[1] += 1
                r[2] += dev_s
        saved.append((owner, name, fn))
        setattr(owner, name, inner)

    for n in names:
        wrap(ttnn, n, n)
    for n in [n for n in dir(ttnn.experimental) if not n.startswith("_")]:
        wrap(ttnn.experimental, n, "experimental." + n)
    try:
        for _ in range(2):
            ttnn.deallocate(call())
        acc.clear()
        for _ in range(reps):
            ttnn.deallocate(call())
        # The bracket's floor: sync + an op that does nothing on the device; an empty trace's replay.
        do_trace["v"] = False
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        sync(); t0 = time.perf_counter()
        for _ in range(10):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        sync()
        empty_trace_us = (time.perf_counter() - t0) / 10 * 1e6
        ttnn.release_trace(dev, tid)
        fl = []
        for _ in range(50):
            sync(); t0 = time.perf_counter(); sync(); fl.append(time.perf_counter() - t0)
    finally:
        for owner, name, fn in reversed(saved):
            setattr(owner, name, fn)
    rows = sorted(({"key": k, "dev_us": round(v[2] / reps * 1e6, 2), "us_per_block": round(v[0] / reps * 1e6, 2),
                    "n_per_block": v[1] / reps} for k, v in acc.items()), key=lambda r: -r["dev_us"])
    return {"sync_floor_us": round(statistics.median(fl) * 1e6, 2), "empty_trace_us": round(empty_trace_us, 2),
            "n_ops": sum(r["n_per_block"] for r in rows), "sum_us": round(sum(r["us_per_block"] for r in rows), 1),
            "sum_dev_us": round(sum(r["dev_us"] for r in rows), 1), "rows": rows}


if __name__ == "__main__":
    main()
