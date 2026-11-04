"""Device time of every op in one Mol-1 diffusion step, without a profiler build.

    cd state/mol1/inputs && TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 python <tt-bio>/perf/mol1/bh/diffstep.py \
        --input mol1-256.json --out <dir>

Folds once cold, then once warm with the first denoise of the diffusion stage instrumented. That denoise runs
eagerly (edm_device traces from the second step on), so each ttnn op it dispatches is called for real, then called
again inside a trace of its own, and that trace is replayed until it has run >= 3 ms; the replay time over the count
is the op's device time with no host dispatch in it. The whole denoise, the atom encoder, the token DiT and the atom
decoder are each also captured whole, so the per-op sum can be checked against them. One line per op goes to
<out>/ops.jsonl (op, shapes, dtypes, FLOP, bytes, ms, the tt_bio call chain); diffstep_table.py turns it into the
per-family table. The instrumented fold's coordinates are not graded: ops that write into an input are not re-run
(they are listed with ms null), every other re-run writes a fresh buffer.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fair"))

import clocksample  # noqa: E402
from run import run_config  # noqa: E402

SKIP = {"copy", "deallocate", "fill_implicit_tile_padding", "from_torch", "to_torch", "to_device", "from_device",
        "copy_host_to_device_tensor", "copy_device_to_host_tensor", "synchronize_device", "begin_trace_capture",
        "end_trace_capture", "execute_trace", "release_trace", "dump_device_profiler", "open_device", "close_device",
        "init_device_compute_kernel_config", "get_device_tensors", "allocate_tensor_on_device",
        "allocate_tensor_on_host", "empty", "empty_like", "zeros", "ones", "full", "arange", "rand"}
ESIZE = {"FLOAT32": 4, "BFLOAT16": 2, "BFLOAT8_B": 1088 / 1024, "BFLOAT4_B": 576 / 1024, "UINT32": 4, "INT32": 4,
         "UINT16": 2, "UINT8": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="mol1")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--min-ms", type=float, default=3.0, help="replay each op until this much device time has run")
    ap.add_argument("--cli-extra", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import worker as W

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, jobs = run_config(a.model, Path(a.input).resolve(), out, a.cli_extra)
    W._ensure_local_artifacts(cfg)
    state = W._WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("mol1-diffstep", cfg)
    m = state.model
    D = m.diffusion
    dev = D.dev
    st = {"armed": False, "depth": 0, "stack": [], "ops": [], "whole": {}, "seq": 0}

    def tensors(x):
        if isinstance(x, ttnn.Tensor):
            yield x
        elif isinstance(x, (list, tuple)):
            for y in x:
                yield from tensors(y)
        elif isinstance(x, dict):
            for y in x.values():
                yield from tensors(y)

    def desc(t):
        try:
            dt = str(t.dtype).split(".")[-1]
            return {"shape": list(t.shape), "dtype": dt, "dev": t.storage_type() == ttnn.StorageType.DEVICE,
                    "bytes": math.prod(t.padded_shape) * ESIZE.get(dt, 2)}
        except Exception:
            return None

    def replay(fn):
        """Capture fn() alone and replay it: (device ms per call, error)."""
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        try:
            res = fn()
        except Exception as e:  # noqa: BLE001
            try:
                ttnn.end_trace_capture(dev, tid, cq_id=0)
            finally:
                ttnn.release_trace(dev, tid)
            return None, repr(e)[:200]
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        try:
            n = 3
            while True:
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(n):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev)
                dt = (time.perf_counter() - t0) * 1000
                if dt >= a.min_ms or n >= 400:
                    return dt / n, None
                n = min(400, max(2 * n, int(n * a.min_ms / max(dt, 1e-3)) + 1))
        finally:
            ttnn.release_trace(dev, tid)
            for t in tensors(res):
                try:
                    ttnn.deallocate(t)
                except Exception:  # noqa: BLE001
                    pass

    def chain():
        f, c = sys._getframe(2), []
        while f is not None and len(c) < 8:
            if "tt_bio" in f.f_code.co_filename:
                s = f.f_locals.get("self")
                c.append(f"{type(s).__name__ + '.' if s is not None else ''}{f.f_code.co_name}:{f.f_lineno}")
            f = f.f_back
        return c

    def op_wrap(name, fn):
        def w(*args, **kw):
            if not st["armed"] or st["depth"] or not st["stack"]:   # only inside the instrumented denoise
                return fn(*args, **kw)
            st["depth"] += 1
            try:
                r = fn(*args, **kw)
                ins = [desc(t) for t in tensors((args, kw))]
                outs = [desc(t) for t in tensors(r)]
                addr_in = set()
                for t in tensors((args, kw)):
                    try:
                        addr_in.add(t.buffer_address())
                    except Exception:  # noqa: BLE001
                        pass
                inplace = any(k in kw for k in ("output_tensor", "optional_output_tensor"))
                for t in tensors(r):
                    try:
                        inplace |= t.buffer_address() in addr_in
                    except Exception:  # noqa: BLE001
                        pass
                ms, err = (None, "in-place, view or allocation: not re-run") if (
                    inplace or name in SKIP or not any(i and i["dev"] for i in ins)) else replay(
                    lambda: fn(*args, **kw))
                st["seq"] += 1
                st["ops"].append({"i": st["seq"], "op": name, "fam": list(st["stack"]), "chain": chain(),
                                  "ins": ins, "outs": outs, "ms": ms, "err": err,
                                  "kw": {k: str(v)[:80] for k, v in kw.items() if not isinstance(v, ttnn.Tensor)}})
                return r
            finally:
                st["depth"] -= 1
        return w

    for mod, pre in ((ttnn, ""), (ttnn.transformer, "transformer."), (ttnn.experimental, "experimental.")):
        for n in dir(mod):
            o = getattr(mod, n)
            if type(o).__name__ == "FastOperation":
                setattr(mod, n, op_wrap(pre + n, o))

    def fam_wrap(obj, attr, fam, last=False):
        fn = getattr(obj, attr)

        def w(*args, **kw):
            if not st["armed"] or st["depth"]:
                return fn(*args, **kw)
            st["stack"].append(fam)
            try:
                r = fn(*args, **kw)
            finally:
                st["stack"].pop()
            st["depth"] += 1
            try:
                ms, err = replay(lambda: fn(*args, **kw))
            finally:
                st["depth"] -= 1
            st["whole"].setdefault(fam, []).append({"ms": ms, "err": err})
            if last:
                st["armed"] = False
            return r
        if callable(fn) and not hasattr(fn, "__func__"):   # a module object (atxE): keep its other methods
            class Proxy:
                __call__ = staticmethod(w)

                def __getattr__(self, k):
                    return getattr(fn, k)
            w = Proxy()
        setattr(obj, attr, w)

    fam_wrap(D, "_denoise_device", "denoise", last=True)
    fam_wrap(D, "_denoise_device_m", "denoise", last=True)
    fam_wrap(D, "atxE", "atom_enc")
    fam_wrap(D, "atxD", "atom_dec")
    fam_wrap(D, "_token_dit_device", "dit")
    fam_wrap(D, "_dit_shoist", "dit_shoist")

    with open(out / "folds.jsonl", "a") as f:
        for i in range(2):
            st["armed"] = i == 1
            sdir = out / f"r{i}"
            sdir.mkdir(exist_ok=True)
            with clocksample.during(period=1.0) as clk:
                t0 = time.perf_counter()
                metrics, _, _ = state.predict_one(jobs[0], dict(cfg, seed=a.seed + i, struct_dir=str(sdir)))
                wall = time.perf_counter() - t0
            line = {"kind": "cold" if i == 0 else "instrumented", "input": Path(a.input).name,
                    "tokens": metrics.get("n_tokens"), "wall_s": round(wall, 3), "aiclk": clk.summary().get(0),
                    "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            print(json.dumps(line), flush=True)
            f.write(json.dumps(line) + "\n")
    with open(out / "ops.jsonl", "w") as f:
        for o in st["ops"]:
            f.write(json.dumps(o) + "\n")
    (out / "whole.json").write_text(json.dumps(st["whole"], indent=1))
    print(f"ops {len(st['ops'])} whole {json.dumps(st['whole'])}", flush=True)


if __name__ == "__main__":
    main()
