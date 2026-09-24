"""Count wrong matmul elements at the op, inside a real tt-bio run.

    python3 perf/mgx_wh_matmul/fold_census.py --label boltz2-3abq --out census.jsonl -- \
        predict perf/mgx/ref/fixtures/3abq_1536.yaml --model boltz2 --out_dir out/

Everything after `--` goes to `python -m tt_bio.main`. Like `perf/mgx_sdpa/census.py`, a
`sitecustomize.py` on PYTHONPATH wraps `ttnn.matmul` and `ttnn.linear` in every process of the
run, because `predict` folds in spawned workers.

Per call it records the signature (caller site, M/K/N, batch, in0_block_w or "auto", fidelity,
fp32 dest, output dtype). A call is SCORED while the time spent scoring stays under
`--frac` of the run's wall so far (the first call of every signature is always scored): up to
`--channels` leading batch slices are read back and compared against a float64 product of the
same operands (the device's own bf16 / bfp8 values, so input quantisation is not counted).

Rows past MM_CENSUS_ROWS (4096) of a tall operand are not read (the K and N axes always are).

wrong: |err| > 0.25 * (sqrt(sum_k a_k^2 b_k^2) + 16 output ulps of |ref|). The dot product's own
       scale: HiFi4's ordinary accumulation error is ~1e-3 of it and the -2^k misses are one to
       tens of times it, so the bar sits in the empty gap between them. `max_q` is the largest
       |err| / that scale seen per signature.
gross: |err| > 5 % of the largest |ref| in the scored slice

Calls inside a trace capture are counted, never read (a host read breaks the capture). Run with
tracing off where the model allows, so the trunk's matmuls are reachable.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

HOOK = r'''
import os
if os.environ.get("MM_CENSUS_DIR"):
    import atexit, json, math, time, traceback
    import torch, ttnn
    _T0 = time.time()
    _spent = [0.0]
    _frac = float(os.environ.get("MM_CENSUS_FRAC", "0.5"))
    _nch = int(os.environ.get("MM_CENSUS_CHANNELS", "2"))
    _rows, _cap = {}, [0]
    _om, _ol = ttnn.matmul, ttnn.linear
    _begin, _end = ttnn.begin_trace_capture, ttnn.end_trace_capture

    def _b(*a, **kw):
        _cap[0] += 1
        return _begin(*a, **kw)

    def _e(*a, **kw):
        _cap[0] -= 1
        return _end(*a, **kw)
    ttnn.begin_trace_capture, ttnn.end_trace_capture = _b, _e

    def _site():
        for fr in reversed(traceback.extract_stack()[:-2]):
            if "/tt_bio/" in fr.filename:
                return f"{fr.filename.split('/tt_bio/')[-1]}:{fr.lineno}:{fr.name}"
        return "?"

    def _ulp(x, dt):
        # one unit in the last place of |x| in the output dtype (bf16: 8 significant bits)
        bits = {"DataType.FLOAT32": 24, "DataType.BFLOAT8_B": 8}.get(dt, 8)
        e = torch.floor(torch.log2(x.abs().clamp(min=1e-30)))
        return torch.pow(2.0, e - (bits - 1))

    def _dump():
        p = os.path.join(os.environ["MM_CENSUS_DIR"], f"{os.getpid()}.json")
        with open(p + ".tmp", "w") as f:
            json.dump(_rows, f)
        os.replace(p + ".tmp", p)

    _rmax = int(os.environ.get("MM_CENSUS_ROWS", "4096"))

    def _host(t, n, rows=False):
        """The first n slices of the batch dims (and, with rows, the first _rmax rows) as float64."""
        sh = [int(s) for s in t.shape]
        end = list(sh)
        if len(sh) >= 3:
            end[:-3] = [1] * (len(sh) - 3)
            end[-3] = min(n, sh[-3])
        if rows and sh[-2] > _rmax:
            end[-2] = _rmax
        if end == sh:
            return ttnn.to_torch(t).double()
        s = ttnn.slice(t, [0] * len(sh), end)
        h = ttnn.to_torch(s).double()
        ttnn.deallocate(s)
        return h

    def _wrap(orig, kind):
        def call(a, b, *args, **kw):
            o = orig(a, b, *args, **kw)
            try:
                ta, tb = bool(kw.get("transpose_a")), bool(kw.get("transpose_b"))
                sa, sb = [int(s) for s in a.shape], [int(s) for s in b.shape]
                K = sa[-2] if ta else sa[-1]
                M = sa[-1] if ta else sa[-2]
                N = sb[-2] if tb else sb[-1]
                pc = kw.get("program_config")
                ck = kw.get("compute_kernel_config")
                key = json.dumps([_site(), kind, M, K, N, int(math.prod(sa[:-2])),
                                  getattr(pc, "in0_block_w", "auto") if pc is not None else "auto",
                                  type(pc).__name__ if pc is not None else None,
                                  str(getattr(ck, "math_fidelity", None)), getattr(ck, "fp32_dest_acc_en", None),
                                  getattr(ck, "packer_l1_acc", None), str(o.dtype), str(a.dtype), str(b.dtype),
                                  kw.get("core_grid") is not None])
                r = _rows.setdefault(key, {"calls": 0, "scored": 0, "elems": 0, "wrong": 0, "gross": 0,
                                           "worst": [], "capture": 0})
                r["calls"] += 1
                if _cap[0]:
                    r["capture"] += 1
                    return o
                if kw.get("activation") or (args and kind == "linear"):
                    r["skip"] = "activation or positional bias"
                    return o
                wall = time.time() - _T0
                if r["scored"] and _spent[0] > _frac * wall:
                    return o
                t0 = time.time()
                A, B, O = _host(a, _nch, rows=not ta), _host(b, _nch), _host(o, _nch, rows=not ta)
                if ta:
                    A = A.transpose(-1, -2)
                if tb:
                    B = B.transpose(-1, -2)
                n = min(A.shape[-3] if A.dim() >= 3 else 1, O.shape[-3] if O.dim() >= 3 else 1)
                if A.dim() >= 3:
                    A = A[..., :n, :, :]
                if B.dim() >= 3 and B.shape[-3] > 1:
                    B = B[..., :n, :, :]
                O = O[..., :n, :, :] if O.dim() >= 3 else O
                ref = torch.matmul(A, B)
                bias = kw.get("bias")
                if bias is not None:
                    ref = ref + ttnn.to_torch(bias).double().reshape(-1)[:ref.shape[-1]]
                scale = torch.matmul(A * A, B * B).sqrt()
                e = O.reshape(ref.shape) - ref
                q = e.abs() / (scale + 16 * _ulp(ref, str(o.dtype)))
                w = q > 0.25
                g = e.abs() > 0.05 * ref.abs().max()
                r["max_q"] = max(r.get("max_q", 0.0), round(float(q.max()), 4))
                r["scored"] += 1
                r["elems"] += ref.numel()
                r["wrong"] += int(w.sum())
                r["gross"] += int(g.sum())
                if len(r["worst"]) < 8 and w.any():
                    idx = torch.nonzero(w)[:8 - len(r["worst"])].tolist()
                    r["worst"] += [[round(float(ref[tuple(i)]), 4), round(float(e[tuple(i)]), 4),
                                    round(float(scale[tuple(i)]), 4)] for i in idx]
                _spent[0] += time.time() - t0
                r["score_s"] = round(r.get("score_s", 0) + time.time() - t0, 2)
                if r["scored"] == 1 or r["calls"] % 64 == 0:
                    _dump()
            except Exception as ex:  # a failed score must not fail the run it counts
                r = _rows.setdefault("error", {"calls": 0})
                r["calls"] += 1
                r["last"] = str(ex).splitlines()[0][:200] if str(ex) else type(ex).__name__
            return o
        return call

    ttnn.matmul, ttnn.linear = _wrap(_om, "matmul"), _wrap(_ol, "linear")
    atexit.register(_dump)
'''

FIELDS = ("site", "kind", "M", "K", "N", "batch", "in0_block_w", "config", "fidelity", "fp32_dest",
          "l1_acc", "out", "a_dtype", "b_dtype", "core_grid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.5, help="scoring time budget, fraction of wall")
    ap.add_argument("--channels", type=int, default=2, help="batch slices read back per scored call")
    ap.add_argument("--env", action="append", default=[], metavar="K=V", help="extra env for the run")
    ap.add_argument("cli", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cli = a.cli[1:] if a.cli[:1] == ["--"] else a.cli
    with tempfile.TemporaryDirectory() as hook, tempfile.TemporaryDirectory() as dump:
        Path(hook, "sitecustomize.py").write_text(HOOK)
        env = dict(os.environ, MM_CENSUS_DIR=dump, MM_CENSUS_FRAC=str(a.frac),
                   MM_CENSUS_CHANNELS=str(a.channels),
                   PYTHONPATH=os.pathsep.join([hook, str(REPO)] + [p for p in os.environ.get(
                       "PYTHONPATH", "").split(os.pathsep) if p]), **dict(e.split("=", 1) for e in a.env))
        rc = subprocess.call([sys.executable, "-m", "tt_bio.main"] + cli, cwd=REPO, env=env)
        merged = {}
        for p in glob.glob(os.path.join(dump, "*.json")):
            for key, r in json.load(open(p)).items():
                m = merged.setdefault(key, {})
                for k, v in r.items():
                    m[k] = (max(m.get(k, 0), v) if k == "max_q" else m.get(k, 0) + v) if isinstance(v, (int, float)) \
                        else (m.get(k) or v)
    err = merged.pop("error", None)
    rows = [dict(zip(FIELDS, json.loads(k)), **r) for k, r in merged.items()]
    rows.sort(key=lambda r: (-r["wrong"], r["site"]))
    tot = {k: sum(r.get(k, 0) for r in rows) for k in ("calls", "scored", "elems", "wrong", "gross", "capture")}
    with open(a.out, "a") as f:
        f.write(json.dumps({"label": a.label, "cli": cli, "env": a.env, "rc": rc, **tot, "error": err, "rows": rows}) + "\n")
    print(f"{a.label}: rc={rc} {tot} error={err}")
    for r in rows[:40]:
        print("  ", {k: r.get(k) for k in ("site", "M", "K", "N", "batch", "in0_block_w", "core_grid",
                                            "calls", "scored", "elems", "wrong", "gross", "max_q", "capture")},
              r.get("worst", [])[:3])
    return rc


if __name__ == "__main__":
    sys.exit(main())
