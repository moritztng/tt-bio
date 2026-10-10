"""Inside a real fold, run every large OuterProductMean call whole AND row-blocked on the same inputs, and grade
both against a float64 host reference of the same call.

    python perf/spd_msa/opm_whole_vs_blocked.py [MIN_TOKENS=512] perf/spd/bench.py --out ... --inputs 9W8A ...

The blocked run goes first, on a copy of the residual only, so the whole call meets the memory the fold gives
it; the fold continues on the whole call's output. The float64 reference covers 64 token rows (first, middle,
last), the full one being ~5 GB at 778 tokens. Each probed call logs
to <out>/opm_wvb.jsonl: tokens, the path census delta of each run, and for whole / blocked the max abs and
relative-L2 error against float64 plus whole-vs-blocked. The blocked run is forced through the engine's own
refusal memo (_OPM_DRAM_ROW_CAP), so it is the path a refused whole call takes.
"""
import importlib.abc
import importlib.util
import json
import runpy
import sys
import time
from pathlib import Path

MIN_TOK = 512
if sys.argv[1:2] and sys.argv[1].isdigit():
    MIN_TOK = int(sys.argv.pop(1))
SCRIPT = sys.argv[1]
sys.argv = sys.argv[1:]
OUT = Path(sys.argv[sys.argv.index("--out") + 1])
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "opm_wvb.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time()
    LOG.write(json.dumps(kw, default=str) + "\n")
    LOG.flush()
    print("opm_wvb", json.dumps(kw, default=str), file=sys.stderr, flush=True)


def install(T):
    import torch
    ttnn = T.ttnn
    call = T.OuterProductMean.__call__
    n = [0]

    def host(t):
        return ttnn.to_torch(t).double()

    def reference(self, x, mask, n_msa, residual, rows):
        """float64 output at token `rows` only: the full one is (I, J, C*D) doubles, ~5 GB at 778 tokens."""
        keys = ("norm.weight", "norm.bias", "proj_a.weight", "proj_b.weight", "proj_o.weight", "proj_o.bias",
                "proj_a.bias", "proj_b.bias")
        w = {k: self.weights[k].double() for k in keys if k in self.weights}
        z = 0
        for s in range(0, x.shape[0], 1024):
            m = torch.nn.functional.layer_norm(x[s:s + 1024].double(), x.shape[-1:], w["norm.weight"],
                                               w["norm.bias"], 1e-5)
            a = (m[:, rows] @ w["proj_a.weight"].t() + w.get("proj_a.bias", 0))
            b = m @ w["proj_b.weight"].t() + w.get("proj_b.bias", 0)
            if mask is not None:
                a = a * mask.reshape(x.shape[0], x.shape[1], -1)[s:s + 1024, rows].double()
            z = z + torch.einsum("sic,sjd->ijcd", a, b)
        z = z.reshape(len(rows), x.shape[1], -1)
        scale = 1 / (n_msa if n_msa is not None else x.shape[0])
        if self.scale_bias:
            o = (z @ w["proj_o.weight"].t() + w["proj_o.bias"]) * scale
        else:
            o = (z * scale) @ w["proj_o.weight"].t() + w["proj_o.bias"]
        return o if residual is None else residual.reshape(x.shape[1], x.shape[1], -1)[rows].double() + o

    def err(got, ref):
        got = got.reshape(ref.shape)
        d = got - ref
        return {"max_abs": float(d.abs().max()), "rel_l2": float(d.norm() / ref.norm()),
                "ref_l2": float(ref.norm())}

    def probed(self, x, msa_mask=None, n_msa=None, residual=None):
        k = n[0]
        n[0] += 1
        # A depth-chunk list carries plain tensors (the mask is then None) or (chunk, mask) pairs.
        chunks = [c if isinstance(c, tuple) else (c, None) for c in x] if isinstance(x, list) else [(x, msa_mask)]
        I, cm = tuple(chunks[0][0].shape)[-2:]
        if I < MIN_TOK:
            return call(self, x, msa_mask, n_msa, residual)
        xs = torch.cat([ttnn.to_torch(c).reshape(-1, I, cm) for c, _ in chunks])
        ms = None if chunks[0][1] is None else torch.cat([ttnn.to_torch(m).reshape(-1, I, 1) for _, m in chunks])
        rs = None if residual is None else ttnn.to_torch(residual)
        rows = sorted(set(range(16)) | set(range(I // 2 - 8, I // 2 + 8)) | set(range(I - 32, I)))
        # Blocked first, on a copy of the residual (a blocked join frees it); `x` is never consumed.
        # Only the residual is copied, so the whole call meets the memory the fold gives it.
        r2 = None if residual is None else ttnn.clone(residual)
        key = (I, self.weights["proj_a.weight"].shape[0], self.weights["proj_b.weight"].shape[0], I)
        saved = dict(T._OPM_DRAM_ROW_CAP)
        T._OPM_DRAM_ROW_CAP.clear()
        T._OPM_DRAM_ROW_CAP[key] = 64
        s0 = dict(T.OPM_ROW_STATS)
        try:
            blocked = call(self, x, msa_mask, n_msa, r2)
        finally:
            T._OPM_DRAM_ROW_CAP.clear()
            T._OPM_DRAM_ROW_CAP.update(saved)
        hb = host(blocked).reshape(I, I, -1)[rows]
        ttnn.deallocate(blocked)
        s1 = dict(T.OPM_ROW_STATS)
        whole = call(self, x, msa_mask, n_msa, residual)
        s2 = dict(T.OPM_ROW_STATS)
        hw = host(whole).reshape(I, I, -1)[rows]
        try:
            ref = reference(self, xs, ms, n_msa, rs, rows)
            graded = {"whole": err(hw, ref), "blocked": err(hb, ref)}
        except Exception as e:  # a reference bug must not cost the fold or the whole-vs-blocked reading
            graded = {"reference_error": repr(e)}
        log(call=k, tokens=I, depth=xs.shape[0], key=key, residual=residual is not None,
            census_blocked={c: s1[c] - s0[c] for c in s0}, census_whole={c: s2[c] - s1[c] for c in s1},
            whole_vs_blocked=err(hw, hb), **graded)
        return whole

    T.OuterProductMean.__call__ = probed


class _Patch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "tt_bio.tenstorrent":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        run = spec.loader.exec_module

        def exec_module(mod):
            run(mod)
            install(mod)
        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Patch())
runpy.run_path(SCRIPT, run_name="__main__")
