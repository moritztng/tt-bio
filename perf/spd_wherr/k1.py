"""Run a fold with every fp32-accumulating matmul at K block 1, where the Wormhole erratum cannot occur.

    python perf/spd_wherr/k1.py --k1-out K1.json [--only REGEX] -- <bench.py args>

Rewrites, before perf/spd/bench.py runs (args after `--`):
* ttnn.linear / ttnn.matmul under fp32 dest acc: a program config with in0_block_w > 1 is cloned at 1; an
  unconfigured call on a 2D weight gets a 2D multicast config at in0_block_w 1 on the device grid.
* ttnn.experimental.minimal_matmul under fp32 dest acc: K_block_size 1 (config None -> the op's default at K 1).
* the generic_op launchers that bake a whole-K block (triatt head-major qkv/tail, fused qkvg, trimul dual-noc,
  the trimul tail) are turned off, so their arithmetic goes through the stock ops above.
`--only` keeps the rewrite to calls whose tt_bio stack matches REGEX (a stage bisect). A rewrite the device
refuses falls back to the original call and is counted. K1.json lists every rewritten, kept and refused site.
"""
import atexit, json, os, re, runpy, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for k in ("TT_BIO_TRIATT_HEAD_MAJOR_QKV", "TT_BIO_TRIATT_HEAD_MAJOR_TAIL", "TT_BIO_TRIATT_FUSED_QKVG",
          "TT_BIO_TRIATT_FUSED_QKVGB", "TT_BIO_TRIMUL_DUAL_NOC"):
    os.environ[k] = "0"

import ttnn  # noqa: E402

FIELDS = ("compute_with_storage_grid_size", "in0_block_w", "out_subblock_h", "out_subblock_w", "out_block_h",
          "out_block_w", "per_core_M", "per_core_N", "transpose_mcast", "fused_activation", "fuse_batch",
          "mcast_in0", "gather_in0", "hop_cores", "num_global_cb_receivers", "untilize_out")


def stack():
    f, out = sys._getframe(2), []
    while f is not None:
        p = f.f_code.co_filename
        if "/tt_bio/" in p:
            out.append(f"{p.rsplit('/tt_bio/', 1)[1]}:{f.f_lineno}:{f.f_code.co_name}")
        f = f.f_back
    return " < ".join(out)


def fp32acc(ck):
    return ck is not None and bool(getattr(ck, "fp32_dest_acc_en", False))


def clone(pc, **over):
    kw = {f: getattr(pc, f) for f in FIELDS if hasattr(pc, f)}
    kw.update(over)
    return type(pc)(**kw)


def auto2d(a, b):
    """2D multicast at in0_block_w 1, out block bounded so the fp32 partials fit L1."""
    ash, bsh = [int(d) for d in a.shape], [int(d) for d in b.shape]   # ttnn.Shape does not slice
    if len(bsh) > 2 and any(d != 1 for d in bsh[:-2]):
        return None
    if a.is_sharded():
        return None
    g = a.device().compute_with_storage_grid_size(); gx, gy = g.x, g.y
    m = 1
    for d in ash[:-1]:
        m *= d
    mt, nt = -(-m // 32), -(-bsh[-1] // 32)
    pm, pn = -(-mt // gy), -(-nt // gx)
    sw = max(s for s in range(1, min(4, pn) + 1) if pn % s == 0)
    sh = max(h for h in range(1, max(1, 4 // sw) + 1) if pm % h == 0)
    bh = max(h for h in range(sh, pm + 1, sh) if pm % h == 0 and (h * pn <= 32 or h == sh))
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=1, out_subblock_h=sh, out_subblock_w=sw,
        out_block_h=bh, out_block_w=pn, per_core_M=pm, per_core_N=pn, transpose_mcast=False,
        fused_activation=None, fuse_batch=True)


class K1:
    def __init__(self, out, only):
        self.out, self.only = out, re.compile(only) if only else None
        self.stats, self.refused = Counter(), {}
        atexit.register(self.dump)

    def dump(self):
        rows = sorted(self.stats.items(), key=lambda kv: -kv[1])
        Path(self.out).write_text(json.dumps({"only": self.only.pattern if self.only else None,
                                              "calls": [[k, v] for k, v in rows],
                                              "refused": self.refused}, indent=1) + "\n")

    def count(self, what, op, st, a, b):
        self.stats[f"{what} {op} {list(a.shape)}x{list(b.shape)} {st.split(' < ')[0]}"] += 1

    def wrap(self, op, fn, getab):
        def w(*args, **kw):
            try:
                return inner(*args, **kw)
            except Exception as e:  # the instrument must never kill the fold
                self.refused.setdefault("instrument " + stack().split(" < ")[0], f"{type(e).__name__}: {e}"[:300])
                return fn(*args, **kw)

        def inner(*args, **kw):
            ck = kw.get("compute_kernel_config")
            if not fp32acc(ck):
                return fn(*args, **kw)
            st = stack()
            a, b = getab(args, kw)
            if self.only is not None and not self.only.search(st):
                self.count("kept(only)", op, st, a, b); return fn(*args, **kw)
            kw2 = dict(kw)
            if op == "minimal_matmul":
                c = kw.get("config")
                g = a.device().compute_with_storage_grid_size()
                kw2["config"] = ttnn.MinimalMatmulConfig(
                    M_block_size=c.M_block_size if c else 8, K_block_size=1, N_block_size=c.N_block_size if c else 8,
                    subblock_h=c.subblock_h if c else 2, subblock_w=c.subblock_w if c else 2,
                    compute_with_storage_grid_size=c.compute_with_storage_grid_size if c else ttnn.CoreCoord(g.x, g.y))
            else:
                pc = kw.get("program_config")
                if pc is not None:
                    if getattr(pc, "in0_block_w", 1) <= 1:
                        self.count("already1", op, st, a, b); return fn(*args, **kw)
                    try:
                        kw2["program_config"] = clone(pc, in0_block_w=1)
                    except Exception as e:
                        self.refused.setdefault(st.split(" < ")[0], f"clone {type(pc).__name__}: {e}"[:200])
                        self.count("refused", op, st, a, b); return fn(*args, **kw)
                else:
                    pc2 = auto2d(a, b)
                    if pc2 is None:
                        self.count("kept(batched/sharded)", op, st, a, b); return fn(*args, **kw)
                    kw2["program_config"] = pc2
                    kw2.pop("core_grid", None)
            key = st.split(" < ")[0] + str(list(a.shape)) + str(list(b.shape))
            if key in self.refused:
                self.count("refused", op, st, a, b); return fn(*args, **kw)
            try:
                y = fn(*args, **kw2)
            except Exception as e:
                self.refused[key] = str(e)[:300]
                self.count("refused", op, st, a, b); return fn(*args, **kw)
            self.count("k1", op, st, a, b)
            return y
        return w

    def install(self):
        mm = lambda args, kw: (args[0] if args else kw["input_tensor_a"], args[1] if len(args) > 1 else kw["input_tensor_b"])
        ttnn.matmul = self.wrap("matmul", ttnn.matmul, mm)
        ttnn.linear = self.wrap("linear", ttnn.linear, mm)
        mmm = lambda args, kw: (args[0] if args else kw["input_tensor"], args[1] if len(args) > 1 else kw["weight_tensor"])
        ttnn.experimental.minimal_matmul = self.wrap("minimal_matmul", ttnn.experimental.minimal_matmul, mmm)
        from tt_bio import trimul_tail
        trimul_tail.fused_tail = lambda *a, **k: None
        trimul_tail.eligible = lambda *a, **k: "k1"


def main():
    i = sys.argv.index("--")
    own, rest = sys.argv[1:i], sys.argv[i + 1:]
    out = own[own.index("--k1-out") + 1]
    only = own[own.index("--only") + 1] if "--only" in own else None
    K1(out, only).install()
    sys.argv = [str(REPO / "perf" / "spd" / "bench.py")] + rest
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
