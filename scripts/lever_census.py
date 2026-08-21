"""Census of the shipped perf levers: what the installed package resolved, and what fired.

A lever that is merged and switched on still delivers nothing if its guard never admits a
real fold, so reading a default proves half the question. This runs a fold through the
installed package and reports, per lever, the value the artifact resolved and how many
times the fast path actually served work.

`tt-bio predict` does not fold in the process you launch it from: it spawns worker
processes (multiprocessing "spawn"). Counters read in the launcher are therefore always
zero - measured 2026-08-17, a real protenix-v2 fold left every one of the 24 levers at
served=0, which reads as "every lever is dark" and is an artifact of where you looked. So
the census runs the CLI as a subprocess with a generated `sitecustomize.py` on PYTHONPATH.
Every process of the fold, launcher and workers alike, dumps its own counters into a
directory, and the census sums them afterwards.

Most levers keep a `*_STATS = [served, declined]` counter next to their guard; the hook
reads those. Seven have no counter and are counted by wrapping their helper, marked `wrap`
below. Every decline also carries the reason its guard refused, so a lever that reads
served=0 can be told apart from one that is correctly declining.

A wrap counts only the calls made after it lands, so it has to be installed before the fold
starts calling, which is why it hangs off an import hook (`_wrap_on_import`). Driving it from
the 3-second dump thread instead made those seven levers, and only those, read `0/0` whenever
the host was busy enough that the thread lost the race: 11446 calls counted on an idle box
against 7456 with three concurrent folds, same fold, same commit.

    # one model, from the installed venv (PYTHONPATH is set by this script - do not add
    # the repo, or tt_bio resolves from the tree instead of the artifact under test)
    python3 scripts/lever_census.py --tt-bio /venv/bin/tt-bio --label ef2-512 \
        --out census_ef512.json -- predict t512.yaml --model esmfold2

    # the table across every model
    python3 scripts/lever_census.py --report census_*.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# flag, module, resolved attribute, counter spec, how
#   counter spec: "MODULE.NAME" for a [served, declined] list, or None
LEVERS = [
    ("SPLIT_SWIGLU", "tt_bio.esmc", "_SPLIT_SWIGLU", "tt_bio.esmc.SPLIT_STATS", "stats"),
    ("SPLIT_SWIGLU_SMALL_GRID", "tt_bio.esmc", "_SPLIT_SWIGLU_SMALL_GRID",
     "tt_bio.esmc.SPLIT_STATS", "stats-shared"),
    ("PAIR_FFN_L1_FC1", "tt_bio.esmc", "_PAIR_FFN_L1_FC1", "tt_bio.esmc.L1_FC1_STATS", "stats"),
    ("PAIR_FFN_L1_LN", "tt_bio.esmc", "_PAIR_FFN_L1_LN", "tt_bio.esmc.L1_LN_STATS", "stats"),
    ("PAIR_FFN_L1_SLICE", "tt_bio.esmc", "_PAIR_FFN_L1_SLICE",
     "tt_bio.esmc.L1_SLICE_STATS", "stats"),
    ("PAIR_FFN_FUSED_RESIDUAL", "tt_bio.esmc", "_PAIR_FFN_FUSED_RESIDUAL",
     "tt_bio.esmc.FUSED_RESID_STATS", "stats"),
    ("PAIR_FFN_FILL_ASSEMBLY", "tt_bio.esmc", "_PAIR_FFN_FILL_ASSEMBLY",
     "tt_bio.esmc.FILL_ASSEMBLY_STATS", "stats"),
    ("TRIMUL_IN_PROJ_DUAL_NOC", "tt_bio.mm_dualnoc", "_ENABLED",
     "tt_bio.mm_dualnoc.STATS", "stats"),
    ("ADALN_S_HOIST", "tt_bio.tenstorrent", "ADALN_S_HOIST", None, "wrap"),
    ("FP32_SOFTMAX_BIAS_HOIST", "tt_bio.tenstorrent", "FP32_SOFTMAX_BIAS_HOIST",
     "tt_bio.tenstorrent.FP32_SOFTMAX_STATS", "stats-dict"),
    ("PAIR_TRANSPOSE_VIA_ROW_MAJOR", "tt_bio.tenstorrent", "_PT_ROW_MAJOR", None, "wrap"),
    ("PAIR_PROJ_MINIMAL_MATMUL", "tt_bio.tenstorrent", "_PAIR_PROJ_MM", None, "wrap"),
    ("TRIMUL_TAIL_F1", "tt_bio.tenstorrent", "_TRIMUL_TAIL_F1",
     "tt_bio.trimul_tail.STATS", "stats"),
    ("QKV_MM_CONFIG", "tt_bio.tenstorrent", "_MM_CFG", None, "wrap"),
    ("DEVICE_LM_HANDOFF", "tt_bio.esmfold2_runtime", "_DEVICE_LM_HANDOFF",
     "tt_bio.esmfold2_runtime.LM_HANDOFF_STATS", "stats"),
    ("REBLOCK_PERMUTE", "tt_bio.reblock_permute", "_ENABLED",
     "tt_bio.reblock_permute.STATS", "stats"),
    ("REBLOCK_PERMUTE_BACK", "tt_bio.reblock_permute", "_ENABLED_BACK",
     "tt_bio.reblock_permute.STATS_BACK", "stats"),
    ("REBLOCK_PERMUTE_GATED", "tt_bio.reblock_permute", "_ENABLED_GATED",
     "tt_bio.reblock_permute.STATS_GATED", "stats"),
    ("TRIATT_PERSISTENT_MASK", "tt_bio.triatt_sdpa", "_ENABLED",
     "tt_bio.triatt_sdpa.STATS", "stats"),
    ("SDPA_WIDE_K", "tt_bio.tenstorrent", "SDPA_WIDE_K",
     "tt_bio.tenstorrent.SDPA_K_CHUNK_STATS", "stats"),
    ("RFD3_SPARSE_BIAS", "tt_bio.rfd3_bias", "_ENABLED", "tt_bio.rfd3_bias.STATS", "stats"),
    ("RFD3_FUSED_SCORES", "tt_bio.rfd3_bias", "_FUSED_ENABLED",
     "tt_bio.rfd3_bias.FSTATS", "stats"),
    ("TRIATT_HEAD_MAJOR_QKV", "tt_bio.triatt_qkv", "_ENABLED",
     "tt_bio.triatt_qkv.STATS", "stats"),
    ("TRIATT_HEAD_MAJOR_TAIL", "tt_bio.triatt_qkv", "_TAIL_ENABLED",
     "tt_bio.triatt_qkv.TAIL_STATS", "stats"),
    ("TRIATT_TAIL_OVER_L1", "tt_bio.triatt_qkv", "_TAIL_OVER_L1",
     "tt_bio.triatt_qkv.TAIL_STATS", "stats-shared"),
    # Boltz-2 diffusion, landed 6c07446f. L7 and L6 default ON, S6 default OFF (not bit-exact).
    # L7 fires once per bias per fold by design, so a served count of 2-3 is the whole win --
    # what it replaces is 6000 per-step slices, which the counter cannot see.
    ("B2_BIAS_SLICE_HOIST", "tt_bio.tenstorrent", "_B2_BIAS_SLICE_HOIST", None, "wrap"),
    ("B2_ADALN_S_MEMO", "tt_bio.tenstorrent", "_B2_ADALN_S_MEMO", None, "wrap"),
    ("B2_TOKEN_DIT_SDPA", "tt_bio.tenstorrent", "_B2_TOKEN_DIT_SDPA", None, "off-by-design"),
    # The two size-conditioned L1 gates that `tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa`
    # root-caused alongside K2. K2 has a counter of its own (TRIATT_PERSISTENT_MASK); these two
    # had none, so a census built on the table above could see one third of that defect.
    ("TRANSPOSE_L1_RESIDENT", "tt_bio.tenstorrent", "_TRANSPOSE_L1_HEADROOM", None, "wrap"),
    ("SDPA_Q_CHUNK_FITS", "tt_bio.tenstorrent", "_SDPA_WIDE_Q",
     "tt_bio.tenstorrent._SDPA_Q_CHUNK_OVER_L1", "setlen"),
    # The third gate of the same family, added 2026-08-20. The pair projections' L1-destination
    # leg refuses through a bare except that memoises the operand class, so a fold whose trimul
    # out-projection does not fit runs the whole rest of the process on the DRAM leg with no
    # counter and no log line. Found in Nesso-1 at 576 padded tokens on a 13x10 grid; the code
    # is shared, so every model on the pair track can hit it.
    ("PAIR_PROJ_L1_OUT", "tt_bio.tenstorrent", "_PAIR_PROJ_L1_OUT",
     "tt_bio.tenstorrent.PAIR_PROJ_L1_OUT_STATS", "stats"),
]

HOW = {flag: how for flag, _m, _a, _c, how in LEVERS}

# Six modules already keep a `(reason, shape) -> count` reject dict, and every one of them built it
# on purpose: `trimul_tail.eligible`'s docstring says "every clause is a real assumption of the fork,
# so a decline names which one". The census read the served/declined counts and threw the reason
# away, so "why is this lever dark at this size" had to be argued from source instead of read off
# the counter -- the same defect as the two size-conditioned L1 gates that had no counter at all.
# The reason is what makes a size-ladder exemption entry evidence rather than an opinion, so record
# it. Aggregated by reason with the shape dropped: the shape is per-call noise, the clause is the
# finding, and a gate baseline has to stay diffable.
REJECTS_ATTR = {
    "TRIMUL_IN_PROJ_DUAL_NOC": "tt_bio.mm_dualnoc.REJECTS",
    "TRIMUL_TAIL_F1": "tt_bio.trimul_tail.REJECTS",
    # Only the ungated lever. reblock_permute.eligible and eligible_gated share one
    # REJECTS dict, so attributing it to both reported the ungated lever's clause against
    # a gated lever that was never even offered (0 served / 0 declined) -- a measured
    # clause is worth nothing if it is attributed to the wrong guard.
    "REBLOCK_PERMUTE": "tt_bio.reblock_permute.REJECTS",
    "TRIATT_PERSISTENT_MASK": "tt_bio.triatt_sdpa.REJECTS",
    "TRIATT_HEAD_MAJOR_QKV": "tt_bio.triatt_qkv.REJECTS",
    "TRIATT_HEAD_MAJOR_TAIL": "tt_bio.triatt_qkv.TAIL_REJECTS",
    "RFD3_SPARSE_BIAS": "tt_bio.rfd3_bias.REJECTS",
    "RFD3_FUSED_SCORES": "tt_bio.rfd3_bias.REJECTS",
    "PAIR_PROJ_L1_OUT": "tt_bio.tenstorrent.PAIR_PROJ_L1_OUT_REJECTS",
}


def _reject_reasons(flag):
    """{reason: count} for one lever, or None when its module keeps no reject dict.

    Keys are `(reason, shape)` in all six modules; the shape is dropped so the result is stable
    across folds of the same model at the same size and small enough to live in a gate baseline.
    """
    attr = REJECTS_ATTR.get(flag)
    if not attr:
        return None
    mod, _, name = attr.rpartition(".")
    m = sys.modules.get(mod)
    d = getattr(m, name, None) if m is not None else None
    if not isinstance(d, dict):
        return None
    out = {}
    for k, v in d.items():
        reason = k[0] if isinstance(k, tuple) and k else str(k)
        out[str(reason)] = out.get(str(reason), 0) + v
    return out or None


# Why a wrap-counted lever declined, keyed flag -> {reason: count}. A guard that never fires
# has to say which of its terms refused, or "dark" cannot be told from "correctly declined":
# PAIR_TRANSPOSE_VIA_ROW_MAJOR counts an L1 destination as a decline even though L1 is the
# strictly faster route, so the bare counter reads as a defect and is not one. These are the
# wrap-counted levers, which keep no REJECTS dict of their own for REJECTS_ATTR to point at.
WRAP_REJECTS: dict = {}


def _wreject(flag, reason):
    d = WRAP_REJECTS.setdefault(flag, {})
    d[reason] = d.get(reason, 0) + 1


WRAP_KEYS = ("ADALN_S_HOIST", "PAIR_TRANSPOSE_VIA_ROW_MAJOR",
             "PAIR_PROJ_MINIMAL_MATMUL", "QKV_MM_CONFIG",
             "B2_BIAS_SLICE_HOIST", "B2_ADALN_S_MEMO", "TRANSPOSE_L1_RESIDENT")
WRAP_COUNTS = {k: [0, 0] for k in WRAP_KEYS}


# ----------------------------------------------------------------- child-side hook

def _install_wraps():
    """Counters for the four levers whose guard keeps no `*_STATS` of its own.

    Called repeatedly from the hook thread: it is a no-op until `tt_bio.tenstorrent` and
    `ttnn` are imported, and marks the module so a second call cannot double-wrap.
    """
    T = sys.modules.get("tt_bio.tenstorrent")
    ttnn = sys.modules.get("ttnn")
    if T is None or ttnn is None or getattr(T, "_census_wrapped", False):
        return
    # A module is in sys.modules from the moment its execution STARTS, so being importable is
    # not the same as being ready. Every name this function rebinds has to exist before it
    # claims the flag: half-installed wraps cannot be retried without double-counting, and
    # claiming the flag then failing would silently zero all seven counters for the process.
    if not all(hasattr(T, a) for a in
               ("AdaLN", "DiffusionModule", "_pair_transpose_impl",
                "_transpose_memory_config", "_pair_proj_minimal_matmul", "_qkv_mm_config")):
        return
    T._census_wrapped = True

    # The hoist is the only caller of AdaLN.s_terms outside the rollout, so a call means
    # the conditioning half was precomputed rather than recomputed per step.
    inner = T.AdaLN.s_terms

    def s_terms(self, *a, **kw):
        WRAP_COUNTS["ADALN_S_HOIST"][0 if T.ADALN_S_HOIST else 1] += 1
        out = inner(self, *a, **kw)
        # L6 shares this function. The memo returns its stored tuple BY IDENTITY on both the
        # storing call and every hit, and a fresh tuple otherwise, so identity separates them.
        if T._B2_ADALN_S_MEMO and getattr(self, "atom_level", False):
            hit = out is getattr(self, "_s_memo", None)
            WRAP_COUNTS["B2_ADALN_S_MEMO"][0 if hit else 1] += 1
        return out

    T.AdaLN.s_terms = s_terms

    # L7 returns a list of per-layer head-ranges when it fires and its input tensor when it
    # declines, so the return type is the verdict.
    hoist = T.DiffusionModule._hoist_layer_bias

    def _hoist_layer_bias(self, bias, transformer):
        out = hoist(self, bias, transformer)
        WRAP_COUNTS["B2_BIAS_SLICE_HOIST"][0 if isinstance(out, list) else 1] += 1
        return out

    T.DiffusionModule._hoist_layer_bias = _hoist_layer_bias

    # `_pair_transpose_impl` takes the row-major route under exactly this predicate.
    impl = T._pair_transpose_impl

    def _pair_transpose_impl(t, memory_config):
        rm = (T._PT_ROW_MAJOR and len(t.shape) == 3
              and memory_config.buffer_type == ttnn.BufferType.DRAM
              and t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT)
        WRAP_COUNTS["PAIR_TRANSPOSE_VIA_ROW_MAJOR"][0 if rm else 1] += 1
        if not rm:
            if not T._PT_ROW_MAJOR:
                why = "flag_off"
            elif len(t.shape) != 3:
                why = "rank=%d" % len(t.shape)
            elif memory_config.buffer_type != ttnn.BufferType.DRAM:
                # NOT a defect: the L1 destination is 2.47-2.74x against this route's 1.60x.
                why = "l1_dest_is_faster"
            elif t.dtype != ttnn.bfloat16:
                why = "dtype"
            else:
                why = "layout"
            _wreject("PAIR_TRANSPOSE_VIA_ROW_MAJOR",
                     why + ":" + "x".join(str(int(d)) for d in t.padded_shape))
        return impl(t, memory_config)

    T._pair_transpose_impl = _pair_transpose_impl

    # `_transpose_memory_config` returns L1_MEMORY_CONFIG or DRAM_MEMORY_CONFIG and nothing else, so
    # the buffer type IS the verdict on `_TRANSPOSE_L1_HEADROOM`. This is the gate that stopped
    # answering L1 at N>=560 with no error and no log line.
    tmc = T._transpose_memory_config

    # *a/**kw, not the real signature. A counting wrapper has no business knowing how many
    # arguments the function it counts takes, and hardcoding them broke this arm completely:
    # 421eee0c ("perf(rf3): lever 8") gave `_transpose_memory_config` a `reserve_per_core`
    # second parameter and a call site that passes it, this wrapper still took one, and every
    # ending-variant triangle attention raised TypeError. That call site is unconditional, so
    # the size-generality arm was dead for EVERY model, and its own baseline could not be
    # re-recorded to notice.
    def _transpose_memory_config(*a, **kw):
        out = tmc(*a, **kw)
        WRAP_COUNTS["TRANSPOSE_L1_RESIDENT"][0 if out.buffer_type == ttnn.BufferType.L1 else 1] += 1
        return out

    T._transpose_memory_config = _transpose_memory_config

    # Both return None when they decline, so a non-None result is a firing.
    for key, fname in (("PAIR_PROJ_MINIMAL_MATMUL", "_pair_proj_minimal_matmul"),
                       ("QKV_MM_CONFIG", "_qkv_mm_config")):
        orig = getattr(T, fname)

        def wrapper(*a, _orig=orig, _key=key, **kw):
            out = _orig(*a, **kw)
            WRAP_COUNTS[_key][0 if out is not None else 1] += 1
            if out is None:
                try:
                    _wreject(_key, _pp_reason(T, _key, *a))
                except Exception:                                        # noqa: BLE001
                    _wreject(_key, "unknown")
            return out

        setattr(T, fname, wrapper)


def _compute_grid():
    """The main compute grid this process opened, as "13x10", or None before device open.

    Recorded because a lever's fired/dark verdict is NOT machine-independent: the same board
    type can present a different grid after harvesting, and a guard sized against the grid
    flips with it (protenix-v2's K2 is admitted on 11x10 and refused on 13x10). A census
    compared across grids is a false alarm waiting to happen.
    """
    m = sys.modules.get("tt_bio.tenstorrent")
    g = getattr(m, "COMPUTE_GRID_MAIN", None) if m is not None else None
    try:
        return f"{int(g[0])}x{int(g[1])}" if g else None
    except Exception:                                                    # noqa: BLE001
        return None


def _pp_reason(T, key, x, w, *rest):
    """Which term of the pair-proj / qkv guard refused, as `reason:(kt,nt,mt)`."""
    import ttnn
    kt = (int(w.shape[-2]) + 31) // 32
    nt = (int(w.shape[-1]) + 31) // 32
    tag = "(%d,%d)" % (kt, nt)
    if key == "PAIR_PROJ_MINIMAL_MATMUL" and not T._PAIR_PROJ_MM:
        return "flag_off"
    if key == "QKV_MM_CONFIG" and not T._MM_CFG:
        return "flag_off"
    if x.dtype != ttnn.bfloat16 or w.dtype != ttnn.bfloat16:
        return "dtype:" + tag
    if len(w.shape) != 2:
        return "weight_rank:" + tag
    # `_pair_proj_minimal_matmul` is scoped to a single K block, kt == 8.
    if key == "PAIR_PROJ_MINIMAL_MATMUL" and kt != 8:
        return "k_tiles=%d:%s" % (kt, tag)
    blk = T._mm_block_for(w)
    if blk is None:
        return "no_mm_block:" + tag
    mt = 1
    for d in [int(d) for d in x.shape][:-1]:
        mt *= d
    mt = (mt + 31) // 32
    M, K, N, _sh, _sw = blk
    if blk is not T._MM_DEFAULT and (kt % K or mt % M or nt % N):
        return "block_divisibility:(%d,%d,%d)" % (kt, nt, mt)
    return "op_threw:" + tag


def _snapshot_process():
    """This process's view: only levers whose module it actually imported."""
    rows = {}
    for flag, mod, attr, counter, how in LEVERS:
        m = sys.modules.get(mod)
        if m is None:
            continue
        served = declined = None
        if how == "wrap":
            served, declined = WRAP_COUNTS.get(flag, [None, None])
        elif how == "setlen":
            # A set of shapes that overflowed their circular-buffer budget and fell back. It only
            # grows, and every member is a fold silently taking the slow path, so its size is the
            # decline count; there is no served counterpart to read.
            cmod, _, cname = counter.rpartition(".")
            cm = sys.modules.get(cmod)
            c = getattr(cm, cname, None) if cm is not None else None
            served, declined = (0, len(c)) if c is not None else (None, None)
        elif counter:
            cmod, _, cname = counter.rpartition(".")
            cm = sys.modules.get(cmod)
            c = getattr(cm, cname, None) if cm is not None else None
            if isinstance(c, dict):
                served, declined = c.get("calls"), c.get("blocked")
            elif isinstance(c, (list, tuple)) and len(c) >= 2:
                served, declined = c[0], c[1]
        # A module that keeps a `_reject()` records its reasons in a module-level REJECTS
        # dict, read via REJECTS_ATTR (_reject_reasons); wrap-counted levers have no dict
        # of their own and use WRAP_REJECTS instead (populated by the wrappers above via
        # _wreject). Either way the reason is already on hand and only the emit was
        # missing. NOT a generic "read REJECTS off the counter's module" lookup: reblock_
        # permute.eligible and eligible_gated share one REJECTS dict, so that would
        # misattribute the ungated lever's clause to the gated lever it was never even
        # offered to (see REJECTS_ATTR's comment on REBLOCK_PERMUTE).
        rej = dict(WRAP_REJECTS.get(flag, {}))
        for reason, n in (_reject_reasons(flag) or {}).items():
            rej[reason] = rej.get(reason, 0) + n
        rows[flag] = {"resolved": str(getattr(m, attr, "MISSING")),
                      "served": served, "declined": declined,
                      "rejects": rej or None}
    return rows


def install_child_hook():
    """Entry point for the generated `sitecustomize.py`. Runs in every process."""
    outdir = os.environ.get("LEVER_CENSUS_DIR")
    if not outdir:
        return
    import atexit
    import threading
    import time

    path = os.path.join(outdir, f"pid{os.getpid()}.json")

    def dump():
        rows = _snapshot_process()
        if not rows:
            return
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            json.dump({"pid": os.getpid(), "argv": sys.argv[:4], "rows": rows,
                       "grid": _compute_grid()}, fh)
        os.replace(tmp, path)

    def tick():
        # Polling rather than an import hook: a worker that dies on a signal never runs
        # atexit, so the counts have to already be on disk. That reasoning covers the DUMP.
        # The WRAPS are installed by _wrap_on_import below; this call is only the backstop for
        # a process where the import hook was replaced by something else.
        while True:
            time.sleep(3)
            try:
                _install_wraps()
                dump()
            except Exception:                                            # noqa: BLE001
                pass

    def at_exit():
        try:
            dump()
        except Exception:                                                # noqa: BLE001
            pass

    _wrap_on_import()
    threading.Thread(target=tick, daemon=True).start()
    atexit.register(at_exit)


def _wrap_on_import():
    """Install the wrap counters as soon as their modules are ready, off an import hook.

    Seven levers keep no `*_STATS` of their own and are counted by monkeypatching a helper
    (`how="wrap"`): ADALN_S_HOIST, QKV_MM_CONFIG, TRANSPOSE_L1_RESIDENT, B2_ADALN_S_MEMO,
    B2_BIAS_SLICE_HOIST, PAIR_PROJ_MINIMAL_MATMUL, PAIR_TRANSPOSE_VIA_ROW_MAJOR. A
    monkeypatch counts only the calls made after it lands, and `_install_wraps` used to be
    reached only from the 3-second dump thread, so every call between `tt_bio.tenstorrent`
    becoming importable and the next tick was invisible. Whether the thread won that race
    depended on how busy the host was.

    So those seven levers, and only those, read `0/0` on a loaded box: the `*_STATS` levers
    count inside the shipped code and cannot be raced. Measured on the boltz2-affinity fold at
    256 aa, 11446 calls counted with the box idle against 7456 with three concurrent folds,
    the 3990-call gap being exactly that set while six of the seven are served on the apo
    fold. A census that changes with the load on the machine, reported as a lever going dark,
    and nothing in the artifact tells the two apart.

    `_install_wraps` is a no-op until both `tt_bio.tenstorrent` and `ttnn` are in
    `sys.modules` and the former is fully executed, so the cheapest correct trigger is "after
    any import completes". The hook removes itself once the wraps are in, so it does not
    outlive the import phase.
    """
    import builtins

    real_import = builtins.__import__
    if getattr(real_import, "_census_import_hook", False):
        return

    def hooked(*a, **kw):
        mod = real_import(*a, **kw)
        try:
            _install_wraps()
            T = sys.modules.get("tt_bio.tenstorrent")
            if T is not None and getattr(T, "_census_wrapped", False):
                builtins.__import__ = real_import      # done; stop paying for the hook
        except Exception:                                                # noqa: BLE001
            pass
        return mod

    hooked._census_import_hook = True
    builtins.__import__ = hooked


# ----------------------------------------------------------------- parent side

def _write_hookdir(hookdir: Path) -> None:
    hookdir.mkdir(parents=True, exist_ok=True)
    scripts_dir = str(Path(__file__).resolve().parent)
    # Appended, not prepended: nothing in scripts/ may shadow a stdlib or site-packages
    # module for the process under test.
    (hookdir / "sitecustomize.py").write_text(
        "import sys\n"
        f"sys.path.append({scripts_dir!r})\n"
        "try:\n"
        "    from lever_census import install_child_hook\n"
        "    install_child_hook()\n"
        "except Exception:\n"
        "    pass\n")


def collect(dumpdir: Path, label: str, cli: list, rc: int) -> dict:
    agg = {}
    grids = set()
    dumps = sorted(dumpdir.glob("pid*.json"))
    for p in dumps:
        try:
            d = json.loads(p.read_text())
        except Exception:                                                # noqa: BLE001
            continue
        if d.get("grid"):
            grids.add(d["grid"])
        for flag, r in d.get("rows", {}).items():
            a = agg.setdefault(flag, {"resolved": set(), "served": None, "declined": None,
                                      "rejects": {}})
            a["resolved"].add(r["resolved"])
            for k in ("served", "declined"):
                if r.get(k) is not None:
                    a[k] = (a[k] or 0) + r[k]
            for reason, n in (r.get("rejects") or {}).items():
                a["rejects"][reason] = a["rejects"].get(reason, 0) + n
    rows = []
    for flag, _m, _a, counter, how in LEVERS:
        a = agg.get(flag)
        rows.append({"flag": flag, "how": how, "counter": counter,
                     "resolved": "/".join(sorted(a["resolved"])) if a else "not-imported",
                     "served": a["served"] if a else None,
                     "declined": a["declined"] if a else None,
                     "rejects": (a["rejects"] or None) if a else None})
    return {"label": label, "cli": cli, "rc": rc, "processes": len(dumps), "rows": rows,
            "grid": "/".join(sorted(grids)) or None}


def report(paths):
    merged = {}
    labels = []
    for p in paths:
        d = json.load(open(p))
        labels.append(d["label"])
        for r in d["rows"]:
            m = merged.setdefault(r["flag"], {"resolved": set(), "how": r["how"], "by": {}})
            if r["resolved"] != "not-imported":
                m["resolved"].add(r["resolved"])
            m["by"][d["label"]] = r["served"]
    width = max(len(f) for f in merged)
    print(f"{'lever'.ljust(width)}  resolved  how           " +
          "  ".join(l.ljust(12) for l in labels))
    dark = []
    for flag, m in merged.items():
        res = "/".join(sorted(m["resolved"])) or "?"
        cells = "  ".join(str(m["by"].get(l, "-")).ljust(12) for l in labels)
        print(f"{flag.ljust(width)}  {res.ljust(8)}  {m['how'].ljust(13)} {cells}")
        if res == "True" and not any((m["by"].get(l) or 0) > 0 for l in labels):
            dark.append(flag)
    print()
    print("ON but served 0 everywhere:", ", ".join(dark) if dark else "none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tt-bio", help="path to the installed `tt-bio` entry point")
    ap.add_argument("--pythonpath", help="prepend this to the child PYTHONPATH, after the hook "
                                         "dir -- the only way to census a WORKTREE rather than an "
                                         "installed artifact (the venv still supplies ttnn)")
    ap.add_argument("--out")
    ap.add_argument("--label", default="run")
    ap.add_argument("--report", nargs="+")
    ap.add_argument("cli", nargs="*")
    args = ap.parse_args()

    if args.report:
        report(args.report)
        return

    if not args.tt_bio or not args.cli:
        ap.error("give --tt-bio and a CLI after `--`, or --report")

    work = Path(args.out or "census").resolve().parent / f".census-{args.label}"
    hookdir, dumpdir = work / "hook", work / "dumps"
    _write_hookdir(hookdir)
    dumpdir.mkdir(parents=True, exist_ok=True)
    for stale in dumpdir.glob("pid*.json"):
        stale.unlink()

    env = dict(os.environ)
    # The hook dir must come first or `sitecustomize` resolves to something else. Anything the
    # caller adds goes after it and before site-packages, which is what makes `--pythonpath <wt>`
    # census the tree under test while ttnn still comes from the venv the entry point belongs to.
    env["PYTHONPATH"] = os.pathsep.join([str(hookdir)] +
                                        ([args.pythonpath] if args.pythonpath else []))
    env["LEVER_CENSUS_DIR"] = str(dumpdir)
    rc = subprocess.call([args.tt_bio, *args.cli], env=env)

    snap = collect(dumpdir, args.label, args.cli, rc)
    if args.out:
        json.dump(snap, open(args.out, "w"), indent=2)
    print(f"--- {args.label}: {snap['processes']} processes, cli rc={rc}")
    for r in snap["rows"]:
        print(f"{r['flag']:32s} {r['resolved']:14s} served={r['served']} "
              f"declined={r['declined']}")
        for why, n in sorted((r.get("rejects") or {}).items(), key=lambda kv: -kv[1])[:6]:
            print(f"{'':34s}  why {why} x{n}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
