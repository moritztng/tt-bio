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
reads those. Four have no counter (ADALN_S_HOIST, PAIR_TRANSPOSE_VIA_ROW_MAJOR,
PAIR_PROJ_MINIMAL_MATMUL, QKV_MM_CONFIG) and are counted by wrapping their helper, marked
`wrap` below.

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
]

HOW = {flag: how for flag, _m, _a, _c, how in LEVERS}

WRAP_KEYS = ("ADALN_S_HOIST", "PAIR_TRANSPOSE_VIA_ROW_MAJOR",
             "PAIR_PROJ_MINIMAL_MATMUL", "QKV_MM_CONFIG",
             "B2_BIAS_SLICE_HOIST", "B2_ADALN_S_MEMO")
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
        return impl(t, memory_config)

    T._pair_transpose_impl = _pair_transpose_impl

    # Both return None when they decline, so a non-None result is a firing.
    for key, fname in (("PAIR_PROJ_MINIMAL_MATMUL", "_pair_proj_minimal_matmul"),
                       ("QKV_MM_CONFIG", "_qkv_mm_config")):
        orig = getattr(T, fname)

        def wrapper(*a, _orig=orig, _key=key, **kw):
            out = _orig(*a, **kw)
            WRAP_COUNTS[_key][0 if out is not None else 1] += 1
            return out

        setattr(T, fname, wrapper)


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
        elif counter:
            cmod, _, cname = counter.rpartition(".")
            cm = sys.modules.get(cmod)
            c = getattr(cm, cname, None) if cm is not None else None
            if isinstance(c, dict):
                served, declined = c.get("calls"), c.get("blocked")
            elif isinstance(c, (list, tuple)) and len(c) >= 2:
                served, declined = c[0], c[1]
        rows[flag] = {"resolved": str(getattr(m, attr, "MISSING")),
                      "served": served, "declined": declined}
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
            json.dump({"pid": os.getpid(), "argv": sys.argv[:4], "rows": rows}, fh)
        os.replace(tmp, path)

    def tick():
        # Polling rather than an import hook: a worker that dies on a signal never runs
        # atexit, so the counts have to already be on disk.
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

    threading.Thread(target=tick, daemon=True).start()
    atexit.register(at_exit)


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
    dumps = sorted(dumpdir.glob("pid*.json"))
    for p in dumps:
        try:
            d = json.loads(p.read_text())
        except Exception:                                                # noqa: BLE001
            continue
        for flag, r in d.get("rows", {}).items():
            a = agg.setdefault(flag, {"resolved": set(), "served": None, "declined": None})
            a["resolved"].add(r["resolved"])
            for k in ("served", "declined"):
                if r.get(k) is not None:
                    a[k] = (a[k] or 0) + r[k]
    rows = []
    for flag, _m, _a, counter, how in LEVERS:
        a = agg.get(flag)
        rows.append({"flag": flag, "how": how, "counter": counter,
                     "resolved": "/".join(sorted(a["resolved"])) if a else "not-imported",
                     "served": a["served"] if a else None,
                     "declined": a["declined"] if a else None})
    return {"label": label, "cli": cli, "rc": rc, "processes": len(dumps), "rows": rows}


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
    sys.exit(rc)


if __name__ == "__main__":
    main()
