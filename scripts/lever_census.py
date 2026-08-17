"""Census of the shipped perf levers: what the installed package resolved, and what fired.

A lever that is merged and switched on still delivers nothing if its guard never admits a
real fold, so reading a default proves half the question. This runs a fold through the
installed package and reports, per lever, the value the artifact resolved and how many
times the fast path actually served work.

Most levers keep their own `*_STATS = [served, declined]` counter next to the guard; this
reads those. Four have no counter (`ADALN_S_HOIST`, `PAIR_TRANSPOSE_VIA_ROW_MAJOR`,
`PAIR_PROJ_MINIMAL_MATMUL`, `QKV_MM_CONFIG`) and are counted by wrapping their helper here,
marked `wrap` in the `how` column.

    python3 scripts/lever_census.py --out census_esmfold2.json -- predict foo.yaml --model esmfold2
    python3 scripts/lever_census.py --report census_*.json
"""

import argparse
import importlib
import json
import sys

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
]

WRAP_COUNTS = {}


def _resolve(dotted):
    mod, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(mod), name, None)


def _install_wraps():
    """Counters for the four levers whose guard keeps no `*_STATS` of its own."""
    import ttnn

    T = importlib.import_module("tt_bio.tenstorrent")

    for key in ("ADALN_S_HOIST", "PAIR_TRANSPOSE_VIA_ROW_MAJOR",
                "PAIR_PROJ_MINIMAL_MATMUL", "QKV_MM_CONFIG"):
        WRAP_COUNTS[key] = [0, 0]

    # 9: the hoist is the only caller of AdaLN.s_terms outside the rollout, so a call means
    # the conditioning half was precomputed rather than recomputed per step.
    inner = T.AdaLN.s_terms

    def s_terms(self, *a, **kw):
        WRAP_COUNTS["ADALN_S_HOIST"][0 if T.ADALN_S_HOIST else 1] += 1
        return inner(self, *a, **kw)

    T.AdaLN.s_terms = s_terms

    # 11: `_pair_transpose_impl` takes the row-major route under exactly this predicate.
    impl = T._pair_transpose_impl

    def _pair_transpose_impl(t, memory_config):
        rm = (T._PT_ROW_MAJOR and len(t.shape) == 3
              and memory_config.buffer_type == ttnn.BufferType.DRAM
              and t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT)
        WRAP_COUNTS["PAIR_TRANSPOSE_VIA_ROW_MAJOR"][0 if rm else 1] += 1
        return impl(t, memory_config)

    T._pair_transpose_impl = _pair_transpose_impl

    # 12 and 14 both return None when they decline, so a non-None result is a firing.
    for key, fname in (("PAIR_PROJ_MINIMAL_MATMUL", "_pair_proj_minimal_matmul"),
                       ("QKV_MM_CONFIG", "_qkv_mm_config")):
        orig = getattr(T, fname)

        def wrapper(*a, _orig=orig, _key=key, **kw):
            out = _orig(*a, **kw)
            WRAP_COUNTS[_key][0 if out is not None else 1] += 1
            return out

        setattr(T, fname, wrapper)


def snapshot(label):
    rows = []
    for flag, mod, attr, counter, how in LEVERS:
        try:
            resolved = getattr(importlib.import_module(mod), attr, "MISSING")
        except Exception as exc:                                            # noqa: BLE001
            resolved = f"IMPORT-ERROR {exc}"
        served = declined = None
        if how == "wrap":
            served, declined = WRAP_COUNTS.get(flag, [None, None])
        elif counter:
            c = _resolve(counter)
            if isinstance(c, dict):
                served, declined = c.get("calls"), c.get("blocked")
            elif isinstance(c, list) and len(c) >= 2:
                served, declined = c[0], c[1]
        rows.append({"flag": flag, "resolved": str(resolved), "served": served,
                     "declined": declined, "how": how, "counter": counter})
    return {"label": label, "rows": rows}


def report(paths):
    merged = {}
    labels = []
    for p in paths:
        d = json.load(open(p))
        labels.append(d["label"])
        for r in d["rows"]:
            m = merged.setdefault(r["flag"], {"resolved": set(), "how": r["how"], "by": {}})
            m["resolved"].add(r["resolved"])
            m["by"][d["label"]] = r["served"]
    width = max(len(f) for f in merged)
    print(f"{'lever'.ljust(width)}  resolved  how          " +
          "  ".join(l.ljust(12) for l in labels))
    dark = []
    for flag, m in merged.items():
        res = "/".join(sorted(m["resolved"]))
        cells = "  ".join(str(m["by"].get(l, "-")).ljust(12) for l in labels)
        print(f"{flag.ljust(width)}  {res.ljust(8)}  {m['how'].ljust(12)} {cells}")
        if res == "True" and not any((m["by"].get(l) or 0) > 0 for l in labels):
            dark.append(flag)
    print()
    print("ON but served 0 everywhere:", ", ".join(dark) if dark else "none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--label", default="run")
    ap.add_argument("--report", nargs="+")
    ap.add_argument("cli", nargs="*")
    args = ap.parse_args()

    if args.report:
        report(args.report)
        return

    _install_wraps()

    rc = 0
    if args.cli:
        from tt_bio.main import cli

        sys.argv = ["tt-bio", *args.cli]
        try:
            cli.main(args=args.cli, standalone_mode=False)
        except SystemExit as exc:
            rc = exc.code or 0
    snap = snapshot(args.label)
    snap["cli"] = args.cli
    snap["rc"] = rc
    if args.out:
        json.dump(snap, open(args.out, "w"), indent=2)
    for r in snap["rows"]:
        print(f"{r['flag']:32s} {r['resolved']:8s} served={r['served']} declined={r['declined']}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
