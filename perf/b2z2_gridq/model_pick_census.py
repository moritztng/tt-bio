#!/usr/bin/env python3
"""Which q_chunk does every model on the shared path actually ask for, and what would the rule do?

`AttentionPairBias` and the plain attention around it are shared. A lever measured on one model is
a per-model patch wearing a property gate, so this records, for every SDPA call a real run makes,
the (q_len, k_len, work) it arrives with, the chunk the shipped constant returns, the chunk the
grid rule returns, and how many work units each leaves on the grid that is actually open.

It patches the picker in `tenstorrent` AND in every module that imported it by name -- `esmc`,
`esmfold2` and `saprot` bind it at import, so patching only the definition site records nothing
for three of the six call sites.

    python3 model_pick_census.py --model esmc-300m --input examples/prot.fasta --out out/x.json

No timing is taken here. This answers "does the pick move, and where", and the ladder answers
"is the move worth anything".
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

CALLS: Counter = Counter()
TRI: Counter = Counter()


def install(T) -> None:
    import tt_bio.tenstorrent as _T
    orig = T._sdpa_program_config_for_lengths
    orig_tri = T._tri_att_sdpa_program_config

    def wrapped(q_len, k_len, work=0):
        cores = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
        cap = T._capped_sdpa_chunk_size(q_len)
        rule = T._grid_q_chunk(q_len, int(work), cap, cores) if work else cap
        CALLS[(int(q_len), int(k_len), int(work), cap, rule, cores)] += 1
        return orig(q_len, k_len, work)

    def wrapped_tri(q_len, k_len):
        TRI[(int(q_len), int(k_len))] += 1
        return orig_tri(q_len, k_len)

    # `_configure_active_compute_grid` clears these caches at device open, so a wrapper without
    # a `cache_clear` kills the run at the first open rather than censusing it.
    wrapped.cache_clear = getattr(orig, "cache_clear", lambda: None)
    wrapped_tri.cache_clear = getattr(orig_tri, "cache_clear", lambda: None)
    T._sdpa_program_config_for_lengths = wrapped
    T._tri_att_sdpa_program_config = wrapped_tri
    for name in ("tt_bio.esmc", "tt_bio.esmfold2", "tt_bio.saprot"):
        try:
            m = __import__(name, fromlist=["x"])
        except Exception:                                           # noqa: BLE001
            continue
        if hasattr(m, "_sdpa_program_config_for_lengths"):
            m._sdpa_program_config_for_lengths = wrapped


def report(out_path: Path, meta: dict) -> None:
    cores = meta.get("cores") or 0
    rows = []
    for (q, k, w, cap, rule, c), n in sorted(CALLS.items(), key=lambda kv: -kv[1]):
        padded = -(-q // 32) * 32
        su = w * -(-padded // cap) if w else 0
        ru = w * -(-padded // rule) if w else 0
        rows.append({"q_len": q, "k_len": k, "work": w, "calls": n, "cores": c,
                     "shipped_chunk": cap, "rule_chunk": rule, "moves": cap != rule,
                     "shipped_units": su, "rule_units": ru,
                     "shipped_occupancy": round(min(su, c) / c, 3) if su else None,
                     "rule_occupancy": round(min(ru, c) / c, 3) if ru else None})
    res = {"meta": meta, "sdpa_calls": rows,
           "tri_att_calls": [{"q_len": q, "k_len": k, "calls": n}
                             for (q, k), n in sorted(TRI.items(), key=lambda kv: -kv[1])],
           "total_sdpa_calls": sum(CALLS.values()),
           "calls_whose_pick_moves": sum(n for (q, k, w, cap, rule, c), n in CALLS.items()
                                         if cap != rule)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1))
    print(f"\n  {res['total_sdpa_calls']} SDPA calls, "
          f"{res['calls_whose_pick_moves']} of them get a different chunk under the rule")
    for r in rows:
        mark = "MOVES" if r["moves"] else "     "
        print(f"  {mark} q={r['q_len']:5d} k={r['k_len']:5d} work={r['work']:6d} x{r['calls']:5d}  "
              f"{r['shipped_chunk']:4d} -> {r['rule_chunk']:4d}   occ "
              f"{r['shipped_occupancy']} -> {r['rule_occupancy']}")
    print(f"  wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cmd", default="predict")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()

    os.environ.setdefault("TT_BIO_SDPA_GRID_Q_CHUNK", "0")   # census the SHIPPED run
    import tt_bio.tenstorrent as T
    install(T)
    from tt_bio.main import cli as tt_main

    argv = [a.cmd, a.input, "--model", a.model] + [v for v in a.extra if v != "--"]
    t0 = time.time()
    rc = 0
    try:
        sys.argv = ["tt-bio"] + argv
        tt_main()
    except SystemExit as e:                                          # noqa: PERF203
        rc = int(e.code or 0)
    except Exception as exc:                                         # noqa: BLE001
        print(f"  ! run raised {type(exc).__name__}: {str(exc)[:200]}")
        rc = 1
    cores = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
    # The engine's own counter for the branch the lever lives on. A census that reads zero calls
    # cannot tell "the model does not use this picker" from "the hook never ran", and this does.
    stats = {"B2_TOKEN_DIT_SDPA_STATS(served,declined)": list(getattr(T, "B2_TOKEN_DIT_SDPA_STATS", [])),
             "SDPA_K_CHUNK_STATS": list(getattr(T, "SDPA_K_CHUNK_STATS", []))}
    print("  engine counters:", stats)
    report(a.out, {"engine_counters": stats, "model": a.model, "input": a.input, "argv": argv, "rc": rc,
                   "grid": list(T.COMPUTE_GRID_MAIN), "cores": cores,
                   "seconds": round(time.time() - t0, 1),
                   "card": os.environ.get("TT_VISIBLE_DEVICES")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
