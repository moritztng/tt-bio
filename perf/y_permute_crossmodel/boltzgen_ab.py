#!/usr/bin/env python3
"""BoltzGen leg of the census/A-B. Design, not fold, so it cannot use `build_fold`.

`tt-bio design --model boltzgen` runs the BoltzGen CLI IN-PROCESS (`tt_bio.main._run_boltzgen_cli`
-> `tt_bio.boltzgen.cli.boltzgen.main`), so with a single `--devices` id the census hook installed
here sees the real calls. The check that it did: `channel_move_calls_per_design` must be non-zero.
A zero there means the hook never ran, which is NOT the same as "no eligible calls" and is reported
as untested rather than unaffected.

    TT_VISIBLE_DEVICES=2 python3 perf/y_permute_crossmodel/boltzgen_ab.py --rounds 2
"""
from __future__ import annotations

import argparse, json, os, shutil, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = Path(__file__).resolve().parent


def med(v):
    return sorted(v)[len(v) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="examples/binder.yaml")
    ap.add_argument("--protocol", default="protein-anything")
    ap.add_argument("--num-designs", type=int, default=1)
    ap.add_argument("--aa-rounds", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default=str(OUT / "ab_boltzgen_binder.json"))
    a = ap.parse_args()

    from tt_bio import reblock_permute as RP
    import tt_bio.tenstorrent as T

    SEEN: dict = {}
    _orig_elig = RP.eligible

    def _elig(x, mc):
        v = _orig_elig(x, mc)
        key = (int(x.shape[1]), int(x.shape[3]),
               str(x.memory_config().buffer_type).rsplit(".", 1)[-1],
               str(mc.buffer_type).rsplit(".", 1)[-1],
               str(x.dtype).rsplit(".", 1)[-1], str(x.layout).rsplit(".", 1)[-1], bool(v))
        SEEN[key] = SEEN.get(key, 0) + 1
        return v

    RP.eligible = _elig

    TM_CALLS: dict = {}
    _orig_tm = T.TriangleMultiplication.__call__

    def _count_tm(self, x, mask=None):
        s = tuple(int(d) for d in x.shape)
        TM_CALLS[s] = TM_CALLS.get(s, 0) + 1
        return _orig_tm(self, x, mask)

    T.TriangleMultiplication.__call__ = _count_tm

    workdir = Path.home() / "ypx_boltzgen_out"

    def one_design():
        if workdir.exists():
            shutil.rmtree(workdir)
        argv = ["run", str(REPO / a.spec), "--output", str(workdir),
                "--num_designs", str(a.num_designs), "--protocol", a.protocol,
                "--device_ids", os.environ.get("TT_VISIBLE_DEVICES", "0")]
        from tt_bio.main import _run_boltzgen_cli
        t0 = time.perf_counter()
        try:
            _run_boltzgen_cli("tt-bio design", argv)
        except SystemExit as e:
            if e.code not in (None, 0):
                raise
        return time.perf_counter() - t0

    import importlib.metadata as md
    R = {"wheel": md.version("ttnn"), "host": "qb2", "model": "boltzgen",
         "card": os.environ.get("TT_VISIBLE_DEVICES"), "spec": a.spec,
         "protocol": a.protocol, "num_designs": a.num_designs, "aa": [], "rounds": []}

    # cold, then the census on a cold ON design
    RP.set_enabled(False)
    c0 = one_design()
    SEEN.clear(); TM_CALLS.clear(); RP.STATS[0] = RP.STATS[1] = 0; RP.REJECTS.clear()
    RP.set_enabled(True)
    c1 = one_design()
    RP.set_enabled(False)
    R["cold"] = {"base_s": round(c0, 3), "wire_s": round(c1, 3)}
    R["census"] = {
        "channel_move_calls_per_design": sum(SEEN.values()),
        "eligible_served_per_design": RP.STATS[0],
        "refused_per_design": RP.STATS[1],
        "by_shape": [{"N": k[0], "C": k[1], "in": k[2], "out": k[3], "dtype": k[4],
                      "layout": k[5], "eligible": k[6], "calls": v}
                     for k, v in sorted(SEEN.items())],
        "reject_reasons": {f"{k[0]}:{list(k[1])}": v for k, v in RP.REJECTS.items()},
        "trimul_invocations_per_design": {"x".join(str(d) for d in k): v
                                          for k, v in sorted(TM_CALLS.items())},
        "hook_saw_calls": sum(SEEN.values()) > 0,
    }
    print("cold:", json.dumps(R["cold"]), flush=True)
    print("census:", json.dumps(R["census"], indent=1), flush=True)
    Path(a.out).write_text(json.dumps(R, indent=1))

    if R["census"]["eligible_served_per_design"] == 0:
        R["verdict_shortcut"] = (
            "zero eligible calls on this design spec: the flip cannot change boltzgen here"
            if R["census"]["hook_saw_calls"] else
            "the census hook saw NO channel moves at all -- untested, not unaffected")
        print(R["verdict_shortcut"], flush=True)
        Path(a.out).write_text(json.dumps(R, indent=1))
        return 0

    for r in range(a.aa_rounds):
        RP.set_enabled(False)
        ta = one_design()
        RP.set_enabled(False)
        tb = one_design()
        R["aa"].append({"round": r, "a_s": round(ta, 4), "b_s": round(tb, 4),
                        "apparent_delta_ms": round((ta - tb) * 1e3, 1)})
        print("aa:", R["aa"][-1], flush=True)
        Path(a.out).write_text(json.dumps(R, indent=1))

    for r in range(a.rounds):
        RP.set_enabled(False)
        n0 = RP.STATS[0]
        tb = one_design()
        nb = RP.STATS[0] - n0
        RP.set_enabled(True)
        n1 = RP.STATS[0]
        tw = one_design()
        nw = RP.STATS[0] - n1
        RP.set_enabled(False)
        R["rounds"].append({"round": r, "base_s": round(tb, 4), "wire_s": round(tw, 4),
                            "delta_ms": round((tb - tw) * 1e3, 1),
                            "eligible_served_base": nb, "eligible_served_wire": nw})
        print("ab:", R["rounds"][-1], flush=True)
        Path(a.out).write_text(json.dumps(R, indent=1))

    if R["rounds"]:
        base = [x["base_s"] for x in R["rounds"]]
        wire = [x["wire_s"] for x in R["rounds"]]
        R["summary"] = {"base_median_s": round(med(base), 4), "wire_median_s": round(med(wire), 4),
                        "delta_ms_median": round((med(base) - med(wire)) * 1e3, 1),
                        "signs": [1 if x["delta_ms"] > 0 else -1 for x in R["rounds"]],
                        "ratio": round(med(base) / med(wire), 5)}
        print("summary:", R["summary"], flush=True)
    Path(a.out).write_text(json.dumps(R, indent=1))
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
