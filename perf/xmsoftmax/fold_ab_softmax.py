#!/usr/bin/env python3
"""Fold-embedded A/B for the accurate-softmax flip, with the A/A control interleaved.

Why this file exists. Two prior passes measured this flip one fold per process, arms in sequential
blocks, and got an A/A floor of 10.0% at 256 aa and 19.2% on the 20 aa perf cell against effects
predicted at 1.27-1.41%. Under those floors a PASS is not evidence. Three separate causes, all
fixed here:

1. **Sequential blocks let drift land on one arm.** Arms alternate per rep here (A, A', B), so the
   off arm brackets the on arm in the same window and box drift hits both.
2. **A new process per fold re-pays device open, kernel compile and weight load, and each fold sees
   its own slice of co-tenant load.** One process, one device, one weight load, one MSA cache.
3. **The fold wall cannot resolve this lever and never could.** The softmax is 0.22-0.33% of a fold
   for these two models, so the fold wall is asked to resolve 1.3% through everything else in the
   fold. The headline instrument here is the synchronised block wall of the modules the lever is
   actually inside (`AttentionPairBias.__call__` and `PairformerLayer.__call__`), summed over their
   real in-fold executions. That is not a per-call cost times a call census: the calls are the
   fold's own, at the fold's own shapes and residency, and the number is an absolute ms delta.
   `perf/size512/fold_ab512.py` uses the same instrument for the same reason (its fold-wall A/A
   floor is 758.3 ms and cannot resolve a null).

The arm is a call-time attribute write, not a rebuild. Construction runs once with
``TT_BIO_ACCURATE_SOFTMAX_AB`` naming the sites under test, which is what marks the instances the
verdict is about; the harness then collects exactly those instances and flips
``accurate_softmax`` on them between folds. So the two arms share weights, allocation order and
device state, and the only difference is the branch inside `AttentionPairBias.__call__`.

Reachability is counted, not assumed: `_accurate_softmax` is wrapped, so every record carries how
many times the accurate chain actually ran. An on arm with a zero count is a broken A/B, and this
task has already produced three levers that did not reach their code.

Usage:
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:<slug> \\
  TT_BIO_ACCURATE_SOFTMAX_AB=opendde.trunk,opendde.confidence,opendde.refiner \\
  python3 perf/xmsoftmax/fold_ab_softmax.py --model opendde --sizes 256 --reps 3 --out out.json
"""
import argparse, hashlib, json, os, socket, statistics as st, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
STATE = {"dev": None, "sites": [], "acc_calls": 0}


def timed_call(key, fn, *a, **kw):
    import ttnn
    ttnn.synchronize_device(STATE["dev"])
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    ttnn.synchronize_device(STATE["dev"])
    w = WALL[key]
    w["n"] += 1
    w["s"] += time.perf_counter() - t0
    return out


def sha_dir(d):
    out = {}
    for p in sorted(Path(d).glob("*")):
        if p.is_file():
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def collect_lever_instances(root):
    """Every object under `root` whose `accurate_softmax` construction flag came out True.

    Construction ran with TT_BIO_ACCURATE_SOFTMAX_AB set, so True is exactly the site selection
    under test. Walking for it rather than for a class name keeps the harness honest about scope:
    what gets flipped is what the selector reached, nothing wider.
    """
    found, seen = [], set()
    stack = [root]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, (str, bytes, int, float, bool, type(None))):
            continue
        if isinstance(o, dict):
            stack.extend(o.values())
            continue
        if isinstance(o, (list, tuple, set)):
            stack.extend(o)
            continue
        d = getattr(o, "__dict__", None)
        if d is None:
            continue
        if o.__dict__.get("accurate_softmax") is True:
            found.append(o)
        stack.extend(d.values())
    return found


def _selftest():
    """Device-free check of the walker, so a quiet box is not spent debugging it.

    The walker is the one piece of this harness with no prior art: everything else is the same
    build_fold/timed_call shape perf/size512/fold_ab512.py already uses. It has to reach through
    plain attributes, lists and dicts (a Pairformer keeps its layers in `self.blocks`, OpenDDE keeps
    its trunk behind `self._protenix`), find only the instances whose construction flag came out
    True, and terminate on the cycles a module graph has.
    """
    class N:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    on1, on2, off1 = N(accurate_softmax=True), N(accurate_softmax=True), N(accurate_softmax=False)
    plain = N(x=1)
    leaf = N(accurate_softmax=True)
    root = N(blocks=[N(apb=on1), N(apb=off1)], nested={"trunk": N(deep=[[on2]])},
             _protenix=N(refiner=leaf), plain=plain, scalar=3, text="accurate_softmax=True")
    root.self_ref = root                      # a cycle must not hang the walk
    plain.back = root
    found = collect_lever_instances(root)
    ids = {id(o) for o in found}
    assert ids == {id(on1), id(on2), id(leaf)}, \
        "walker found %d instances, expected the 3 flagged ones" % len(found)
    for o in found:
        o.accurate_softmax = False
    assert collect_lever_instances(root) == [], "flipping off must empty the selection"
    print("selftest OK: 3 flagged instances through list, dict, nested list and private attr; "
          "cycle terminated; off arm selects nothing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="opendde")
    ap.add_argument("--sizes", default="256")
    ap.add_argument("--reps", type=int, default=3, help="A, A', B triplets per size")
    ap.add_argument("--arms", default="off,off,on",
                    help="one triplet, repeated --reps times; off,off,on gives a paired A/A and A/B")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--instrument", choices=("pf", "apb", "both"), default="pf",
                    help="which block wall to time. pf (default) is the outer one: it contains the "
                         "whole lever and costs 1208 syncs per fold instead of apb's 10568")
    ap.add_argument("--selftest", action="store_true",
                    help="check the instance walker without a device, then exit")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return
    assert a.out, "--out is required unless --selftest"

    sel = os.environ.get("TT_BIO_ACCURATE_SOFTMAX_AB", "")
    assert sel, "set TT_BIO_ACCURATE_SOFTMAX_AB to the sites under test before constructing"

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import importlib.metadata as im
    from tt_bio.main import (_detect_p300_devices, _find_ttnn_mesh_graph_descriptor,
                             _resolve_recycling_steps, _resolve_sampling_steps)
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)

    assert Path(T.__file__).resolve().is_relative_to(ROOT), \
        "scoring the wrong tree: %s" % T.__file__

    # Reachability counter. The lever is a branch inside AttentionPairBias.__call__, so a wrapped
    # `_accurate_softmax` is the only way to know the on arm ran it and the off arm did not.
    _acc = T._accurate_softmax

    def counted(*x, **k):
        STATE["acc_calls"] += 1
        return _acc(*x, **k)
    T._accurate_softmax = counted

    # ONE level at a time, and by default the outer one. Wrapping both nests the timers:
    # PairformerLayer contains AttentionPairBias, so the PF wall would carry APB's syncs as well as
    # its own, and the two numbers stop being independent. Worse, APB runs 5284 times per 256 aa
    # protenix fold, so wrapping it costs 10568 `synchronize_device` calls -- the measured APB wall
    # then swings 41% between two folds of the SAME arm (2096.7 vs 2967.7 ms), because a host-side
    # sync spin is exactly what co-tenant load lengthens. That is the oversync inflation from
    # tt-bio-isolated-op-timing-oversync-inflates-cost, at 5284x. PairformerLayer runs 604 times,
    # 1208 syncs, and its A/A spread came out 5x tighter on the same data.
    INSTR = {"pf": ((T.PairformerLayer, "block:PairformerLayer"),),
             "apb": ((T.AttentionPairBias, "block:AttentionPairBias"),),
             "both": ((T.AttentionPairBias, "block:AttentionPairBias"),
                      (T.PairformerLayer, "block:PairformerLayer"))}
    for cls, key in INSTR[a.instrument]:
        f = cls.__call__
        cls.__call__ = (lambda g, k: lambda self, *x, **kw: timed_call(k, g, self, *x, **kw))(f, key)

    def set_arm(name):
        on = name == "on"
        for o in STATE["sites"]:
            o.accurate_softmax = on

    res = {"_meta": {"instrument": a.instrument, "host": socket.gethostname(), "ttnn": im.version("ttnn"),
                     "chip": os.environ.get("TT_VISIBLE_DEVICES", "?"),
                     "sites": sel, "model": a.model, "tt_bio": T.__file__,
                     "recycling_steps": B.RECYCLING_STEPS,
                     "sampling_steps": B.SAMPLING_STEPS,
                     "instrument": "synchronised block wall of the lever's own modules, summed "
                                   "over their in-fold executions; arms interleaved per rep"},
           "runs": []}

    def flush():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))

    for size in [int(s) for s in a.sizes.split(",")]:
        tgt = a.fixdir / f"cdk2x2_{size}.yaml"
        a3m = a.fixdir / f"cdk2x2_{size}.a3m"
        one_fold, meta, state = B.build_fold(
            a.model, ROOT / f".msa_xmsm_{a.model}_{size}", tgt, a3m)
        STATE["dev"] = T.get_device()
        STATE["sites"] = collect_lever_instances(state.model)
        struct_dir = Path(meta["struct_dir"])
        n_sites = len(STATE["sites"])
        print(f"=== {a.model} {size} aa: {n_sites} lever instances from '{sel}' ===", flush=True)
        assert n_sites, "the selector reached no construction site: nothing to A/B"

        # Cold fold warms kernels, caches and the MSA. Its wall is not comparable to anything.
        set_arm("off")
        STATE["acc_calls"] = 0
        WALL.clear()
        cold_s, cold_m = one_fold()
        print(f"  cold(off) {cold_s:.2f}s acc_calls={STATE['acc_calls']} "
              f"n_tokens={cold_m.get('n_tokens')}", flush=True)
        assert STATE["acc_calls"] == 0, "the off arm ran the accurate chain: arm scoping is wrong"

        arms = [x.strip() for x in a.arms.split(",") if x.strip()]
        for rep in range(a.reps):
            for i, arm in enumerate(arms):
                set_arm(arm)
                WALL.clear()
                STATE["acc_calls"] = 0
                la = os.getloadavg()[0]
                t0 = time.perf_counter()
                try:
                    fold_s, m = one_fold()
                except Exception as e:                                          # noqa: BLE001
                    res["runs"].append({"size": size, "arm": arm, "rep": rep, "slot": i,
                                        "error": f"{type(e).__name__}: {e}"[:400]})
                    flush()
                    print(f"  {arm} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
                    continue
                rec = {"size": size, "arm": arm, "rep": rep, "slot": i,
                       "fold_s": round(fold_s, 3), "loadavg": round(la, 2),
                       "acc_calls": STATE["acc_calls"], "n_sites": n_sites,
                       "plddt": m.get("plddt"), "n_tokens": m.get("n_tokens"),
                       "cif_sha256": sha_dir(struct_dir),
                       "wall_ms": {k: {"n": v["n"], "ms": round(v["s"] * 1e3, 2)}
                                   for k, v in sorted(WALL.items())}}
                res["runs"].append(rec)
                flush()
                blk = rec["wall_ms"].get("block:PairformerLayer", {})
                apb = rec["wall_ms"].get("block:AttentionPairBias", {})
                print(f"  rep{rep} slot{i} {arm:>3}: fold {fold_s:7.2f}s  "
                      f"pf {blk.get('ms', 0):9.1f}ms/{blk.get('n', 0)}  "
                      f"apb {apb.get('ms', 0):8.1f}ms/{apb.get('n', 0)}  "
                       f"acc={STATE['acc_calls']} load={la:.1f}", flush=True)
                if arm == "on":
                    assert STATE["acc_calls"] > 0, "on arm never ran the accurate chain"

    # Paired summary. The A/A pair is the floor; the A/B delta is only claimable against it.
    summ = {}
    for size in sorted({r["size"] for r in res["runs"] if "fold_s" in r}):
        rs = [r for r in res["runs"] if r.get("size") == size and "fold_s" in r]
        def series(arm, slot):
            return [r for r in rs if r["arm"] == arm and r["slot"] == slot]
        def metric(r, key):
            return r["wall_ms"].get(key, {}).get("ms")
        out = {}
        keys = sorted({k for r in rs for k in r["wall_ms"]})
        for key in keys:
            pairs_aa, pairs_ab = [], []
            for rep in sorted({r["rep"] for r in rs}):
                byslot = {r["slot"]: r for r in rs if r["rep"] == rep}
                a0, a1, b = byslot.get(0), byslot.get(1), byslot.get(2)
                if a0 and a1 and metric(a0, key) and metric(a1, key):
                    pairs_aa.append(metric(a1, key) / metric(a0, key) - 1.0)
                if a0 and a1 and b and metric(b, key):
                    base = (metric(a0, key) + metric(a1, key)) / 2
                    pairs_ab.append(metric(b, key) / base - 1.0)
            out[key] = {
                "paired_aa_pct": [round(x * 100, 2) for x in pairs_aa],
                "paired_ab_pct": [round(x * 100, 2) for x in pairs_ab],
                "aa_median_pct": round(st.median(pairs_aa) * 100, 2) if pairs_aa else None,
                "ab_median_pct": round(st.median(pairs_ab) * 100, 2) if pairs_ab else None,
                "aa_spread_pct": round((max(pairs_aa) - min(pairs_aa)) * 100, 2) if pairs_aa else None,
            }
        folds_off = [r["fold_s"] for r in rs if r["arm"] == "off"]
        folds_on = [r["fold_s"] for r in rs if r["arm"] == "on"]
        out["fold_s"] = {"off": folds_off, "on": folds_on,
                         "off_median": round(st.median(folds_off), 3) if folds_off else None,
                         "on_median": round(st.median(folds_on), 3) if folds_on else None}
        cifs = {json.dumps(r["cif_sha256"], sort_keys=True) for r in rs if r["arm"] == "off"}
        out["off_arm_cifs_identical"] = len(cifs) == 1
        summ[str(size)] = out
    res["summary"] = summ
    flush()
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
