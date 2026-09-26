#!/usr/bin/env python3
"""Split a round's DEVICE column into the things inside it that are not block work.

`perf/bcx_round/analyze.py` charges the whole `_taped`/`_backward` callback to `device`, which
is right for the host/device split and wrong for a block attribution: the callback also copies
four numpy arrays, pads them, casts to bfloat16, uploads, waits on `synchronize_device` and
brings the answer back. `bcx-p10-devmap`'s 2.603 s residual is measured against that column, so
every one of those rows is inside it.

Reads `run_resid.py`'s event log and prints, per round and per module:

    device column = marshal_in + dispatch + sync + marshal_out + trunk_load + unaccounted

`trunk_load` is the lazy weight upload the pool does the first time a round asks for a
checkpoint it does not hold; it lands inside whichever device call touched it first.
"""
import collections
import json
import statistics as st
import sys

CATS = ("marshal_in", "dispatch", "sync", "marshal_out", "trunk_load")


def med(xs):
    return round(st.median(xs), 4) if xs else None


def cat(phase):
    if phase.startswith("marshal_in"):
        return "marshal_in"
    if phase.startswith("marshal_out"):
        return "marshal_out"
    if phase == "dispatch:sync":
        return "sync"
    if phase.startswith("dispatch"):
        return "dispatch"
    if phase == "trunk_load":
        return "trunk_load"
    return "other"


def main(path, out=None):
    d = json.load(open(path))
    ev, stamp = d["events"], d["stamp"]
    starts = [e for e in ev if e["kind"] == "round_start"]
    stop = [e for e in ev if e["kind"] == "round_stop"]
    bounds = [e["t0"] for e in starts] + ([stop[0]["t0"]] if stop else [])

    rows = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        inside = [e for e in ev if e.get("t1") is not None and t0 - 1e-6 <= e["t0"]
                  and e["t1"] <= t1 + 1e-6]
        dev = [e for e in inside if e["kind"] == "device"]
        cross = [e for e in inside if e["kind"] == "crossing"]
        seam = [e for e in inside if e["kind"] == "seam"]
        sg = [e for e in inside if e["phase"] == "sequence_gradients"]
        by = collections.Counter()
        for e in seam:
            by[(e["call"].split(":")[0], cat(e["phase"]))] += e["dt"]
        col = sum(e["dt"] for e in dev)
        row = {"round": starts[i]["round"], "wall": round(t1 - t0, 3),
               "sg": round(sum(e["dt"] for e in sg), 3),
               "device_col": round(col, 3),
               "models": sorted({e["model"] for e in cross if e.get("model")}),
               "fresh_trunk": sorted({e["model"] for e in cross if not e["trunk_held"]}),
               "trunk_load_s": round(by[("evoformer", "trunk_load")]
                                     + by[("extra_msa", "trunk_load")], 4)}
        for module in ("evoformer", "extra_msa"):
            mdev = sum(e["dt"] for e in dev if e["module"] == module)
            row[module] = round(mdev, 3)
            for c in CATS:
                row[f"{module}:{c}"] = round(by[(module, c)], 4)
            row[f"{module}:unacc"] = round(mdev - sum(by[(module, c)] for c in CATS), 4)
        rows.append(row)

    body = rows[1:] or rows
    keys = ([f"{m}:{c}" for m in ("evoformer", "extra_msa")
             for c in CATS + ("unacc",)] + ["device_col", "wall", "sg", "trunk_load_s",
                                            "evoformer", "extra_msa"])
    summary = {k: med([r[k] for r in body]) for k in keys}
    summary["rounds_measured"] = len(rows)
    summary["seam_total"] = round(sum(summary[f"{m}:{c}"] for m in
                                      ("evoformer", "extra_msa")
                                      for c in ("marshal_in", "sync", "marshal_out")), 4)
    # The whole-run view of the trunk loads, which are one-offs and do not belong in a median.
    loads = [e for e in ev if e["kind"] == "seam" and e["phase"] == "trunk_load"]
    res = {"stamp": {k: stamp.get(k) for k in
                     ("host", "card", "commit", "exact", "extra_msa_on_device", "shipped_pool",
                      "binder_pinned", "seed", "pool_selections", "pool_on_card",
                      "device_calls", "extra_msa_calls", "extra_msa_swapped",
                      "loadavg_start", "loadavg_end", "wall_seconds")},
           "summary": summary, "rounds": rows,
           "trunk_loads": [{"model": e.get("model"), "s": e["dt"], "round": e["round"],
                            "call": e["call"]} for e in loads]}
    print(json.dumps({"summary": summary, "trunk_loads": res["trunk_loads"]}, indent=1))
    for r in rows:
        print(f"round {r['round']:2d} wall {r['wall']:7.3f} dev {r['device_col']:7.3f} "
              f"evo {r['evoformer']:7.3f} extra {r['extra_msa']:6.3f} | "
              f"marshal {r['evoformer:marshal_in'] + r['extra_msa:marshal_in']:.3f}"
              f"/{r['evoformer:marshal_out'] + r['extra_msa:marshal_out']:.3f} "
              f"sync {r['evoformer:sync'] + r['extra_msa:sync']:.3f} "
              f"load {r['trunk_load_s']:.3f} {','.join(r['fresh_trunk']) or '-'}")
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
