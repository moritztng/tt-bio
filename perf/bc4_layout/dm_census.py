"""Price the ttnn data-movement calls of one BindCraft 2 iteration from a `bc2_evo_devprof.py --census`.

`synced_s` carries a per-call instrument floor, and it is two different floors: a call that only
rewrites metadata (a view reshape) reads 12-15 us and moves nothing, a call that launches a program
reads >= ~30 us even on a [2, 224] slice. So a row below `--view-us` per call is priced at 0 and
every other row at `synced - calls * floor_us`. What remains is device time spent moving bytes.

Rows are tagged with the producer -> consumer pair they belong to (PAIRS, by signature) and whether
the consumer could read the producer's layout directly. Usage:
    python perf/bc4_layout/dm_census.py devprof.json [--json out.json]
"""
import argparse
import collections
import json
import re

DM = {"permute", "transpose", "typecast", "concat", "reshape", "slice", "clone", "to_layout",
      "tilize", "untilize", "pad", "repeat"}

# (regex on "name sig", pair, root fix). First match wins; untagged rows are "other".
PAIRS = [
    (r"^permute \[224, 128, 224\]|^permute \[224, 224, 128\]",
     "OPM matmul [I,c,J] -> pair [I,J,c]", "none in ttnn: matmul cannot broadcast a 2D in0 over in1's batch"),
    (r"^reshape \[224, 4096\]|^to_layout \[224, 4096\]|^to_layout \[224, 32, 128\]|^reshape \[224, 32, 128\]|^permute \[224, 32, 128\]|^permute \[224, 128, 32\]"
     r"|^concat \(\[224, 128, 32\]|^slice \[224, 128, 64\]|^permute \[224, 32\]|^permute \[32, 224\]",
     "OPM a@W -> [I,c,D] operand", "fold the (D,c) -> (c,D) order into W: one reshape, no permute"),
    (r"^reshape \[224, 224\]:\S+,\(1,224,224,1\)",
     "OPM mask norm -> divide", "a pure function of msa_mask, cache it per mask like _MaskBiasCache"),
    (r"^transpose \[224, 1, 224, 128\]|^transpose \[224, 4, 32, 224\]|^transpose \[224, 1, 128, 224\]"
     r"|^transpose \[224, 4, 224, 32\]",
     "triatt grad merged -> heads (split_heads VJP)", "one head-split op, or the triatt bw kernel reads [S,H*d]"),
    (r"^permute \[1, 224, 224, (4|8)\]|^transpose \[1, (4|8), 224, 224\]|^transpose \[1, 224, (4|8), 224\]",
     "pair-bias linear [N,N,H] -> attention [H,N,N]", "attention kernel reads [N,N,H] (triatt seam)"),
    (r"^typecast \[2, 8, 224, 224\]|^typecast \[1, 4, 224, 224\]",
     "fp32 softmax logits casts", "precision verb, not layout"),
    (r"^concat \(\[224, 4, 224, 16\]",
     "template triatt d=16 head concat", "sub-tile last-dim concat (~14 GB/s); keep heads tile-aligned"),
    (r"^reshape \[224, 4, 16, 224\]", "template d=16 head merge", "sub-tile head width"),
]


def tag(o):
    key = f"{o['name']} {o['sig']}"
    for rx, pair, fix in PAIRS:
        if re.search(rx, key):
            return pair, fix
    return "other", ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("devprof")
    p.add_argument("--floor-us", type=float, default=30.0)
    p.add_argument("--view-us", type=float, default=20.0)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--json")
    a = p.parse_args()
    d = json.load(open(a.devprof))
    ops = d["census"]["ops"]
    rows, view_s, synced = [], 0.0, 0.0
    for o in ops:
        if o["name"] not in DM:
            continue
        synced += o["synced_s"]
        per = o["synced_s"] / o["calls"]
        if per * 1e6 < a.view_us:
            view_s += o["synced_s"]
            continue
        corr = max(0.0, o["synced_s"] - o["calls"] * a.floor_us * 1e-6)
        pair, fix = tag(o)
        rows.append(dict(phase=o["phase"], origin=o["origin"], name=o["name"], sig=o["sig"][:120],
                         calls=o["calls"], us_per_call=round(per * 1e6, 1), corr_s=round(corr, 5),
                         gb=round(o["bytes"] / 1e9, 3), pair=pair, fix=fix))
    rows.sort(key=lambda r: -r["corr_s"])
    by_pair = collections.defaultdict(lambda: [0.0, 0, 0.0])
    by_phase = collections.defaultdict(float)
    for r in rows:
        by_pair[r["pair"]][0] += r["corr_s"]
        by_pair[r["pair"]][1] += r["calls"]
        by_pair[r["pair"]][2] += r["gb"]
        by_phase[r["phase"]] += r["corr_s"]
    total = sum(r["corr_s"] for r in rows)
    out = dict(source=a.devprof, sha=d.get("git_sha"), host=d.get("host"), card=d.get("tt_visible_devices"),
               resident=d.get("env", {}).get("TT_BIO_BC2_RESIDENT_BLOCKS"), floor_us=a.floor_us,
               dm_synced_s=round(synced, 4), view_only_synced_s=round(view_s, 4),
               moving_corr_s=round(total, 4), by_phase={k: round(v, 4) for k, v in by_phase.items()},
               by_pair={k: dict(corr_s=round(v[0], 4), calls=v[1], gb=round(v[2], 2))
                        for k, v in sorted(by_pair.items(), key=lambda kv: -kv[1][0])},
               rows=rows)
    print(f"{a.devprof}: sha {out['sha']} host {out['host']} card {out['card']} K={out['resident']}")
    print(f"data movement synced {synced:.3f} s, view-only {view_s:.3f}, byte-moving corrected {total:.3f}")
    for k, v in out["by_pair"].items():
        print(f"  {v['corr_s']:7.4f} s  {v['calls']:5d} calls {v['gb']:7.2f} GB  {k}")
    for r in rows[:a.top]:
        print(f"  {r['corr_s']:.4f} {r['us_per_call']:7.1f}us x{r['calls']:<4d} {r['phase']:<18s} {r['name']} {r['sig'][:70]}  [{r['pair']}]")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
