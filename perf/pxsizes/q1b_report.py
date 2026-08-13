#!/usr/bin/env python3
"""Read the q1b one-arm-per-process JSONs and print the 768/1024 qb2 rows.

Nothing here computes a number the JSONs do not contain: the ratio comes from the second process of
each arm (the first is the compile discard, see run_q1b_qb2.sh), each arm's own cross-process spread
is the floor, and the stop rule is applied rather than described.
"""
import json
import pathlib
import sys

O = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/pxsizes")
COMPS = ["body:TriangleAttention", "body:TriangleMultiplication", "body:Transition",
         "body:AttentionPairBias", "body:PairWeightedAveraging", "block:PairformerLayer"]


def load(size, arm, p):
    f = O / f"q1b_{size}_{arm}_{p}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    if not d.get("runs"):
        return None
    r = d["runs"][0]
    r["_ttnn"], r["_host"], r["_chip"], r["_file"] = d["ttnn"], d["host"], d["chip"], f.name
    return r


def ms(r, k):
    return r.get("wall_ms", {}).get(k, {}).get("ms")


for size in (768, 1024):
    got = {(arm, p): load(size, arm, p) for arm in ("on", "base4") for p in ("p1", "p2")}
    have = {k: v for k, v in got.items() if v}
    if not have:
        print(f"\n## {size} aa: no process has written a JSON yet\n")
        continue
    a = next(iter(have.values()))
    print(f"\n## {size} aa  (host {a['_host']}, card {a['_chip']}, ttnn {a['_ttnn']}, "
          f"grid {a.get('grid')})\n")
    print("| process | arm | fold s | plDDT | CIF sha256 | block ms | calls |")
    print("|---|---|---:|---:|---|---:|---:|")
    for (arm, p), r in got.items():
        if not r:
            print(f"| {p} | `{arm}` | — (no JSON) | | | | |")
            continue
        if "error" in r:
            print(f"| {p} | `{arm}` | **ERROR** {r['error'][:60]} | | | | |")
            continue
        print(f"| {p} | `{arm}` | {r['fold_s']:.3f} | {r['plddt']} | `{r['cif_sha256'][:16]}` | "
              f"{r.get('block_wall_ms')} | {r.get('block_calls')} |")

    shas = {r["cif_sha256"] for r in have.values() if "cif_sha256" in r}
    plddts = {r["plddt"] for r in have.values() if "plddt" in r}
    print(f"\n- distinct CIF sha256 across arms: **{len(shas)}** {[s[:16] for s in shas]}")
    print(f"- distinct plDDT: {plddts}")

    for arm in ("on", "base4"):
        r1, r2 = got[(arm, "p1")], got[(arm, "p2")]
        if r1 and r2 and "fold_s" in r1 and "fold_s" in r2:
            d = r1["fold_s"] - r2["fold_s"]
            print(f"- `{arm}` cross-process spread p1-p2 = **{d:+.3f} s** "
                  f"({r1['fold_s']:.3f} then {r2['fold_s']:.3f})")

    on2, b42 = got[("on", "p2")], got[("base4", "p2")]
    if on2 and b42 and "fold_s" in on2 and "fold_s" in b42:
        gap = b42["fold_s"] - on2["fold_s"]
        ratio = b42["fold_s"] / on2["fold_s"]
        floors = [abs(got[(a_, "p1")]["fold_s"] - got[(a_, "p2")]["fold_s"])
                  for a_ in ("on", "base4")
                  if got[(a_, "p1")] and "fold_s" in got[(a_, "p1")]]
        worst = max(floors) if floors else None
        print(f"\n**base4/on (p2 of each) = {b42['fold_s']:.3f} / {on2['fold_s']:.3f} = "
              f"{ratio:.4f}x**, gap {gap:+.3f} s")
        if worst is not None:
            ok = worst < gap / 3
            print(f"- worst per-arm cross-process spread {worst:.3f} s vs gap/3 = {gap/3:.3f} s -> "
                  f"**{'RESOLVED' if ok else 'NOT RESOLVED'}** "
                  f"(gap is {gap/worst:.1f}x the spread)" if worst else "")
        print("\n| component | `base4` ms | `on` ms | delta ms | share of block gap |")
        print("|---|---:|---:|---:|---:|")
        bgap = (ms(b42, "block:PairformerLayer") or 0) - (ms(on2, "block:PairformerLayer") or 0)
        for c in COMPS:
            x, y = ms(b42, c), ms(on2, c)
            if x is None or y is None:
                continue
            share = f"{100*(x-y)/bgap:.1f} %" if bgap else "—"
            print(f"| {c.split(':')[1]} | {x:.1f} | {y:.1f} | {x-y:+.1f} | "
                  f"{share if c != 'block:PairformerLayer' else '(block gap)'} |")

    print("\n| lever counter | " + " | ".join(f"`{a_}` {p}" for (a_, p) in got if got[(a_, p)]) + " |")
    print("|---|" + "---|" * len(have))
    def row(label, fn):
        cells = []
        for k, r in got.items():
            if r:
                try:
                    cells.append(str(fn(r)))
                except Exception:
                    cells.append("—")
        print(f"| {label} | " + " | ".join(cells) + " |")
    row("E6 `gated_kernel` [enabled,[served,declined]]", lambda r: r["gated_kernel"])
    row("`back_kernel` [enabled,[served,declined]]", lambda r: r["back_kernel"])
    row("K1 head-major qkv served/declined",
        lambda r: f"{r['head_major_qkv']['served']}/{r['head_major_qkv']['declined']}")
    row("K1b tail served/declined",
        lambda r: f"{r['head_major_qkv']['tail_served']}/{r['head_major_qkv']['tail_declined']}")
    row("K2 `persistent_mask` enabled, served/declined",
        lambda r: f"{r['persistent_mask']['enabled']}, {r['persistent_mask']['served']}/"
                  f"{r['persistent_mask']['declined']}")
    row("K2 rejects", lambda r: ", ".join(f"{k}={v}" for k, v in
                                          list(r["persistent_mask"]["rejects"].items())[:3]) or "—")
    row("K1 rejects", lambda r: ", ".join(f"{k}={v}" for k, v in
                                          list(r["head_major_qkv"]["rejects"].items())[:3]) or "—")
    row("`_SDPA_WIDE_Q`", lambda r: r["sdpa_wide_q"])
    row("e6_default", lambda r: r["e6_default"])
    row("loadavg at write", lambda r: r["loadavg"][0])
