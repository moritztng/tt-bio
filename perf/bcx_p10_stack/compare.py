#!/usr/bin/env python3
"""Pool a chain's arms and report the composed round against all-off.

Reads every `round_events.json` under `out/`, keys each by the arm its own stamp records
(extra_msa_on_device, template_on_device, triatt_taped_sdpa), pools the timed rounds across
cycles and prints the per-arm median with the AICLK and loadavg that arm actually ran at.
Round 1 of every process carries BindCraft 2's jit compile and is dropped, the same rule
`perf/bcx_round/analyze.py` uses.

Every arm also prints its three reach counters. A lever that never fired reads as an agreeing
A/B, and this campaign has been bitten by that twice, so an arm whose counters are zero while
its switch is on is reported as DEAD rather than as a result.

    PYTHONPATH=. python3 perf/bcx_p10_stack/compare.py perf/bcx_p10_stack/out
"""
import json
import pathlib
import statistics as st
import sys

LEVERS = ("extra_msa_on_device", "template_on_device", "triatt_taped_sdpa", "triatt_hifi",
          "triatt_bw_fused")

#: The third lever has two routes and they are mutually exclusive, so an arm is named by which
#: one it took rather than by a bit. Keying on `triatt_taped_sdpa` alone would pool a `hifi` arm
#: with a triangle-attention-OFF arm, because hifi leaves that flag false.
ROUTE = {(0, 0): "off", (1, 0): "agtri", (0, 1): "hifi"}


def med(xs):
    return round(st.median(xs), 3) if xs else None


def rounds_of(path):
    d = json.load(open(path))
    ev, stamp, clk = d["events"], d["stamp"], d["aiclk"]
    starts = [e for e in ev if e["kind"] == "round_start"]
    stop = [e for e in ev if e["kind"] == "round_stop"]
    bounds = [e["t0"] for e in starts] + ([stop[0]["t0"]] if stop else [])
    out = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        inside = [e for e in ev if e.get("t1") is not None
                  and e["t0"] >= t0 - 1e-6 and e["t1"] <= t1 + 1e-6]
        sg = [e for e in inside if e["phase"] == "sequence_gradients"]
        if not sg:
            continue
        w0, w1 = sg[0]["t0"], sg[0]["t1"]
        dev = [e for e in inside if e["kind"] == "device" and w0 <= e["t0"] and e["t1"] <= w1]
        dev_s = sum(e["dt"] for e in dev)
        s = [(c, l) for t, c, l in clk if t0 <= t <= t1]
        aiclk = sorted(c for c, _ in s)
        out.append({"wall": t1 - t0, "sg": sum(e["dt"] for e in sg), "dev": dev_s,
                    "host": sum(e["dt"] for e in sg) - dev_s, "n_dev": len(dev),
                    "aiclk": aiclk[len(aiclk) // 2] if aiclk else None,
                    "aiclk_min": aiclk[0] if aiclk else None,
                    "load1": sum(l for _, l in s) / len(s) if s else None})
    # round 1 carries the jit compile
    return (out[1:] or out), stamp


def reach(stamp):
    """(extra-MSA blocks spliced, template taped calls, triangle attentions served).

    The third entry counts whichever route the arm took. A `hifi` arm that declined because the
    call was taped is the failure this counter exists to catch, so it is surfaced separately.
    """
    e = stamp.get("extra_msa_swapped") or []
    t = (stamp.get("template_calls") or {}).get("taped", 0)
    fused = stamp.get("fused_hifi_stats") or {}
    tri = ((stamp.get("triatt_sdpa_stats") or {}).get("served", 0)
           + fused.get("served", 0))
    # The FORWARD route and the BACKWARD kernel are two different levers on the same call and
    # either can fire without the other, so they are counted separately. A backward arm reading
    # 0 here with the switch on has measured the forward twice.
    bw = (stamp.get("triatt_bw_stats") or {}).get("served", 0)
    return (sum(e), t, tri, bw)


def dead_hifi(stamp):
    """A hifi arm that served nothing and declined for being taped never fired."""
    if not stamp.get("triatt_hifi"):
        return None
    f = stamp.get("fused_hifi_stats") or {}
    if not f.get("served"):
        return "hifi served 0, taped=%s declined=%s" % (f.get("taped"), f.get("declined"))
    return None


def main(root):
    arms = {}
    for p in sorted(pathlib.Path(root).rglob("round_events.json")):
        if "smoke" in str(p) or "headline" in str(p):  # headline/ is a copy
            continue
        rows, stamp = rounds_of(p)
        key = tuple(int(bool(stamp.get(k))) for k in LEVERS)
        a = arms.setdefault(key, {"rows": [], "src": [], "reach": []})
        a["rows"] += rows
        a["src"].append(p.parent.name)
        a["reach"].append(reach(stamp))
        a.setdefault("dead", []).append(dead_hifi(stamp))

    base = arms.get((0, 0, 0, 0, 0))
    print(f"{'arm (msa,tmpl,route)':<24} {'n':>3} {'round s':>9} {'min':>7} {'max':>7} "
          f"{'host':>7} {'dev':>7} {'x':>6} {'AICLK':>6} {'load1':>6}  reach")
    ref = med([r["wall"] for r in base["rows"]]) if base else None
    for key in sorted(arms, key=lambda k: (-sum(k), k)):
        a = arms[key]
        w = med([r["wall"] for r in a["rows"]])
        clk = sorted(r["aiclk"] for r in a["rows"] if r["aiclk"])
        dead = [i for i, on in enumerate(key[:2]) if on
                and all(r[i] == 0 for r in a["reach"])]
        if sum(key[2:4]) and all(r[2] == 0 for r in a["reach"]):
            dead.append(2)
        if key[4] and all(r[3] == 0 for r in a["reach"]):
            dead.append(4)
        tag = "  DEAD:" + ",".join(LEVERS[i] for i in dead) if dead else ""
        tag += "".join("  " + d for d in a.get("dead", []) if d)
        name = "(%d,%d,%s%s)" % (key[0], key[1], ROUTE.get((key[2], key[3]), "BOTH?"),
                                 "+bw" if key[4] else "")
        print(f"{name:<24} {len(a['rows']):>3} {w:>9.3f} "
              f"{min(r['wall'] for r in a['rows']):>7.3f} "
              f"{max(r['wall'] for r in a['rows']):>7.3f} "
              f"{med([r['host'] for r in a['rows']]):>7.3f} "
              f"{med([r['dev'] for r in a['rows']]):>7.3f} "
              f"{(ref / w if ref else 0):>6.3f} "
              f"{(clk[len(clk)//2] if clk else 0):>6} "
              f"{med([r['load1'] for r in a['rows']]):>6.1f}  "
              f"{[tuple(x) for x in a['reach']]}{tag}")
    print()
    for key in sorted(arms, key=lambda k: (-sum(k), k)):
        print(key, "<-", arms[key]["src"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "perf/bcx_p10_stack/out")
