#!/usr/bin/env python3
"""mp_read.py: old-release cells (qb1 + .114) vs gapboard's main cells -> the month table (markdown on stdout).

Old cells are read with gapboard's own PROBE (same median, clock-on-own-node, load and n filters); main cells are
gapboard's current() quotes from the ledger. Design cells (af2ig/bc2/boltzgen) are read by read_design().
"""
import json, re, shlex, statistics as st, subprocess, sys
sys.path.insert(0, "/home/moritz/.coworker/scripts")
import gapboard as G

OLD = {"v0.10.0": "1ee324e01", "v0.13.1": "60a3796f0"}
SRC = {"p150a": ("qb1", "~/spd/monthpost/out/{s}/*"), "wh": ("ubuntu@172.16.102.114", "~/spd/spd-monthpost/wh{s}/*")}


def probe(host, glob):
    out = G.sh(G.SSH + [host, "python3 - " + " ".join(map(shlex.quote, ["[]", glob]))], inp=G.PROBE, t=300)
    rows = []
    for l in out.splitlines():
        try:
            r = json.loads(l)
        except ValueError:
            continue
        v = sorted(r.get("vals") or [])
        r["value_s"] = (v[len(v) // 2] if len(v) % 2 else sum(v[len(v) // 2 - 1:len(v) // 2 + 1]) / 2) if v else None
        why = []
        if r.get("err"): why.append(r["err"])
        if r["rc"]: why.append(f"rc={r['rc']}")
        if len(v) < 3: why.append(f"n={len(v)}")
        arch = "wh" if "172.16" in host else "p150a"
        if (r.get("mhz") or 0) < G.MIN_MHZ[arch]: why.append(f"clock {r.get('mhz')}")
        if (r.get("load") or 0) > 1.0: why.append(f"load {r.get('load')}")
        r["why"] = "; ".join(why)
        r["cell"] = r["src"].rstrip("/").rsplit("/", 1)[1]
        rows.append(r)
    return rows


def main():
    cur = G.current(G.ledger())
    old = {}
    for tag, s in OLD.items():
        for arch, (h, g) in SRC.items():
            for r in probe(h, g.format(s=s)):
                old[(tag, arch, r["cell"])] = r
    print("| model | arch | mode | v0.10.0 s | v0.13.1 s | main s (sha) | x vs 0.10.0 | x vs 0.13.1 | GAP then -> now | notes |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in G.FOLD + G.EMBED:
        for arch in ("p150a", "wh"):
            for mode in ("normal", "fast"):
                n = cur.get((m, mode, arch))
                ref = G.REFS[m][mode == "fast"]
                cells, notes = [], []
                for tag in OLD:
                    r = old.get((tag, arch, f"{m}-{mode}"))
                    if r and not r["why"]:
                        cells.append(r["value_s"]); notes.append(f"{tag} {r['mhz']:.0f} MHz n={len(r['vals'])} dig={r.get('dig')}")
                    else:
                        cells.append(None); notes.append(f"{tag}: {r['why'] if r else 'not run'}")
                nv = n["value_s"] if n else None
                f = lambda x: "-" if x is None else (f"{x:.4f}" if x < 1 else f"{x:.2f}")
                sp = lambda o: "-" if (o is None or nv is None) else f"{o / nv:.2f}x"
                gap = "-" if nv is None else f"{(cells[0] or cells[1] or float('nan')) / ref:.2f} -> {nv / ref:.2f}"
                print(f"| {m} | {arch} | {mode} | {f(cells[0])} | {f(cells[1])} | {f(nv)} ({n['sha'] if n else '-'}, "
                      f"{n['mhz']:.0f} MHz)" if n else f"| {m} | {arch} | {mode} | {f(cells[0])} | {f(cells[1])} | - ",
                      end="")
                print(f" | {sp(cells[0])} | {sp(cells[1])} | {gap} | {'; '.join(notes)} |")


if __name__ == "__main__":
    main()
