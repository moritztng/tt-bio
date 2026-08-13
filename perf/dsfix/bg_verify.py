"""Post-hoc sanity re-verification for every BoltzGen TT ladder point.

The in-harness check was too weak: it globbed the first *.cif under the output dir, and the
top-level <name>.cif is a target-only reference copy (R3: 3158 atoms, chain A only). Finiteness
passed on a file that contains no designed chain, so it could not have caught a run that
featurised the target and produced no binder.

This checks the real artifact, intermediate_designs/<name>.cif, and requires: two chains present,
the designed chain exactly 100 residues, the target chain exactly its known residue count, and
every coordinate finite.
"""
import collections, json, pathlib, sys

TARGET_RES = {"R0": 117, "R1": 196, "R2": 318, "R3": 414, "R4": 585}
BINDER = 100

rows = []
for d in sorted(pathlib.Path("/tmp").glob("bg_R*_b*_*")):
    name = d.name                       # bg_R3_b1_no
    parts = name.split("_")
    rung, b, arm = parts[1], parts[2], parts[3]
    cif = d / "intermediate_designs" / ("bg_%s.cif" % rung)
    if not cif.exists():
        rows.append({"dir": name, "ok": False, "why": "no intermediate_designs cif"})
        continue
    res, atoms, bad = collections.OrderedDict(), 0, 0
    for line in cif.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            f = line.split()
            atoms += 1
            res[(f[6], f[8])] = 1
            for tok in f[10:13]:
                try:
                    v = float(tok)
                except ValueError:
                    continue
                if v != v or abs(v) == float("inf"):
                    bad += 1
    ch = collections.Counter(k[0] for k in res)
    want_t = TARGET_RES[rung]
    sizes = sorted(ch.values())
    ok = (len(ch) == 2 and sizes == sorted([BINDER, want_t]) and bad == 0)
    why = ""
    if len(ch) != 2:
        why = "expected 2 chains, got %d" % len(ch)
    elif sizes != sorted([BINDER, want_t]):
        why = "chain sizes %s != [%d, %d]" % (sizes, BINDER, want_t)
    elif bad:
        why = "%d non-finite coords" % bad
    rows.append({"dir": name, "rung": rung, "batch": b, "arm": arm, "ok": ok, "why": why,
                 "atoms": atoms, "residues": len(res), "chains": dict(ch)})

for r in rows:
    print(("PASS" if r["ok"] else "FAIL"), r["dir"], r.get("atoms"), r.get("residues"),
          r.get("chains"), r["why"])
print("\n%d/%d pass" % (sum(1 for r in rows if r["ok"]), len(rows)))
pathlib.Path("perf/dsfix/results/bg_sanity.json").write_text(json.dumps(rows, indent=1))
