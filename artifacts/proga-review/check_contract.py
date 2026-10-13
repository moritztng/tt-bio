"""Check tt-bio's real output against every key, glob and type proga's readers consume.

Mirrors ttbio_oracle.py line by line (we do not import proga): _parse_pae_ttbio (pae npz +
_pae.json 'absent.pae' sidecar), _parse_confidence_ttbio (results.json id/status/iptm/
pair_chains_iptm/chains_ptm, all indexed with int(key)), and BoltzOracle._parse_cif's inputs
(structures/ dir, cif, B-factor pLDDT on 0-100). Usage: check_contract.py <run_dir> <name>
"""
import json, sys
from pathlib import Path
import numpy as np

run = Path(sys.argv[1]); name = sys.argv[2]; ok = True
def say(good, what, detail=""):
    global ok
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {what}{(': ' + detail) if detail else ''}")

# oracle line 63-64: run_dir/structures if it exists, else BoltzOracle._find_pred_dir's
# **/predictions/* then "parent of the first **/*.cif" fallback.
struct = run / "structures"
if struct.is_dir():
    pred, how = struct, "run_dir/structures"
else:
    c = next((c for c in run.glob("**/predictions/*") if c.is_dir() and any(c.glob("*.cif"))), None)
    cif0 = next(run.glob("**/*.cif"), None)
    pred = c or (cif0.parent if cif0 else run)
    how = "_find_pred_dir fallback -> " + str(pred.relative_to(run))
say(pred.is_dir() and any(pred.glob("*.cif")), "oracle resolves a pred_dir holding the cif", how)
cifs = sorted(pred.glob("*.cif"))
say(bool(cifs), "cif present for _parse_cif", ", ".join(x.name for x in cifs[:3]))
if cifs:
    import gemmi  # same reader _parse_cif uses
    st = gemmi.read_structure(str(cifs[0])); m = st[0]
    names = [ch.name for ch in m]
    say(set(["A", "B"]) <= set(names), "cif chain names match the yaml ids the oracle indexes by",
        ",".join(names))
    bf = [a.b_iso for ch in m for r in ch for a in r if a.name == "CA"]
    say(bool(bf) and max(bf) > 1.5, "CA B-factor is pLDDT on 0-100 (oracle divides by 100)",
        f"n={len(bf)} min={min(bf):.1f} max={max(bf):.1f}" if bf else "no CA atoms")

p = (next(pred.glob(f"{name}*_pae.npz"), None) or next(pred.glob("*_pae.npz"), None)
     or next(pred.glob("**/*_pae.npz"), None))
say(p is not None, "*_pae.npz found by the oracle's 3-step glob", p.name if p else "NONE")
if p is not None:
    z = np.load(p)
    has = "pae" in z.files
    say(has, "npz carries key 'pae'", ",".join(z.files))
    if has:
        a = z["pae"]; a = a[0] if a.ndim == 3 else a
        say(a.ndim == 2 and a.shape[0] == a.shape[1], "pae is square after the ndim==3 squeeze", str(a.shape))
    else:
        sc = p.with_suffix(".json")
        r = (json.load(open(sc)).get("absent") or {}).get("pae", "") if sc.exists() else ""
        say(bool(r), "sidecar _pae.json carries absent.pae reason", r or "MISSING")

rj = next(pred.glob("results.json"), None) or next(run.glob("**/results.json"), None)
say(rj is not None, "results.json found", str(rj))
if rj is not None:
    data = json.load(open(rj)); recs = data if isinstance(data, list) else [data]
    rec = next((r for r in recs if r.get("id") == name), recs[0])
    say(any(r.get("id") == name for r in recs), "a row has id == the yaml stem",
        ",".join(str(r.get("id")) for r in recs))
    say(rec.get("status") in (None, "ok"), "status is 'ok' or absent", repr(rec.get("status")))
    say("iptm" in rec or "confidence_score" in rec, "iptm (or confidence_score fallback) present",
        f"iptm={rec.get('iptm')}")
    for key in ("pair_chains_iptm", "chains_ptm"):
        d = rec.get(key)
        say(isinstance(d, dict) and bool(d), f"{key} present and non-empty", json.dumps(d)[:160])
        if isinstance(d, dict):
            keys = list(d) + [k for v in d.values() if isinstance(v, dict) for k in v]
            bad = [k for k in keys if not str(k).lstrip("-").isdigit()]
            say(not bad, f"{key} keys survive the oracle's int(k)", f"non-integer keys: {bad}" if bad else f"keys={list(d)}")
print("CONTRACT:", "HOLDS" if ok else "BROKEN", "for", run)
sys.exit(0 if ok else 1)
