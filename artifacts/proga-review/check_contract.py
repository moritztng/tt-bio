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

struct = run / "structures"
pred = struct if struct.is_dir() else run
say(struct.is_dir(), "structures/ exists (oracle line 63)", str(struct))
cifs = sorted(pred.glob("*.cif")) or sorted(pred.glob("**/*.cif"))
say(bool(cifs), "cif present for _parse_cif", ", ".join(p.name for p in cifs[:3]))
if cifs:
    bf = [float(l.split()[14]) for l in cifs[0].read_text().splitlines()
          if l.startswith("ATOM") and len(l.split()) > 14]
    if bf:
        say(max(bf) > 1.5, "pLDDT in B-factor is 0-100 not 0-1 (oracle divides by 100)",
            f"min={min(bf):.2f} max={max(bf):.2f}")

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
