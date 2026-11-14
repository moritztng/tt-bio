"""Device-free pre-flight: does the bench MSA cache cover every protein chain of the four
BH timing complexes? --msa_cache_only turns a miss into a hard error mid-window, and two
four-card windows have already been lost, so this is checked before the window opens."""
import hashlib
import sys
from pathlib import Path

import yaml

MSA = Path(sys.argv[1])
ok = True
for cif in sys.argv[2:]:
    spec = yaml.safe_load(Path(cif).read_text())
    seqs = []
    for ent in spec.get("sequences", []):
        for kind, body in ent.items():
            if kind == "protein" and isinstance(body, dict) and body.get("sequence"):
                seqs.append((body.get("id"), body["sequence"], body.get("msa")))
    miss = []
    for cid, s, pinned in seqs:
        if pinned:  # msa: empty or an explicit a3m path is never looked up in the cache
            continue
        h = hashlib.sha256(s.encode()).hexdigest()[:16]
        p = MSA / f"{h}.a3m"
        if not (p.exists() and p.stat().st_size > 0):
            miss.append(f"chain {cid} len {len(s)} -> {h}.a3m")
    uniq = sorted({s for _, s, _ in seqs})
    pair = None
    if len(uniq) >= 2:
        ph = hashlib.sha256("\n".join(uniq).encode()).hexdigest()[:16]
        pair = MSA / "paired-v2" / ph
        if not pair.exists():
            miss.append(f"paired-v2/{ph}")
    name = Path(cif).stem
    if miss:
        ok = False
        print(f"{name}: MISS  " + "; ".join(miss))
    else:
        depths = []
        for cid, s, pinned in seqs:
            h = hashlib.sha256(s.encode()).hexdigest()[:16]
            n = sum(1 for ln in (MSA / f"{h}.a3m").read_text().splitlines() if ln.startswith(">"))
            depths.append(f"{cid}:{len(s)}aa/{n}")
        pd = ""
        if pair:
            rows = sorted(pair.glob("*"))
            pd = f"  paired-v2 ok ({len(rows)} files)"
        print(f"{name}: ok  " + " ".join(depths) + pd)
print("PREFLIGHT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
