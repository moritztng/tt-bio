"""Depth REACHING the model: run the engine's own a3m parser, not a `grep -c '>'`.
`_parse_a3m_to_msa` dedups by sequence, and a tiled alignment has arrived as 35 rows before."""
import sys, yaml
sys.path.insert(0, "/home/cust-team/mthuening/abagcov/eng-main")
from tt_bio.protenix_data import _parse_a3m_to_msa
for path in sys.argv[1:]:
    d = yaml.safe_load(open(path))
    print(f"== {path}")
    tot = 0
    for s in d["sequences"]:
        p = s["protein"]; q = p["sequence"]; tot += len(q)
        a3m = open(p["msa"]).read() if "msa" in p else None
        if a3m is None:
            print(f"   chain {p['id']}: {len(q)} tokens, no msa: key (server search)"); continue
        raw = _parse_a3m_to_msa(a3m, q)
        rows = len(raw[0]) if isinstance(raw, tuple) else len(raw)
        print(f"   chain {p['id']}: {len(q)} tokens, parser yields {rows} MSA rows")
    print(f"   TOTAL {tot} tokens = 32 * {tot/32}")
