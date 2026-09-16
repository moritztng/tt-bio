import json, pathlib, sys
ROOT = pathlib.Path("/home/ttuser/.coworker/wt/tt-bio-sizeladder-p300c-refresh")
for m in sys.argv[1:]:
    d = json.loads((ROOT / f"docs/size_ladder_baseline.d/{m}.json").read_text())
    e = d["cards"]["p300c"]["models"][m]
    print("=" * 30, m)
    for rung in sorted(e["levers"], key=int):
        for fl, c in e["levers"][rung].items():
            r = (c.get("reason") or "")
            if r.startswith("TODO"):
                print("  %s %s: resolved=%s served=%s declined=%s how=%s rejects=%s\n     %s"
                      % (rung, fl, c.get("resolved"), c.get("served"), c.get("declined"),
                         c.get("how"), sorted(c.get("rejects") or {}), r))
