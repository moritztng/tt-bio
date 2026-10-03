"""Build the comparison screen's numbers from the published benchmark file, so the booth shows
exactly what tt-bio.com/benchmarks shows and nothing typed in by hand.

    python3 demo/sc26/hardware/facts.py        # writes demo/sc26/web/app/lanes/facts.json

The file states the board and its date; its cells do not state the AICLK, so the screen says so. A
row whose model is not on this demo's screen for a licence reason is left out and says why.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "site/data/perf-512aa.json"
OUT = REPO / "demo/sc26/web/app/lanes/facts.json"
GPUS = ("h200", "b200", "a100")
#: Off this demo: its weights' licence is unresolved (state/lic/), so it is off JapanFold too.
EXCLUDED = {"protenix-v2": "weights licence unresolved"}

#: BindCraft 2 is not in perf-512aa.json. Copied from docs/bindcraft2.md, "What the release is
#: worth, measured on the shipped tree"; the test checks these strings are still in that doc.
BINDCRAFT2 = {
    "model": "BindCraft 2", "unit": "gradient round", "size": "288 tokens",
    "tt": 3.653, "h200": 0.696, "board": "one Blackhole chip of a p300c board",
    "aiclk": "median 1350 MHz, min 1300, sampled during the rounds",
    "source": "docs/bindcraft2.md, release 0.12.0",
}


def rows(d: dict) -> list[dict]:
    groups = (("fold", d["models"], "s_per_fold"), ("design", d["design"]["models"], "s_per_design"),
              ("affinity", d["affinity"]["models"], "s_per_fold"))
    out = []
    for group, models, key in groups:
        for m in models:
            if m.get("hidden") or m["id"] in EXCLUDED:
                continue
            c = m["cells"]
            if not all(c.get(p, {}).get("status") == "measured" for p in ("p150a",) + GPUS):
                continue
            out.append({"id": m["id"], "model": m["name"], "group": group,
                        "tt": c["p150a"][key], **{g: c[g][key] for g in GPUS}})
    return out


def build() -> dict:
    d = json.loads(SRC.read_text())
    plat = {p["id"]: p for p in d["platforms"]}
    return {
        "source": "tt-bio.com/benchmarks (site/data/perf-512aa.json)",
        "updated": d["updated"],
        "scope": d["scope"]["description"] + " " + d["scope"]["protocol"],
        "board": plat["p150a"]["measured_on"],
        "unit": "seconds per prediction at 512 residues, one accelerator, lower is better",
        "rows": rows(d),
        "excluded": [{"id": k, "why": v} for k, v in EXCLUDED.items()],
        "bindcraft2": BINDCRAFT2,
        "servers": [{k: plat[i].get(k) for k in ("name", "accelerators", "price_usd", "price_kind")}
                    for i in ("galaxy_bh", "dgx_b200") if i in plat],
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(OUT)
