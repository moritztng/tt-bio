#!/usr/bin/env python3
"""Fill site/data/perf-512aa.json cells from the BMK board, the only source the page may quote.

    python3 perf/bmk/board_to_page.py [--board PATH] [--write]

The board is state/bmk/BOARD.md, a markdown table appended to by the measuring rows:

    | cell | platform | mode | value | n / spread | clock | versions | when | command | row |

`cell` is the page row id (boltz2, nesso1, esmc-6b, ...), `platform` a page platform id (p150a, wh,
a100, h200, b200). Modes: Tenstorrent `normal` (headline) and `fast`; NVIDIA `exact` (the Anthropic
kit's exact mode, headline when present), `fast` (kit fast), `stock` (upstream code, headline when
the kit has no exact run for the row) and `off` (kit control, not published). The newest board row
for a (cell, platform, mode) wins.

A cell that receives a headline run is rebuilt from the board alone: the provenance prose, notes and
host/device split it carried described the old measurement and are dropped. Without --write this
prints old against new for every cell it would change and touches nothing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "site" / "data" / "perf-512aa.json"
BOARD = Path.home() / ".coworker" / "state" / "bmk" / "BOARD.md"
UNIT = {"models": "s_per_fold", "design": "s_per_design", "affinity": "s_per_fold", "embed": "s_per_batch"}
ALIAS = {"kit exact": "exact", "kit-exact": "exact", "kit fast": "fast", "kit-fast": "fast",
         "kit off": "off", "kit-off": "off", "--fast": "fast"}


def num(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group()) if m else None


def board_runs(path: Path) -> dict:
    runs = {}
    for line in path.read_text().splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 10 or c[0] in ("cell", "---") or set(c[0]) <= {"-"} or not c[3] or not c[8]:
            continue
        mode = ALIAS.get(c[2].lower(), c[2].lower())
        value = num(c[3])
        if value is None:
            continue
        run = OrderedDict(value=value)
        n = re.search(r"n\s*=?\s*(\d+)", c[4])
        if n:
            run["n"] = int(n.group(1))
        sp = re.search(r"(\d+(?:\.\d+)?)\s*%", c[4])
        if sp:
            run["spread_pct"] = float(sp.group(1))
        run["clock_mhz"] = num(c[5])
        run["versions"] = c[6]
        run["when"] = c[7]
        runs[(c[0].lower(), c[1].lower(), mode)] = run   # newest last wins
    return runs


def as_run(r: dict, unit: str, vendor: str) -> OrderedDict:
    out = OrderedDict([(unit, r["value"])])
    for k in ("n", "spread_pct"):
        if k in r:
            out[k] = r[k]
    if r.get("clock_mhz"):
        out["aiclk_mhz" if vendor == "tenstorrent" else "clock_mhz"] = int(round(r["clock_mhz"]))
    out["versions"] = r["versions"]
    out["when"] = r["when"]
    return out


def fill(doc: dict, runs: dict) -> list[str]:
    vendor = {p["id"]: p["vendor"] for p in doc["platforms"]}
    changes = []
    for cat, unit in UNIT.items():
        rows = doc[cat] if isinstance(doc[cat], list) else doc[cat]["models"]
        for m in rows:
            for k, old in list(m["cells"].items()):
                get = lambda mode: runs.get((m["id"], k, mode))
                v = vendor[k]
                if v == "tenstorrent":
                    head, ref, extra = get("normal"), None, {"fast": get("fast")}
                elif get("exact"):
                    head, ref, extra = get("exact"), "kit", {"fast": get("fast"), "stock": get("stock")}
                else:
                    head, ref, extra = get("stock"), "stock", {}
                if not head:
                    continue
                cell = OrderedDict(status="measured")
                cell.update(as_run(head, unit, v))
                if ref:
                    cell["ref"] = ref
                batch = [c["batch"] for c in m["cells"].values() if "batch" in c]
                if batch:
                    cell["batch"] = old.get("batch", batch[0])
                if old.get("parity"):
                    cell["parity"] = old["parity"]   # TT parity claim survives; re-check it by hand
                for name, r in extra.items():
                    if r:
                        cell[name] = as_run(r, unit, v)
                m["cells"][k] = cell
                changes.append(f"{cat}/{m['id']}/{k}: {old.get(unit, old.get('status'))} -> {cell[unit]}"
                               f"{' (' + ref + ')' if ref else ''}")
    return changes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--board", type=Path, default=BOARD)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    doc = json.loads(DATA.read_text(), object_pairs_hook=OrderedDict)
    changes = fill(doc, board_runs(a.board))
    print("\n".join(changes) or "no board row matches a page cell")
    stale = [f"{cat}/{m['id']}/{k}" for cat in UNIT
             for m in (doc[cat] if isinstance(doc[cat], list) else doc[cat]["models"])
             for k, c in m["cells"].items() if c.get("provenance")]
    if stale:
        print(f"{len(stale)} cells still carry a pre-campaign measurement: " + ", ".join(stale))
    if a.write and changes:
        DATA.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
