"""Hand-check control: re-find a named sample of ledger rows in their sources by a DIFFERENT route.

``corpus.py`` matches a whitespace-normalised quote. This script does not use that matcher at all.
It greps the raw source file line by line for the row's ratio string and prints the line it lands
on, with its line number, so the match can be read rather than trusted. If the extractor were
inventing rows, the ratio would not be there to find.

    python3 perf/c10_lever_corpus/handcheck.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from corpus import DEFAULT_STATE_ROOT, REPO, resolve  # noqa: E402

SAMPLE = [
    "TT_BIO_DEVICE_CONDITIONING",
    "TT_BIO_ATOM_SHIFT_GATHER",
    "trunk byte levers, Blackhole fold",
    "TT_BIO_TRIMUL_GP_BANK_SPLIT",
    "_FP32_SOFTMAX_L1_GRID -> live grid",
    "TT_BIO_SDPA_FUSED_LARGE_S (above-cap fused triangle-attention SDPA)",
]


def main(state_root: Path = DEFAULT_STATE_ROOT) -> int:
    src = {r["name"]: r for r in json.loads((HERE / "ledger_src.json").read_text())}
    bad = 0
    for name in SAMPLE:
        row = src[name]
        needle = row.get("ratio_text")
        path = resolve(row["evidence"][0]["path"], REPO, state_root)
        hits = [(i, ln.rstrip()) for i, ln in enumerate(path.read_text(errors="replace").split("\n"), 1)
                if needle and needle in ln]
        print(f"\n== {name}\n   ratio {needle}  source {row['evidence'][0]['path']}")
        if not hits:
            print("   NOT FOUND — this row would be a fabrication")
            bad += 1
            continue
        for i, ln in hits[:3]:
            print(f"   line {i}: {ln[:150]}")
    print(f"\n{len(SAMPLE) - bad} of {len(SAMPLE)} sampled ratios re-found in their own source.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
