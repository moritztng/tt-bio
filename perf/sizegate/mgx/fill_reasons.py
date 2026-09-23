"""Fill the size-ladder TODO reasons that the counters alone decide, on one card's cells.

    python perf/sizegate/mgx/fill_reasons.py [card]      (default tt-galaxy-wh-l)

Only two patterns, both read straight off where SDPA_K_CHUNK_STATS is incremented in
`tenstorrent._tri_att_sdpa_at`. Everything else is printed and left as a TODO: a reason is
evidence only when someone read the clause.

- served 0, declined 0: both counter sites sit inside the `len(k_chunks) > 1` block (and the
  above-cap fused route, which counts only a serve). Zero on both means no call entered them,
  i.e. the shipped k_chunk divides the padded length and there is no wider dividing pick.
- served 0, declined > 0: the block was entered and no wider dividing (q, k) pair ran, so every
  call took the shipped pick. That is the lever-off path, so it costs nothing against
  TT_BIO_SDPA_WIDE_K=0.

A lever resolved "none" (a per-site flag with no site on) is off, not dark, so its TODO is
dropped rather than answered.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

WIDE_K_UNREACHED = (
    "structural, not size-specific: both SDPA_K_CHUNK_STATS sites are inside "
    "`_tri_att_sdpa_at`'s `len(k_chunks) > 1` block, and served 0 declined 0 means no call "
    "entered it: the shipped k_chunk divides the padded length at this rung, so there is no "
    "wider dividing pick to offer")
WIDE_K_FELL_BACK = (
    "the shipped k_chunk does not divide the padded length here, and no wider dividing (q, k) "
    "pair ran (the fused kernel declined and the stock op refused on L1, or no q_chunk divides "
    "the padded length), so every call took the shipped pick: the TT_BIO_SDPA_WIDE_K=0 path")


def fill(card: str) -> list:
    left = []
    for f in sorted((ROOT / "docs" / "size_ladder_baseline.d").glob("*.json")):
        d = json.loads(f.read_text())
        blk = (d.get("cards") or {}).get(card)
        if not blk:
            continue
        changed = False
        for model, e in blk["models"].items():
            for rung, lv in (e.get("levers") or {}).items():
                for flag, c in lv.items():
                    if not str(c.get("reason", "")).startswith("TODO"):
                        continue
                    if c.get("resolved") == "none":
                        del c["reason"]
                    elif flag == "SDPA_WIDE_K" and c.get("served") == 0:
                        c["reason"] = WIDE_K_FELL_BACK if c.get("declined") else WIDE_K_UNREACHED
                    else:
                        left.append(f"{model}/{rung} {flag}: {c['reason']}")
                        continue
                    changed = True
        if changed:
            f.write_text(json.dumps(d, indent=2) + "\n")
    return left


if __name__ == "__main__":
    for line in fill(sys.argv[1] if len(sys.argv) > 1 else "tt-galaxy-wh-l"):
        print("TODO left:", line)
