"""Fill the size-ladder TODO reasons that the counters alone decide, on one card's cells.

    python perf/sizegate/mgx/fill_reasons.py [card]      (default tt-galaxy-wh-l)

Only the patterns below, each read straight off the code that increments the counter.
Everything else is printed and left as a TODO: a reason is evidence only when someone read the
clause.

SDPA_WIDE_K, off where SDPA_K_CHUNK_STATS is incremented in `tenstorrent._tri_att_sdpa_at`:

- served 0, declined 0: both counter sites sit inside the `len(k_chunks) > 1` block (and the
  above-cap fused route, which counts only a serve). Zero on both means no call entered them,
  i.e. the shipped k_chunk divides the padded length and there is no wider dividing pick.
- served 0, declined > 0: the block was entered and no wider dividing (q, k) pair ran, so every
  call took the shipped pick. That is the lever-off path, so it costs nothing against
  TT_BIO_SDPA_WIDE_K=0.

TRANSITION_H_CHUNK on the Wormhole Galaxy above 384 tokens, served 0: the counter scores served
only when the height comes out above TRANSITION_H_CHUNK_SIZE=16, and on the small grid nothing
can lift it above its base, which is 16 once W > 384. Model-independent, so it is filled for any
model rather than re-read per recording.

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

TRANSITION_WH_NO_RAISE = (
    "on Wormhole the height cannot exceed its base at this width, whatever the channel. Served "
    "means a height above TRANSITION_H_CHUNK_SIZE=16, and the base is 32 only when W <= 384 "
    "(tenstorrent.TRANSITION_H_CHUNK_BIG_MAX_W). Every small-grid clause after that can only "
    "lower it: the ratio is min(1, ...), the 256 < c <= 384 clause is bounded by "
    "min(_base_h, ...), and the per-core L1 row cap is a min. The raise above the base is the "
    "Blackhole elif, so this board reads 0.0 above 384 tokens on every model and the lever has "
    "no Wormhole raise to decline")
WH_CARD = "tt-galaxy-wh-l"


def _counts(c: dict) -> str:
    rej = sorted((c.get("rejects") or {}).items(), key=lambda kv: -kv[1])
    return f"declines all {c.get('declined')} calls on {', '.join(f'{k} x{v}' for k, v in rej)}"


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
                    elif (flag == "TRANSITION_H_CHUNK" and card == WH_CARD
                          and int(rung) > 384 and c.get("served") == 0):
                        c["reason"] = f"{_counts(c)}: {TRANSITION_WH_NO_RAISE}"
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
