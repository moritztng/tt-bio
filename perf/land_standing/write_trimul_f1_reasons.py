"""Write the four TRIMUL_TAIL_F1 exemption reasons the size-ladder arm still asked a human for.

boltz2 p150a 1152/1280/1408/1536. These are the rungs where the lever IS offered and declines
every one of its 560 calls, and its own reject counter names the clause: `k_tiles=4`. So this is
not a judgement about a lever that went dark, it is the same structural refusal already written
against p150a 256 and p300c 256 -- F1_BLOCK_KEYS allow-lists only (8, 8), and boltz-2's trimul
tail weight resolves 4 K tiles from c_hidden, which does not move with N.

What makes the four rungs LOOK different from 512-1024 on the same card, where the counter reads
0 declined, is not this clause at all: there the gate is never offered, because `fused_tail`'s
only call site (tenstorrent.py:7048) is guarded by `g_out_fused is None` and the gate was fused
into the in-projection. Those rungs already carry that reason. Above 1024 g_out stops being
fused, F1 is offered for the first time, and the k_tiles refusal becomes visible as a decline.
Same inert lever, two different counter signatures, one boundary between them.
"""
import json
import pathlib
import sys

sys.path.insert(0, "scripts")
import release_gate as rg

FRAG = pathlib.Path("docs/size_ladder_baseline.d/boltz2.json")
CARD = "p150a"
RUNGS = ("1152", "1280", "1408", "1536")
WHY = (
    "declines every call on k_tiles=4: F1_BLOCK_KEYS allow-lists only (8, 8) and boltz-2's trimul "
    "tail weight resolves 4 K tiles from c_hidden, not from N, so F1 is inert on this model at "
    "every size. It reads 0 declined at 512-1024 on this card only because it is not offered "
    "there -- `fused_tail`'s one call site (tenstorrent.py:7048) is guarded by `g_out_fused is "
    "None` and the gate rides the in-projection at those rungs. Above 1024 g_out stops being "
    "fused, so the same inert lever becomes visible as a decline rather than as a dark cell. "
    "Falsifier: widening F1_BLOCK_KEYS to cover (4, *) makes served > 0 here and nowhere below "
    "1152, which is the lever working rather than drift")

data = json.loads(FRAG.read_text())
levers = data["cards"][CARD]["models"]["boltz2"]["levers"]
wrote = []
for rung in RUNGS:
    e = levers[rung]["TRIMUL_TAIL_F1"]
    assert rg._size_ladder_dark(e), (rung, "not dark -- refusing to annotate")
    old = str(e.get("reason") or "")
    assert not old or old.startswith("TODO"), (rung, "already has a reason: " + old)
    assert e.get("declined") == 560 and e.get("served") == 0, (rung, "counter is not 0/560")
    e["reason"] = rg._size_ladder_reason_evidence(e) + ": " + WHY
    wrote.append((rung, e["reason"][:60]))
FRAG.write_text(json.dumps(data, indent=2) + "\n")
for r, head in wrote:
    print("wrote %5s %s..." % (r, head))
print("%d entries" % len(wrote))
