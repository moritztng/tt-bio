"""Fill the p300c esmfold2 exemption reasons this re-record left TODO, and normalise every
carried reason whose head the gate cannot regenerate.

A reason is "<evidence>: <judgement>". _size_ladder_fill_reasons only re-measures the evidence
half when the head is one of SIZE_LADDER_EVIDENCE_HEADS; any other head travels verbatim and
states counts the entry beside it no longer says. So every dark row here is rewritten with a
head the gate generates, judgement kept where one already existed and authored where the row
is new. Rung sets come from the data, not from a list, so the script cannot drift from the
ladder it annotates.
"""
import importlib.util, json, pathlib, sys

ROOT = pathlib.Path("/home/ttuser/.coworker/wt/tt-bio-sizeladder-p300c-refresh")
spec = importlib.util.spec_from_file_location("rg", ROOT / "scripts" / "release_gate.py")
rg = importlib.util.module_from_spec(spec)
sys.modules["rg"] = rg
spec.loader.exec_module(rg)

MODEL, CARD = "esmfold2", "p300c"

# Authored judgements, verified this pass against tt_bio/ on d35f9db1f and the commit that
# moved each row. Keyed by flag; a dict value picks the judgement per rung.
J = {
 "FP32_SOFTMAX_L1_GRID":
   "the counter is the L1-resident fp32 score block and ESMFold2 plans none at any size: the "
   "fp32 raw-matmul attention path is its only producer and this model folds in bf16. The "
   "resolved (8, 8) is the rectangle the constant holds, left pinned on Blackhole by "
   "_apply_grid_thresholds, not a block that ran. The row is new because 92d5a9030 gave the "
   "rectangle its own l1_blocks/l1_refused counters instead of scoring it off the bias hoist's",
 "B2_TOKEN_DIT_SDPA":
   "the counter sits on boltz-2's token-DiT attention site (AttentionPairBias.token_dit, set "
   "only by the boltz-2 diffusion transformer at tenstorrent.py:9445) and ESMFold2 runs its "
   "own TokenDiT, so the branch is never reached at any rung. The resolved state moved False "
   "-> True because fc7fed56f flipped the default, not because anything on this model changed",
 "ATOM_AXIS_BUCKET":
   "the bucket is applied on the atom-window axis inside boltz-2's DiffusionModule. ESMFold2 "
   "has no atom-level encoder and reaches no call site at any rung. The row is new because "
   "c0a876e1a added the lever and fc7fed56f made it default",
 "TRANSITION_H_CHUNK":
   "the counter is the row-block height chosen in Transition.__call__, and ESMFold2's pair "
   "transition is an ESMC SwiGLUFFN (esmfold2.py:161), not that Transition, so the lever has "
   "no call site on this model at any rung. The row is new because 7fb08268f gave the lever "
   "its first counter",
 "TRIMUL_MASK_AFTER_MOVE":
   "it unlocks the gated channel move, which is only offered once the trimul's pair tensors "
   "are in DRAM. At 256 aa they are L1-resident -- REBLOCK_PERMUTE refuses every call here on "
   "window_BufferType.L1 -- so nothing is offered at this rung; it serves every call from 512 "
   "up. Default on since 7bfa4ad19",
 "REBLOCK_PERMUTE":
   "the traffic moved to the gated channel, it did not stop. TRIMUL_MASK_AFTER_MOVE "
   "(d86a1eb5f, default on since 7bfa4ad19) moves the pair mask past the channel move, so the "
   "fused chunk+gate route is eligible and REBLOCK_PERMUTE_GATED serves the calls this counter "
   "served when the baseline was recorded. The handoff is visible across the three cards in "
   "commit order: the galaxy entry at f11e5009 reads ungated 2168 / gated 0 at 512, p150a at "
   "9ceae36d reads 16 / 2152, this card reads 0 / 2168. The wrapper is not offered, not refused",
}
# Flags whose dark rungs J is allowed to annotate. Anything else dark keeps its own judgement
# with the head regenerated.
path = ROOT / "docs" / "size_ladder_baseline.d" / f"{MODEL}.json"
data = json.loads(path.read_text())
cards = data["cards"]
assert CARD in cards, f"{CARD} missing from the fragment: {sorted(cards)}"
# The other cards must come out byte-identical: this pass re-recorded p300c only.
OTHERS = {c: json.dumps(v, sort_keys=True) for c, v in cards.items() if c != CARD}
entry = cards[CARD]["models"][MODEL]
levers = entry["levers"]
print("rungs:", sorted(levers, key=int), "runtime:", entry["runtime_s"])

def head_for(e):
    # "declines all 0 calls" is what the generator says for a lever that was never offered;
    # "never reached at this size" is the head the gate keeps for exactly that case and it
    # regenerates too.
    if not (e.get("served") or 0) and not (e.get("declined") or 0):
        return "never reached at this size"
    return rg._size_ladder_reason_evidence(e)

authored = kept = 0
for rung in sorted(levers, key=int):
    for flag, e in sorted(levers[rung].items()):
        if not rg._size_ladder_dark(e):
            continue
        head = head_for(e)
        assert head.startswith(rg.SIZE_LADDER_EVIDENCE_HEADS), head
        old = str(e.get("reason") or "")
        if flag in J:
            why, src = J[flag], "authored"
            authored += 1
        else:
            h, sep, why = old.partition(": ")
            if not (sep and h.startswith(rg.SIZE_LADDER_EVIDENCE_HEADS)):
                # No generated head, so the whole text is judgement: a reason carried across
                # cards arrives that way. It is not a stale-number defect, but the gate cannot
                # regenerate it either, so give it a head rather than leaving it opaque.
                why, sep = old, ": "
            # A reason inherited from a lower rung or another card carries a provenance tag the
            # gate itself strips before re-carrying; strip it here too so it cannot compound.
            why = rg.SIZE_LADDER_RUNG_TAG.sub('', why).replace('[carried from p150a]', '')
            why = why.replace('[carried from tt-galaxy-wh l]', '').strip()
            assert why and not old.startswith("TODO"), \
                f"{flag}@{rung}: nothing to keep in {old!r}"
            src = "kept" if h.startswith(rg.SIZE_LADDER_EVIDENCE_HEADS) else "head-added"
            kept += 1
        e["reason"] = f"{head}: {why}"
        print(f"  {flag:<28} {rung:>4} [{src}]")
        if src == "head-fixed":
            print(f"      stale head was: {h[:100]}")

left = rg._size_ladder_unreasoned(levers)
print(f"\nauthored {authored}, kept {kept}; unreasoned dark levers left:", sorted(left) or "none")
assert not left, left
for rung in levers:
    for flag, e in levers[rung].items():
        r = str(e.get("reason") or "")
        if r:
            assert r.startswith(rg.SIZE_LADDER_EVIDENCE_HEADS), f"{flag}@{rung}: {r[:60]}"
assert OTHERS == {c: json.dumps(v, sort_keys=True) for c, v in cards.items() if c != CARD}, \
    "a card other than p300c changed"
print("untouched cards verified:", sorted(OTHERS))
path.write_text(json.dumps(data, indent=2) + "\n")
print("written", path)
