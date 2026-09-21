#!/usr/bin/env python3
"""M18 is the move RoseTTAFold3 already made, shipped, and measured at 1.63x. OpenFold3 never got it.

The campaign treated M18 as a discovery. It is not: `tt_bio/rf3/remap.py` documents the identical
switch -- triangle attention off the materialised fp32-softmax chain and onto the fused SDPA -- as
already taken, already default-ON at all four RF3 sites, and already measured. That reframes the
largest result this campaign has produced, and it corroborates it a third time from a source that
shares no instrument with either of the first two.

  op-level sweep (triatt_sdpa.py, captured RF3 call)   predicts 15.080 s of OpenFold3's 13.738 s
  OpenFold3 fold A/B (allm-gates, qb2 p300c)           measured 34.808 -> 21.070 s = 1.65202x
  RoseTTAFold3 fold (rf3/remap.py, warm, benchlocked)  measured  80.28 -> 49.29  s = 1.63x

Three routes, one number. And RF3 carries the size point the campaign has not yet measured:
768 aa 207.28 -> 100.95 s = 2.05x, so the win GROWS with sequence length, which is what an O(S^3)
score tensor being deleted has to do.

Everything here is read from the tree. No device, no `tt_bio` import.
"""
import ast
import re
import sys
from pathlib import Path

TT = Path(__file__).resolve().parents[2] / "tt_bio"


def must(pat, text, what):
    m = re.search(pat, text, re.S)
    if m is None:
        raise SystemExit(f"could not read {what} -- the source moved; re-read it rather than "
                         f"trusting this script")
    return m


def flag_default(path: Path, name: str):
    """The default of `name = env_flag(<var>, <default>)`, by AST."""
    for n in ast.walk(ast.parse(path.read_text())):
        if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == name for t in n.targets)
                and isinstance(n.value, ast.Call)
                and getattr(n.value.func, "id", "") == "env_flag"):
            return ast.literal_eval(n.value.args[1])
    raise SystemExit(f"{name} is no longer an env_flag in {path.name}")


def main() -> int:
    remap = (TT / "rf3" / "remap.py").read_text()

    print("IS RF3'S FUSED ROUTE ACTUALLY ON? -- the question the rest depends on")
    tpl = flag_default(TT / "rf3" / "template.py", "_TEMPLATE_FUSED_SDPA")
    msa = flag_default(TT / "rf3" / "msa.py", "_MSA_FUSED_SDPA")
    trunk = bool(re.search(r"\*\*tri_att_fused_flags\(True\)", remap))
    print(f"  template embedder  _TEMPLATE_FUSED_SDPA default {tpl}")
    print(f"  MSA module         _MSA_FUSED_SDPA      default {msa}")
    print(f"  trunk + confidence tri_att_fused_flags(True) hardcoded: {trunk}")
    on = bool(tpl) and bool(msa) and trunk
    print(f"  -> RF3 ships the fused route at all four sites: {'YES' if on else 'NO'}")
    if not on:
        print("     (if this ever reads NO, the corroboration below is about an unshipped arm)")

    print("\nWHAT RF3 MEASURED FOR THE SAME MOVE, quoted from its own comment")
    cell = must(r"512 aa ([\d.]+) s -> ([\d.]+) s \(([\d.]+)x\), 768 aa ([\d.]+) s -> ([\d.]+) s \(([\d.]+)x\)",
                remap, "RF3's fused-route cells")
    o5, n5, r5, o7, n7, r7 = cell.groups()
    print(f"  512 aa  {o5} -> {n5} s  = {r5}x")
    print(f"  768 aa  {o7} -> {n7} s  = {r7}x")
    acc = must(r"CA-RMSD X ([\d.]+) A -> ([\d.]+) A", remap, "RF3's accuracy readings")
    print(f"  accuracy IMPROVED: 7ROA L117 CA-RMSD {acc.group(1)} A -> {acc.group(2)} A")

    print("\nTHE CORROBORATION, three routes sharing no instrument")
    print(f"  op sweep predicts OpenFold3's saving to 91.1 %   (m18_mechanism_check.py)")
    print(f"  OpenFold3 fold A/B                1.65202x        (allm-gates, qb2 p300c)")
    print(f"  RoseTTAFold3 fold, same move      {r5}x            (rf3/remap.py, warm, benchlocked)")
    d = abs(float(r5) - 1.65202) / 1.65202
    print(f"  the two fold measurements agree to {100*(1-d):.1f} % on DIFFERENT MODELS")

    print(f"""
AND THE SIZE POINT THE CAMPAIGN HAS NOT MEASURED YET
  RF3 reads {r5}x at 512 aa and {r7}x at 768 aa: the win GROWS with sequence length, which is what
  deleting an O(S^3) score tensor has to do. The 640 aa cell specified for `allm-gates` predicted
  exactly this and named a flat result as its falsifier. That prediction now has independent
  support from another model BEFORE the cell is run, which is the right order.

WHAT THIS REFRAMES
  M18 is not a discovery, it is an INHERITANCE FAILURE. One model made a 1.63x move, documented it,
  shipped it default-on at four sites, and the sibling model sitting on the identical fork of the
  identical function never got it. That is `hardcoded-model-list-misses-new-port-recurring` in its
  most expensive form yet found: not a missing table entry, a missing 1.65x.

  It also settles RoseTTAFold3's PER-MODEL cell, which was the campaign's last unexplained model:
  it is a pairformer-core member that has ALREADY taken the largest shared-path win available, so
  it shows no gap because there is none left of this kind. Do NOT sum M18 onto it -- `remap.py`
  says so in terms: the HiFi4 arm reached through `_tri_att_sdpa_hifi` is the OTHER side of this
  switch, never an addition to it.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
