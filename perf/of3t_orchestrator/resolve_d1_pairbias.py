#!/usr/bin/env python3
"""Resolve `of3t-d1-pairbias` against the composition: the ROW's side, on Moritz's ruling.

Two rows reached opposite conclusions on the same lines, and this is not a stale-base artifact --
it is a real disagreement about what OpenFold3 should compute:

  of3t-pairbias  (concluded earlier) `scale_pair_bias=False`, with its reasoning in the comment:
                 "Do NOT flip the OF3 trunk default on this evidence" -- over nine seeds on 1UBQ
                 the corrected bias buys 0.050 A of best-of-5 and costs 0.463 A on the structure a
                 user receives, against that target's own 0.324 A seed floor. It named its own
                 release condition: "the fix must ship with a selector fix or not at all".
  of3t-d1-pairbias (GO, this pass) `scale_pair_bias=True` everywhere, on Moritz's ask-9629 ruling.

The row's side wins, and the basis is written down so it can be argued with rather than guessed at:

  1. **Moritz decided it.** `state/ask-9629-decision.md`: *"FIX IT. Match upstream, everywhere."*
     He set one reopen condition -- *"if the repair turns out to be reliably worse across targets
     and seeds -- not one target"*.
  2. **That condition was measured and is not met.** 4 targets, 6 seeds each, 48 folds: 10 of 24
     paired folds regress, two-sided sign test p = 0.541, pooled median negative. The single
     regressing target is 1UBQ at 0.12x-0.24x its own seed floor on two of six seeds.
  3. **1UBQ is the earlier reading's own target**, which makes the 0.463 A the overfitting Moritz
     named rather than a counter-argument to it.
  4. **The earlier row's release condition is now satisfied.** It said the loss is the confidence
     head preferring the looser sample mode (D10), so the fix must ship with a selector fix. The
     unified ranking rule (D10/D24) is in this composition.

The superseded reasoning is NOT deleted -- it stays in the file as the record of why the campaign
held the default for as long as it did, which is the same rule the ledger follows.

`perf/of3t_pairbias/attn_f64.py` is add/add and HEAD keeps it: `of3t-pairbias` owns that namespace
and wrote that file first. Every DATA artifact the newer row produced lands cleanly beside it.

Usage: resolve_d1_pairbias.py <file>  ->  0 resolved, 2 not a shape this knows (caller must stop).
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

MARK = re.compile(r"^<<<<<<< HEAD\n(.*?)^=======\n(.*?)^>>>>>>> origin/wk/of3t-d1-pairbias\n",
                  re.S | re.M)


def resolve(text, keep):
    """keep='theirs' -> the row's side; 'ours' -> HEAD's."""
    def pick(m):
        return m.group(1) if keep == "ours" else m.group(2)
    out, n = MARK.subn(pick, text)
    return out, n


def main(argv):
    if len(argv) < 2:
        return 2
    path = pathlib.Path(argv[1])
    name = str(path)
    if name.endswith("perf/of3t_pairbias/attn_f64.py"):
        keep = "ours"          # the namespace owner's file
    elif name.endswith("tt_bio/openfold3_trunk.py"):
        # NOT a side pick. HEAD's hunk carries two OTHER rows' declared contributions --
        # of3t-foldab's `TT_BIO_OF3_TRI_END_BIAS_FOLLOWS_PAIR` measurement lever and
        # of3t-trunkcliff's pair-bias note -- and taking the row's side wholesale drops both,
        # which is what the co-edit assertion caught on the first attempt. So: keep HEAD, and
        # apply the ONE token the decision actually changes.
        keep = "ours"
    elif name.endswith("tt_bio/tenstorrent.py"):
        keep = "theirs"        # a reworded comment, same content, no other row in the hunk
    else:
        return 2
    text = path.read_text()
    out, n = resolve(text, keep)
    if n == 0:
        print(f"{path}: no of3t-d1-pairbias conflict marker found", file=sys.stderr)
        return 2
    if "<<<<<<<" in out or ">>>>>>>" in out:
        print(f"{path}: a conflict from another row remains", file=sys.stderr)
        return 2
    if name.endswith(".py"):
        try:
            ast.parse(out)
        except SyntaxError as e:
            print(f"{path}: resolution does not parse ({e})", file=sys.stderr)
            return 2
    if name.endswith("tt_bio/openfold3_trunk.py"):
        # Now the one token. `scale_pair_bias=False` in the Pairformer construction becomes True:
        # that single flag is what held the token pair bias at 1/sqrt(24) = 0.204 of reference in
        # all 48 blocks of every fold served.
        flip_from = "scale_pair_bias=False, tri_att_scale_pair_bias=False"
        flip_to = "scale_pair_bias=True, tri_att_scale_pair_bias=False"
        if flip_from not in out and flip_to not in out:
            print(f"{path}: neither the pre- nor post-decision construction is here",
                  file=sys.stderr)
            return 2
        out = out.replace(flip_from, flip_to, 1)
        try:
            ast.parse(out)
        except SyntaxError as e:
            print(f"{path}: the flip does not parse ({e})", file=sys.stderr)
            return 2
        # Everything that has to survive: the decided repair, its provenance, and the two other
        # rows whose work shares this hunk.
        for needle, why in ((flip_to, "the decided repair"),
                            ("0.204 of reference", "the 0.204 provenance"),
                            ("TT_BIO_OF3_TRI_END_BIAS_FOLLOWS_PAIR", "of3t-foldab's env lever"),
                            ("folds the bias inside its own score scale",
                             "of3t-trunkcliff's pair-bias note")):
            if needle not in out:
                print(f"{path}: resolution lost {why}", file=sys.stderr)
                return 2
    path.write_text(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
