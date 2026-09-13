#!/usr/bin/env python3
"""Every symbol this port brings over, compared byte-for-byte against the branch it came from.

A 3-way apply that hits no conflict can still silently keep the base side of a hunk. This is the
check for that: it does not ask whether the patch applied, it asks whether the resulting text of
each ported block equals the text on the source branch.
"""
import sys


def block(path, header, stop="\nclass "):
    s = open(path).read()
    i = s.index(header)
    j = s.index(stop, i + 10)
    return s[i:j]


TT = [("def free(", "\nclass "), ("class StageWall:", "\ndef "),
      ("def pair_gather(", "\nclass "), ("class RelPosGather:", "\nclass "),
      ("class ConfidencePairDevice:", "\nclass "),
      ("class ConfidenceHeadsDevice:", "\nclass ")]
B2 = [("class ConfidenceHeads(nn.Module):", "\nclass "),
      ("class ConfidenceModule(nn.Module):", "\nclass "),
      ("def compute_ptms(", "\ndef "),
      ("def tm_function(", "\ndef "),
      ("class RelativePositionEncoder(Module):", "\nclass ")]

bad = 0
for path, ref, names in (("tt_bio/tenstorrent.py", "/tmp/ptm_tt.py", TT),
                         ("tt_bio/boltz2.py", "/tmp/ptm_b2.py", B2)):
    for h, stop in names:
        try:
            a = block(path, h, stop)
            b = block(ref, h, stop)
        except ValueError:
            print("%-42s NOT FOUND in one of the trees" % h)
            bad += 1
            continue
        if a == b:
            print("%-42s SAME (%d bytes)" % (h, len(a)))
        else:
            print("%-42s DIFFERS: %d bytes here, %d on branch" % (h, len(a), len(b)))
            bad += 1
sys.exit(1 if bad else 0)
