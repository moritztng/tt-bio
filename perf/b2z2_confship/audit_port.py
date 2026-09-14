#!/usr/bin/env python3
"""Every symbol this port brings over, compared byte-for-byte against the branch it came from.

A 3-way apply that hits no conflict can still silently keep the base side of a hunk. This is the
check for that: it does not ask whether the patch applied, it asks whether the resulting text of
each ported block equals the text on the source branch.
"""
import subprocess
import sys
from pathlib import Path

#: The branch the port came from, read straight out of git. This used to compare against copies
#: under /tmp, which did not survive qb2's watchdog reset -- an audit that cannot run is an audit
#: nobody runs.
SRC = "85d0b1b43"


def source(path):
    return subprocess.run(["git", "show", "%s:%s" % (SRC, path)], check=True,
                          capture_output=True, text=True).stdout


def block(s, header, stop="\nclass "):
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
for path, names in (("tt_bio/tenstorrent.py", TT), ("tt_bio/boltz2.py", B2)):
    here, there = Path(path).read_text(), source(path)
    for h, stop in names:
        try:
            a = block(here, h, stop)
            b = block(there, h, stop)
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
