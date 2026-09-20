#!/usr/bin/env python3
"""Flag a brief whose #DISPATCH header says it wants a card while its body says it needs none.

`fleet.sh` treats `card=-` as "ANY TT card" -- only the literal `card=cpu` takes the cardless
path. The failure is silent and reads as capacity ("no free card on any of [pc]"), so the natural
response is to wait or repin the host, both wrong. This has now bitten three times:
2026-08-22 (a control-plane row), 2026-09-07 (3 days of deferrals on a brief whose own body said
`card=cpu` while its header said `card=-`), and 2026-09-20 (mine, two rows, same character).

The memory entry for it has twice recorded "candidate cleanup: flag a brief whose body says it
needs no card while its header disagrees". This is that check.

usage: assert_dispatch_card_token.py <brief.txt> [...]   (no args -> every of3t-*.txt)
"""
import glob
import re
import sys

WS = "/home/moritz/.coworker/workstreams"

# SCOPE, deliberately narrow. My first version matched prose like "do NOT open a Tenstorrent
# device" and flagged 12 of ~30 rows -- including of3t-foldab, which ran 9 seeds x 2 arms of
# folds on qb2-card0 and plainly needed its card. That phrase appears in card-USING briefs too,
# as an instruction about how to use the harness rather than a statement of requirement. A guard
# that is wrong 40 % of the time gets ignored, which is worse than no guard, so only two signals
# survive and both are mechanical:
#
#   1. `card=-` in the header. ALWAYS wrong: it never means what an author intends, because
#      fleet.sh reads it as "any TT card". No interpretation required.
#   2. the literal `card=cpu` in the BODY while the header says something else. That is the brief
#      contradicting itself in the same vocabulary -- the exact 2026-09-07 recurrence, where the
#      body read "this task is dispatched card=cpu deliberately" one line under a card=- header.
#
# English prose about devices is NOT a signal here. It was tried and it does not separate.


def check(path):
    src = open(path).read()
    m = re.search(r"^#DISPATCH:.*$", src, flags=re.M)
    if not m:
        return [f"{path}: no #DISPATCH line"]
    line, body = m.group(0), src[m.end():]
    ct = re.search(r"\bcard=(\S+)", line)
    if not ct:
        return [f"{path}: #DISPATCH has no card= token -- {line}"]
    card = ct.group(1)
    out = []
    if card == "-":
        out.append(f"{path}: card=- means ANY TT CARD to fleet.sh, never 'none'. Use card=cpu "
                   f"if this row needs no device, or name a card. -- {line}")
    elif card != "cpu" and "card=cpu" in body:
        out.append(f"{path}: the body says 'card=cpu' but the header says card={card}. "
                   f"fleet.sh obeys the header. -- {line}")
    return out


def main(argv):
    paths = argv[1:] or sorted(glob.glob(f"{WS}/of3t-*.txt"))
    bad = [b for p in paths for b in check(p)]
    for b in bad:
        print("FAIL", b)
    if bad:
        return 1
    print(f"OK {len(paths)} brief(s): no #DISPATCH card token contradicts its own body")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
