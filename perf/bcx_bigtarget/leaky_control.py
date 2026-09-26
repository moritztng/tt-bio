#!/usr/bin/env python3
"""The leaky control: storage groups hold their members STRONGLY again.

bcx-armtree's `27e000ff2`, on main as `a257dfbbc`, made `Tensor.shares` hold weak references so
a view and its source stop owning each other. That commit is inside the range between the tree
bcx-large measured the n=352 ceiling on (`68b7a49a0`) and main today, and main today shows no
per-block growth at all where bcx-large measured 0.0638 GB. This inverts exactly that one change
and nothing else, so a run under it attributes the flat line rather than merely coinciding with
it. bcx-armtree ran the same control on its own tree at 320 tokens; this one runs on MAIN at 352,
which is the size the campaign's ceiling is quoted at.

  python3 perf/bcx_bigtarget/leaky_control.py apply    # then run curve.py, then:
  git checkout tt_bio/autograd.py

The tree is dirty while it is applied and `A.stamp` records that, which is the point: a control
arm must be identifiable in its own artifact.
"""
import pathlib
import sys

P = pathlib.Path(__file__).resolve().parents[2] / "tt_bio" / "autograd.py"

WEAK = '''    group = a.shares if a.shares is not None else [weakref.ref(a)]
    have = _members(group)
    for t in _members(b.shares) if b.shares is not None else [b]:
        if not any(t is m for m in have):
            group.append(weakref.ref(t))
            have.append(t)
    for t in have:
        t.shares = group


def _members(group) -> list:
    """The live tensors of a storage group."""
    return [t for t in (r() for r in group) if t is not None]'''

STRONG = '''    # LEAKY CONTROL (bcx-bigtarget). The group holds its members STRONGLY, which is what every
    # tree before the weak-group fix did, bcx-large's 68b7a49a0 included. Nothing else in this
    # file differs from main, so a ceiling measured here is the tape lifetime and not some other
    # delta. Never commit this applied.
    group = a.shares if a.shares is not None else [a]
    for t in (b.shares if b.shares is not None else [b]):
        if not any(t is m for m in group):
            group.append(t)
    for t in group:
        t.shares = group


def _members(group) -> list:
    """The live tensors of a storage group. Strong members under the leaky control."""
    return list(group)'''


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    s = P.read_text()
    if mode == "apply":
        if STRONG in s:
            print("already applied")
            return 0
        if WEAK not in s:
            print("REFUSED: main's weak-group text is not where it was; re-read autograd.py")
            return 2
        P.write_text(s.replace(WEAK, STRONG))
        print("applied -- tree is now dirty; git checkout tt_bio/autograd.py when done")
        return 0
    print("applied" if STRONG in s else ("clean" if WEAK in s else "UNKNOWN"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
