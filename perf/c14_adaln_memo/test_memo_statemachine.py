#!/usr/bin/env python3
"""The AdaLN `s_terms` memo state machine, run on the CPU with no device.

The question this answers is not "is the pair correct" -- a memory config does not change values
and the hit path returns the same tensors the miss path computed. It is "how many times does the
memo pay a DRAM retain, and how many times does it get read back", separately for the two callers
that reach `AdaLN(atom_level=True)`:

    Boltz-2   `Diffusion._cache_get("c")` -> `_c_reshaped`, ONE object for all 401 calls
    RF3       `rf3/atom_encoder.py:201`   -> `cw = ttnn.reshape(c, ...)`, a NEW object per call

`tenstorrent.py` cannot be imported without ttnn and a device, so the branch structure of
`s_terms` is transcribed here and the transcription is asserted against the source text below.
A divergence fails the test rather than passing silently.
"""
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "tt_bio" / "tenstorrent.py"


class Adaln:
    """The four lines of `s_terms` that decide retain-vs-recompute, and nothing else."""

    def __init__(self, atom_level=True, flag=True, eager=False):
        self.atom_level, self.flag, self.eager = atom_level, flag, eager
        self._s_memo = None
        self._s_memo_src = None
        self._s_seen = None
        self.hits = self.computes = self.retains = 0

    def s_terms(self, s):
        memo = self.atom_level and self.flag
        if memo and self._s_memo is not None and self._s_memo_src is s:
            self.hits += 1
            return self._s_memo
        if self.eager:
            store = memo
        else:
            store = memo and self._s_seen is s
            if memo:
                self._s_seen = s
            if store:
                self._s_memo = None
                self._s_memo_src = None
        self.computes += 1
        pair = ("s_scale", "s_bias", id(s))
        if store:
            self.retains += 1
            self._s_memo = pair
            self._s_memo_src = s
            return self._s_memo
        return pair

    # `__call__` deallocates the pair unless the memo owns it.
    def call(self, s):
        pair = self.s_terms(s)
        return self._s_memo is not pair          # True == this call deallocated the pair


def run(n, fresh_s):
    a = Adaln()
    obj = object()
    freed = 0
    for _ in range(n):
        freed += a.call(object() if fresh_s else obj)
    return a, freed


def main():
    fail = []

    # 1. The transcription matches the source.
    src = SRC.read_text()
    for needle in ("store = memo and self._s_seen is s",
                   "if memo and self._s_memo is not None and self._s_memo_src is s:",
                   "if store:",
                   "self._s_seen = None",
                   "if _B2_ADALN_MEMO_EAGER:",
                   'env_flag("TT_BIO_ADALN_MEMO_EAGER", False)'):
        if needle not in src:
            fail.append(f"source no longer contains {needle!r}: transcription is stale")

    # 2. Boltz-2: one `s` for 401 calls. Retain exactly once, hit the rest, never leak.
    b, freed = run(401, fresh_s=False)
    if (b.computes, b.retains, b.hits) != (2, 1, 399):
        fail.append(f"boltz-2 401 calls: computes/retains/hits = "
                    f"{b.computes}/{b.retains}/{b.hits}, want 2/1/399")
    if freed != 1:
        fail.append(f"boltz-2: {freed} calls deallocated their pair, want exactly 1 (call 1)")

    # 3. RF3: a fresh `s` per call. NEVER retain, never hit, always free.
    r, freed = run(401, fresh_s=True)
    if (r.computes, r.retains, r.hits) != (401, 0, 0):
        fail.append(f"rf3 401 calls: computes/retains/hits = "
                    f"{r.computes}/{r.retains}/{r.hits}, want 401/0/0")
    if freed != 401:
        fail.append(f"rf3: {freed} of 401 calls freed their pair, want 401 -- a retained pair "
                    f"nothing reads is the defect this change removes")
    if r._s_memo is not None:
        fail.append("rf3: a pair is still retained after 401 calls")

    # 4. At most one pair is held when `s` changes after a memo was stored.
    a = Adaln()
    o1, o2 = object(), object()
    a.call(o1); a.call(o1); a.call(o1)          # stored on 2, hit on 3
    held = a._s_memo
    a.call(o2); a.call(o2)                      # a different `s` takes over
    if a._s_memo is held:
        fail.append("a new `s` did not displace the previous memo")
    if a._s_memo_src is not o2:
        fail.append("memo src did not follow the new `s`")

    # 5. Flag off, and the token path (atom_level=False), retain nothing.
    for kw in ({"flag": False}, {"atom_level": False}):
        a = Adaln(**kw)
        obj = object()
        for _ in range(10):
            a.call(obj)
        if (a.retains, a.hits) != (0, 0):
            fail.append(f"{kw}: retains/hits = {a.retains}/{a.hits}, want 0/0")

    # 6. The negative control: the OLD code must FAIL test 3, or the test proves nothing.
    class Old(Adaln):
        def s_terms(self, s):
            memo = self.atom_level and self.flag
            if memo and self._s_memo is not None and self._s_memo_src is s:
                self.hits += 1
                return self._s_memo
            self.computes += 1
            pair = ("s_scale", "s_bias", id(s))
            if memo:
                self.retains += 1
                self._s_memo, self._s_memo_src = pair, s
                return self._s_memo
            return pair

    o = Old()
    for _ in range(401):
        o.call(object())
    if o.retains != 401:
        fail.append(f"negative control: old code retained {o.retains}/401, expected 401 -- the "
                    f"control does not reproduce the defect, so test 3 is not evidence")

    # 7. The measurement control (`TT_BIO_ADALN_MEMO_EAGER=1`) must reproduce the OLD code call
    #    for call, on both callers. If it does not, the fold A/B's base arm is not the behaviour
    #    the change claims to beat and the whole measurement is against a straw arm.
    for label, fresh in (("boltz-2", False), ("rf3", True)):
        eager, old = Adaln(eager=True), Old()
        obj = object()
        for _ in range(401):
            arg = object() if fresh else obj
            eager.call(arg)
            old.call(arg)
        got = (eager.computes, eager.retains, eager.hits)
        want = (old.computes, old.retains, old.hits)
        if got != want:
            fail.append(f"eager control on {label}: computes/retains/hits {got} != old code {want}")
    if Adaln(eager=True).eager is not True:
        fail.append("the eager arm is not selectable")

    for x in fail:
        print(f"FAIL: {x}")
    if fail:
        return 1
    print(f"OK: boltz-2 401 calls -> 2 computes, 1 retain, 399 hits")
    print(f"OK: rf3     401 calls -> 401 computes, 0 retains, 0 hits (was 401 retains, 0 hits)")
    print(f"OK: negative control reproduces the defect at {o.retains}/401 retains")
    print("OK: the eager measurement control reproduces the old code on both callers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
