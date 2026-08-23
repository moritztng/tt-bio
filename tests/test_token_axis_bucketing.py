"""The token axis of every shipped model is a multiple of 32, or someone owns the reason it is not.

No device, no weights. `ttnn.TILE_LAYOUT` pads physically to 32 while the logical shape stays
ragged, and the fused SDPA admits those padded key columns at a bias of zero -- 71-76x the fp64
reference at any ragged length. Every older port bucketed for kernel-recompilation reasons and was
immune to that as a side effect; RF3 did not inherit the convention and ran 117 tokens raw. So the
convention cannot stay a convention. See tt_bio/token_axis.py and PLAYBOOKS.md §MODEL 2b.

The check that matters most is the first one: a model added to a CLI --model choice without a
bucketing decision fails here, which is exactly the way RF3 got in.
"""
import importlib
import sys

from tt_bio import token_axis as TA


def _fail(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    return ok


def check_every_shipped_model_declared():
    shipped = TA.shipped_models()
    declared = set(TA.TOKEN_AXIS)
    missing = sorted(shipped - declared)
    extra = sorted(declared - shipped)
    ok = _fail(not missing, "every shipped --model is declared in TOKEN_AXIS"
               + ("" if not missing else "; undeclared: " + ", ".join(missing)))
    return ok and _fail(not extra, "TOKEN_AXIS declares no model the CLI does not ship"
                        + ("" if not extra else "; stale: " + ", ".join(extra)))


def check_statuses_are_known():
    bad = sorted(n for n, r in TA.TOKEN_AXIS.items() if r[0] not in TA.STATUSES)
    return _fail(not bad, "every status is one of " + "/".join(TA.STATUSES)
                 + ("" if not bad else "; bad: " + ", ".join(bad)))


def check_unresolved_rows_have_an_owner():
    """PARTIAL, EXPOSED and UNCENSUSED are debts, not resting states: each names the task that
    owes it. PARTIAL counts because its cost is a fused kernel going dark and its risk is that one
    refactor of the safe primitive turns it into EXPOSED."""
    bad = sorted(n for n, r in TA.TOKEN_AXIS.items()
                 if r[0] in TA.NEEDS_OWNER and not (r[3] or "").strip())
    return _fail(not bad, "every exposed/uncensused model names an owning task"
                 + ("" if not bad else "; ownerless: " + ", ".join(bad)))


def check_immune_rows_carry_a_reason():
    bad = sorted(n for n, r in TA.TOKEN_AXIS.items()
                 if r[0] == TA.IMMUNE and len((r[3] or "").strip()) < 20)
    return _fail(not bad, "every immune model says WHY it is immune"
                 + ("" if not bad else "; unexplained: " + ", ".join(bad)))


def check_bucketed_multiples_are_tile_multiples():
    bad = []
    for n, r in TA.TOKEN_AXIS.items():
        if r[0] not in TA.NEEDS_MULTIPLE:
            continue
        m = r[1]
        if not isinstance(m, int) or m <= 0 or m % TA.TILE:
            bad.append(f"{n}={m!r}")
    return _fail(not bad, f"every declared bucket multiple is a positive multiple of {TA.TILE}"
                 + ("" if not bad else "; bad: " + ", ".join(bad)))


def check_declared_multiples_match_the_live_constants():
    """The table claims values that live in five other modules. Import and compare them."""
    bad = []
    for (mod_name, attr), claimed in TA.LIVE_MULTIPLES.items():
        try:
            live = getattr(importlib.import_module(mod_name), attr)
        except Exception as exc:                     # a renamed constant is a real failure
            bad.append(f"{mod_name}.{attr} unreadable ({type(exc).__name__})")
            continue
        if live != claimed:
            bad.append(f"{mod_name}.{attr} is {live}, table says {claimed}")
        elif live % TA.TILE:
            bad.append(f"{mod_name}.{attr} = {live} is not a multiple of {TA.TILE}")
    return _fail(not bad, "the live pad constants match the table and divide "
                 + str(TA.TILE) + ("" if not bad else "; " + "; ".join(bad)))


CHECKS = (
    check_every_shipped_model_declared,
    check_statuses_are_known,
    check_unresolved_rows_have_an_owner,
    check_immune_rows_carry_a_reason,
    check_bucketed_multiples_are_tile_multiples,
    check_declared_multiples_match_the_live_constants,
)


def run_checks() -> bool:
    ok = True
    for c in CHECKS:
        ok = c() and ok
    tally = {st: sum(1 for r in TA.TOKEN_AXIS.values() if r[0] == st) for st in TA.STATUSES}
    print("\n%d shipped models declared: " % len(TA.TOKEN_AXIS)
          + ", ".join(f"{v} {k}" for k, v in tally.items() if v))
    print("ALL OK" if ok else "FAILURES")
    return ok


def test_token_axis_bucketing():
    assert run_checks(), "a check above printed FAIL"


if __name__ == "__main__":
    sys.exit(0 if run_checks() else 1)
