"""The token axis of every shipped model is a multiple of 32, or someone owns the reason it is not.

No device, no weights. `ttnn.TILE_LAYOUT` pads physically to 32 while the logical shape stays
ragged, and the fused SDPA admits those padded key columns at a bias of zero -- 71-76x the fp64
reference at any ragged length. Every older port bucketed for kernel-recompilation reasons and was
immune to that as a side effect; RF3 did not inherit the convention and ran 117 tokens raw. So the
convention cannot stay a convention. See tt_bio/token_axis.py and PLAYBOOKS.md §MODEL 2b.

Two things fail here. A model added to a CLI --model choice without a bucketing decision, which
is exactly the way RF3 got in; and a model that HAS a decision other than "it buckets". EXPOSED,
UNCENSUSED and IMMUNE are all failures now (Moritz, 2026-08-22: "i want to have bucketing
implemented for every model"), with one allow-list entry for the row a live task owns.
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


# The one permitted unresolved row, and the task that owes it. Keyed to the owner string so the
# entry dies with the task rather than outliving it: `rf3-4x-with-accuracy-land` is measuring the
# bucket against its own per-call TT_BIO_SDPA_RAGGED_PAD and picks on the numbers. Delete this
# entry the moment it lands. Nothing else may be added -- an allow-list that grows is the
# convention this file exists to replace.
ALLOWED_UNBUCKETED = {"rf3": "rf3-4x-with-accuracy-land"}


def _unbucketed():
    """Every row not BUCKETED and not the allow-listed one, as name -> status."""
    out = {}
    for n, r in TA.TOKEN_AXIS.items():
        if r[0] == TA.BUCKETED:
            continue
        if ALLOWED_UNBUCKETED.get(n) and ALLOWED_UNBUCKETED[n] in (r[3] or ""):
            continue
        out[n] = r[0]
    return out


def check_no_unresolved_rows():
    """EXPOSED and UNCENSUSED are failures in themselves, not merely debts with a name on them.

    `check_unresolved_rows_have_an_owner` only asked that someone be blamed. That let rf3 sit
    EXPOSED and nesso1 sit UNCENSUSED across releases while the file read as green.
    """
    bad = sorted(f"{n}={s}" for n, s in _unbucketed().items()
                 if s in (TA.EXPOSED, TA.UNCENSUSED))
    return _fail(not bad, "no model is exposed or uncensused"
                 + ("" if not bad else "; " + ", ".join(bad)))


def check_immune_is_not_terminal():
    """IMMUNE is evidence about risk, not an exemption.

    An immune model is correct today because every ragged call happens to land on ttnn.softmax,
    which masks its own tail, rather than on SDPA, which does not. It still pays the kernel
    recompilation tax at every distinct length, and it is one refactor of that route away from
    EXPOSED. The IMMUNE constant stays -- it is the right word for the `why` of a row that also
    buckets -- but no row may rest on it.
    """
    bad = sorted(n for n, s in _unbucketed().items() if s == TA.IMMUNE)
    return _fail(not bad, "no model rests on IMMUNE instead of bucketing"
                 + ("" if not bad else "; " + ", ".join(bad)))


def check_every_bucketed_row_uses_the_shared_table():
    """A BUCKETED row's multiple is `token_axis.BUCKET_MULTIPLE`, so a fifth near-copy of the
    mechanism carrying its own constant fails here rather than drifting quietly."""
    bad = []
    for n, r in TA.TOKEN_AXIS.items():
        if r[0] != TA.BUCKETED:
            continue
        want = TA.BUCKET_MULTIPLE.get(n)
        if want is None:
            bad.append(f"{n} has no BUCKET_MULTIPLE entry")
        elif want != r[1]:
            bad.append(f"{n} runs {r[1]}, BUCKET_MULTIPLE says {want}")
    return _fail(not bad, "every bucketed row's multiple comes from BUCKET_MULTIPLE"
                 + ("" if not bad else "; " + "; ".join(bad)))


CHECKS = (
    check_every_shipped_model_declared,
    check_statuses_are_known,
    check_unresolved_rows_have_an_owner,
    check_immune_rows_carry_a_reason,
    check_bucketed_multiples_are_tile_multiples,
    check_declared_multiples_match_the_live_constants,
    check_no_unresolved_rows,
    check_immune_is_not_terminal,
    check_every_bucketed_row_uses_the_shared_table,
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
