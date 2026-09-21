#!/usr/bin/env python3
"""The published check count is computed from CONFIRMATIONS, so an unrelated drift lowers it.

Pass 236. `audit_evidence.py` publishes "(N checks, 0 drifted)" and guards it against the
number PROVES states. The total was `len(ok) + 1 + len(warn)` -- confirmations only -- so ANY
other guard that failed took the total down by one, and the guard then reported that lower
number under a message naming the wrong cause: "the count drifted when checks were added".

It did exactly that to me twice in one session. GAP had run one paragraph past its 40000 cap;
the audit said a check had gone missing. I went looking for a vanished check and there was
none: the cap guard had simply stopped confirming.

This probe LIFTS the live block out of `audit_evidence.py` -- it does not restate it, so it
cannot go stale the way a copied rule does -- and runs it against three synthetic states.
CPU-only, no device, no artifacts read.
"""
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "audit_evidence.py"


def lift():
    """Pull the check-COUNT block out of the live audit, dedented, ready to exec."""
    text = SRC.read_text()
    start = text.index("# --- and the check COUNT the summary quotes ---")
    start = text.index("if ORCH.is_file():", start)
    end = text.index('print("AUDIT of state/of3t/EVIDENCE.md', start)
    body = text[start:end]
    # drop the `if ORCH.is_file():` header and dedent its body by four
    lines = body.split("\n")[1:]
    out = []
    for ln in lines:
        if ln.startswith("    "):
            out.append(ln[4:])
        elif not ln.strip():
            out.append("")
        else:
            out.append(ln)
    return "\n".join(out)


def run(block, *, n_ok, n_warn, n_bad, stated):
    """Execute the lifted block against a synthetic audit state."""
    ns = {
        "_re": re,
        "ok": [f"ok-{i}" for i in range(n_ok)],
        "warn": [f"warn-{i}" for i in range(n_warn)],
        "bad": [f"bad-{i}" for i in range(n_bad)],
        "o": f"Recomputed from the artifacts on every compose ({stated} checks, 0 drifted):",
    }
    exec(block, ns)
    return ns["ok"][n_ok:], ns["warn"][n_warn:], ns["bad"][n_bad:]


def old_rule(*, n_ok, n_warn, stated):
    """The rule as it stood before pass 236, restated for the contrast only."""
    n_ran = n_ok + 1
    n_now = n_ran + n_warn
    if n_now != stated:
        return (f"DRIFT PROVES states ({stated} checks, 0 drifted) but this audit has {n_now} "
                f"({n_ran} confirmed + {n_warn} that announced they could not run) -- the count "
                f"drifted when checks were added (pass-133 recurrence)")
    return None


def main():
    block = lift()
    print(f"lifted {len(block.splitlines())} lines from {SRC}\n")
    failures = []

    # 1. the clean state: 165 confirmations so far + this one = 166, and PROVES says 166.
    a_ok, a_warn, a_bad = run(block, n_ok=165, n_warn=0, n_bad=0, stated=166)
    print("clean, PROVES states the true total")
    print(f"   ok   {a_ok}\n   warn {a_warn}\n   bad  {a_bad}")
    if len(a_ok) != 1 or a_bad or a_warn:
        failures.append("a correct count in a clean run must confirm and nothing else")

    # 2. the clean state with the count actually stale -- the message this guard exists for.
    b_ok, b_warn, b_bad = run(block, n_ok=165, n_warn=0, n_bad=0, stated=164)
    print("\nclean, PROVES states a stale total (a check really was added)")
    print(f"   ok   {b_ok}\n   warn {b_warn}\n   bad  {b_bad}")
    if len(b_bad) != 1 or "pass-133 recurrence" not in b_bad[0]:
        failures.append("a genuinely stale count must still drift with the pass-133 message")

    # 3. the state that misled me: the count is RIGHT, one unrelated guard drifted.
    c_ok, c_warn, c_bad = run(block, n_ok=164, n_warn=0, n_bad=1, stated=166)
    print("\nPROVES is CORRECT at 166; one unrelated guard (the GAP cap) drifted")
    print(f"   ok   {c_ok}\n   warn {c_warn}\n   bad  {c_bad}")
    if c_bad:
        failures.append("an unrelated drift must not be re-reported as a count drift")
    if len(c_warn) != 1 or "NOT EVALUABLE" not in c_warn[0]:
        failures.append("the guard must announce that it could not run (K60), not stay silent")

    print("\n   and what the pre-pass-236 rule said about that same state:")
    print(f"   {old_rule(n_ok=164, n_warn=0, stated=166)}")
    print("   -- 166 is the right answer and the run was told to go looking for a lost check.")

    if failures:
        for f in failures:
            print(f"\nFAIL {f}")
        return 1
    print("\nPASS the count is audited only where it is evaluable, and the true stale-count "
          "case still fires")
    return 0


if __name__ == "__main__":
    sys.exit(main())
