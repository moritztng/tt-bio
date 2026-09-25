# Kernel-row brief template — backward sprint

Instantiate one per job on `of3t-bwsurvey`'s JOBS list, biggest share of the 6-7x first. Substitute
`<slug>`, `<OP>`, `<TT-TRAIN SOURCE>`, `<CALL SITES>`, `<SHARE>`, `<BINDING ROOF>`. Every kernel row
is engine-level: the slug is named for the OP, never for OpenFold3, because the same kernel serves
Boltz-2, OF3T, BC2 and RFD3.

A dispatch has two halves and one is invisible. Write the brief AND add the row's `EXTRA` entry to
`workstreams/_of3t_donecheck.py` in the same pass, or the row finishes its measurement and then
defers forever unable to conclude (LEDGER R193, cost: 5 iterations).

---

    # Workstream: <slug>
    #DISPATCH: host=<host> card=<card> repo=tt-bio maxit=<n> tier=opus5
    DONE_CHECK: python3 /home/moritz/.coworker/workstreams/_of3t_donecheck.py <slug>

    **READ FIRST: /home/moritz/.coworker/state/of3t/BACKWARD.md** (the shared findings) and
    **§A46 of `state/of3t/PROTOCOL.md`** (how a backward kernel's win is graded). Do not re-derive
    either.

    Build the backward for **<OP>**. `of3t-bwsurvey` sized it at **<SHARE>** of the 6-7x software
    gap and `of3t-intensity` says the part it sits in is bound by **<BINDING ROOF>**; those two
    numbers are why this row exists and why it is ordered where it is.

    1. ADAPTED: start from **<TT-TRAIN SOURCE>**. Say what you changed and why. If you end up
       authoring instead, say what you tried from tt-train and exactly why it did not fit --
       "adapt before you author" is standing, and a row that rewrites what already exists owes
       that sentence.
    2. VJP: the gradient against a float64 reference that is ITSELF validated by central finite
       differences. Per-tensor worst case, located by parameter path. Not the forward: a bfp8 arm
       on this fleet read forward cosine 0.99999 and backward Inf/NaN.
    3. SPEED: seconds off the 466.70 s crop-384 step. Arms interleaved on one board, AICLK sampled
       DURING, board class named, host_quiet green. Price it against <BINDING ROOF> and say what
       fraction of the available headroom you took.
    4. UNIFIED: name every model that executes this site (<CALL SITES>). If the change is reachable
       from an inference fold, the A/B against an A/A floor on each of them, floor reported FIRST,
       before you propose any default. Inference regressing is a hard stop, not a trade.
    5. FIX: land it on `wk/<slug>`, pushed. I compose onto `wk/of3t-bwd`.

    <HOUSE RULES BLOCK>

    State doc `state/<slug>.md`: ADAPTED:, VJP:, SPEED:, UNIFIED:, FIX:, VERDICT:.

---

## The gate entry that must land in the same pass

    "<slug>": {"min":2000, "req":[
       (r"^ADAPTED:\s*\S","ADAPTED: the tt-train op it started from and what changed -- or what "
        "was tried and why it did not fit"),
       (r"^VJP:\s*\S","VJP: per-tensor worst case against a float64 reference validated by "
        "central finite differences, located by parameter path"),
       (r"^SPEED:\s*\S","SPEED: seconds off the 466.70 s step, arms interleaved, DURING-sampled "
        "AICLK, board class named"),
       (r"^UNIFIED:\s*\S","UNIFIED: every model that executes the site, with the inference A/B "
        "against an A/A floor if it is reachable from a fold -- or that it is not"),
       (r"^FIX:\s*\S","FIX: what landed, on which branch"),
       (r"^VERDICT:\s*\S","VERDICT:")],
      "measured":["VJP","SPEED"],
      "checks":["f64","worst_case_vjp","clock"]}

`worst_case_vjp` does not exist yet in `_of3t_donecheck.py`; the existing `worst_case` reads a
`GRADIENTS:` field. Add the variant that reads `VJP:` with the same two conditions (a number, and a
dotted parameter path) when the first kernel row is dispatched. The reason the path is required and
a bare number is not enough: quoting the protocol's own bar back satisfied a bare-number check
while every field read NOT YET MEASURED (found by `of3t-equivalence`, 2026-09-19).
