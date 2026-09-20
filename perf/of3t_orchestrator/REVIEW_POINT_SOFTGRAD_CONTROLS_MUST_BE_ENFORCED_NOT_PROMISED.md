# Review point to settle before `of3t-softgrad` is accepted (pass 185, row still live)

`of3t-softgrad` is measuring the two shippable softmax levers on the 51.1358 % scope's gradient
at full scope. Its design rests on a control I asked for and it has correctly written down:

> Order is deliberate: the two CONTROLS first. If `shipped` does not reproduce **7.426217** and
> `sm64` does not reproduce **0.0777758** the harness is wrong and the two levers in between
> mean nothing.

That is exactly right, and it is **the load-bearing part of the whole row**: the two new arms
(`softmax_precise`, `softmax_accurate`) are only interpretable as points between two ends whose
values are already known. If the harness has drifted, the middle two are numbers against a
moving baseline and say nothing.

## What I checked, and what I found

Both control values appear in `PREREGISTERED.md`, `chain.sh` and `devgrad_sg.sh` — **six, two and
two mentions**. A search across every pushed file in `perf/of3t_softgrad/` for an `assert`,
`exit`, `raise` or `sys.exit` guarding either number returns **nothing**. As of `4621ebf0f` the
stop is a **comment**, not a code fact.

This is the campaign's own recorded shape: an eligibility/firing condition stated in prose is not
a code fact, and a gate's verdict is not its value. A run that quietly produces four numbers with
a drifted `shipped` arm looks exactly like a run that produces four good ones.

## Why I am NOT filing this as a defect

The row is **mid-build**: it has pushed a pre-registration and a harness (`chain.sh`,
`devgrad_sg.sh`, `recap043.sh`, `cost.py`) and has published **no result artifact at all**. The
scoring and reporting step is not written yet, and that is the natural place for the assertion.
Calling this a defect now would be flagging the absence of code the row has not reached, and it
would mean shouting into a live measurement. `of3t-wirefix` closed an equivalent review point on
its own, before I raised it, once it got to the same stage.

## What settles it at conclusion

Either (a) the row's scoring step **asserts** both end arms against 7.426217 and 0.0777758 to a
stated tolerance and exits non-zero otherwise, or (b) the row reports both reproduced values in
its result artifact **beside** the two new arms, so a reader can check the baseline held without
running anything. (a) is better because it cannot be skipped; (b) is acceptable because it is
visible. What is not acceptable is four arms reported with the two ends absent from the artifact
— then the control existed only in a comment, and the middle two are unanchored.

Tolerance also has to be stated, and is not yet: "reproduce 7.426217" is not a test until it
says to how many digits.

---

## RESOLVED at pass 187 — and the control immediately earned its keep

`of3t-softgrad` reports, in `846495be3`: *"shipped reads **7.426217** and softmax_host_f64 reads
**0.0777758**, to every published digit, so the harness is the harness."* Both ends reproduce,
on a **rebuilt** 0.4.3 boundary — the original had been pruned by `of3t_rebase`, and the rebuild
was itself checked against four recorded numbers including a forward rel from a row that never
touched it. So the control is discharged in form (b) of what I asked for: the reproduced values
are reported beside the new arms, where a reader can check them.

**And it was not ceremonial.** With the baseline pinned, `softmax_precise` could be read
correctly as **all 547 tensors bit-identical to shipped** — a no-op (D110) — rather than as a
lever that happened to move nothing. Without a reproduced `shipped` arm, "identical to shipped"
is a statement about an unknown, and the honest reading would have been unavailable.

One thing I asked for is still not stated and should be at conclusion: **the tolerance**. "Reads
7.426217 to every published digit" is a stronger claim than a tolerance and is fine as reported,
but the row should say what it would have done with a mismatch in the last digit.
