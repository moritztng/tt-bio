# c13-land-first: CLOSED, 0 seconds for this row to land

This row charter lists "c13-land-first 0.4798 s, long recorded as SHIPPED: 0.0000 s" as backlog.
Checked against git and against main own source comments rather than against the charter prose.

## The branch is a record, not a candidate

    git rev-list --left-right --count origin/main...origin/wk/c13-land-first  ->  5745  0

0 ahead. Everything that branch did is already contained in main. Rank by behind-count first, and
an ahead-count of zero settles it without reading a diff.

## The 0.4798 s was a STACK and it never described a default

Main says so itself at `tt_bio/tenstorrent.py` on the cond-hoist block: *"The same session also
carried TT_BIO_UNFUSED_SILU and read +0.4798 s for the pair, but that flag is held off on a
Protenix-v2 accuracy regression, so the stack number does not describe any default."*

Its two halves resolve in opposite directions and both are already decided:

**hoist -- `TT_BIO_DIT_COND_HOIST`, DEFAULT ON since 2026-09-18, worth +0.2052 s** at the 512 aa
cdk2x2 fold, paired over 5 interleaved reps at a forced 1350 MHz, CI [+0.1561, +0.2543], against
that session paired A/A floor of +0.0324 +/- 0.1087. Main calls this "the single-lever figure that
ships". Already landed, and by another row, so it is not this row TOTAL to claim.

**silu -- `TT_BIO_UNFUSED_SILU`, DEFAULT OFF, and held there by Moritz on measured accuracy.** On
cdk2x2_512 Protenix-v2 loses 0.05088 and 0.07210 CA-lDDT per domain against 1HCL, and the arms are
fully rank-separated over four seeds: the worst flag-off fold scores 0.03905 and 0.05107 above the
best flag-on one. Main comment ends "Do not reopen without a Protenix-v2 CA-lDDT-vs-1HCL re-score
on the configuration you want to ship."

This row charter is explicit that anything failing the accuracy bar is Moritz and not mine, so the
+0.2336 s silu half is correctly unclaimed and stays that way.

## Net

Nothing to land. The backlog entry was stale in both halves, and the answer was written in the
code the whole time.
