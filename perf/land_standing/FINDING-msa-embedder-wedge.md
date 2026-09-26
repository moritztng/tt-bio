# OpenFold3's MSA embedder wedged inside one ttnn slice at MSA depth 513, 832 tokens

Filed 2026-09-26 by `land-standing`. Seen **once in four otherwise byte-identical legs**, so the
conditions below are what was observed, not a characterised reproduction. The stack is exact.

## What happened

An OpenFold3 fold on qb2 card 3 (p300c) sat at `trunk 0/4` for **28 minutes** and never advanced.
The same fold, same fixture, same settings, had completed three times in **92-187 s** in the
minutes before it.

It was not slow, and it was not blocked. Five `py-spy` dumps over 30 s were byte-identical, and
`utime` advanced **5.97 s per 6 s of wall clock** — a 100 % user-CPU spin inside a single call:

    ttnn/decorators.py:473          __call__
    ttnn/operations/core.py:146     __getitem__          <- one slice, for 28 minutes
    tt_bio/tenstorrent.py:4223      run
    tt_bio/tenstorrent.py:3260      _with_dram_narrowing
    tt_bio/tenstorrent.py:4249      _fp32_softmax_attention
    tt_bio/tenstorrent.py:9022      _attend_heads
    tt_bio/tenstorrent.py:9368      _attend_pair
    tt_bio/tenstorrent.py:5730      row_block_after_refusal
    tt_bio/openfold3_msa_embedder.py:151
    tt_bio/openfold3_trunk.py:249

So: the MSA embedder's attention was refused by DRAM, `row_block_after_refusal` put it on the
row-blocked materialised-fp32 route, and one slice on that route never returned.

## The conditions, stated exactly

- **OpenFold3, 832 tokens, MSA depth 513.** Every other fold this row had taken was
  `--single_sequence`; the wedge appeared in the first group of runs with a deep alignment.
- `TT_BIO_TRIATT_DIVIDING_K=0` — the **shipped default**. The lever under test is not implicated,
  and two of the three fast legs ran the same value.
- The only differences from the three fast legs were `--seed 1` and a fresh `--out_dir`, neither
  of which touches the trunk. **No cause was established.** The honest reading is that something
  about this path is nondeterministic at this depth, not that the seed flag causes it.

## Why it matters beyond this row

The route that wedged is reached on **default settings** — no flag is involved. Folding OpenFold3
at around 832 tokens with a real MSA is an ordinary thing for a user to do, and this path is what
serves it once the embedder's attention exceeds DRAM. One occurrence is not a frequency, but the
failure mode is a silent forever-hang holding a chip, not an error.

## Operational consequence, which is the expensive part

`worker.py:1987` installs a SIGTERM handler that raises. Python delivers a signal between
bytecodes, so a process inside **one** long C call cannot take it: the TERM sat pending for
25 minutes and never landed. Recovering the chip therefore needs SIGKILL, and
`killing-a-wedged-fold-leaves-the-card-unopenable-and-the-next-open-kills-the-host` means SIGKILL
needs a board-pair reset behind it. Card 2 (card 3's board pair, BDFs `0000:03:00.0` and
`0000:04:00.0`) was idle when the kill was decided and was taken by another row within seconds, so
the reset became unavailable and card 3 is blocked rather than reset. See
`state/cardblock-qb2-3`.

**A hang on this path costs a chip until a board pair happens to be free.** That is the part worth
fixing ahead of the hang itself: a wall-clock guard around the row-blocked fallback would turn a
forever-hang into a raise the worker can take.
