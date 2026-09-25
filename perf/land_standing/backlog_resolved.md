# Two backlog items this row was carrying are already settled on main

Both were seeded into the brief as live candidates. Neither is. Both were closed by evidence that
is already committed, and checking them cost no device time.

## 1. `c13-land-first`'s "0.4798 s, long recorded as SHIPPED: 0.0000 s" — it IS shipped, as +0.2052 s

`15f1ab0ac`, `50baeb7f0` and `393cf26e1` are all ancestors of `origin/main`, and
`TT_BIO_DIT_COND_HOIST` reads `env_flag(..., True)` there. Main's own comment at
`tt_bio/tenstorrent.py:1553` says what shipped and what did not, and it is worth quoting because
the brief's number is the wrong one:

> DEFAULT ON since 2026-09-18. Worth **+0.2052 s** at the 512 aa cdk2x2 fold, paired over 5
> interleaved reps at a forced 1350 MHz (95 % CI [+0.1561,+0.2543], arm means 14.5880 -> 14.3920 s)
> against that session's paired A/A floor of +0.0324 s +/-0.1087. That single-lever figure is the
> one that ships. The same session also carried TT_BIO_UNFUSED_SILU and read **+0.4798 s for the
> pair**, but that flag is held off on a Protenix-v2 accuracy regression, so the stack number does
> not describe any default.

So **0.4798 s never described anything landable**: it is a two-lever stack whose other half,
`TT_BIO_UNFUSED_SILU`, is `False` on main because Protenix-v2 loses 0.05 CA-lDDT with it on
(`393cf26e1`). Approving a stack as a stack or not at all is the standing rule, and it was applied
correctly. **Nothing to land. The shipped figure is +0.2052 s.**

## 2. The `out_block_h` lever, "1.2894x on the op" — NO-GO, and by its own pre-registered bar

`033365387` is on `origin/main` and already priced it at the fold's own shapes. The lever's code
(`TT_BIO_LINEAR_KBLOCK`, default `False`) sits unlanded on `wk/c12-kblock-unlock` (`aedbb9d1d`).

ttnn does not choose `out_block_h` at all: `get_multi_dim_per_core_factor` returns `per_core_M`
unchanged whenever the circular buffers fit, so `out_block_h = per_core_M` is the absence of a
choice rather than a bad one. The 53 `CORE_GRID_MAIN` sites spread M over 110 cores, so
`per_core_M` is 2, 3 or 26 at every shape carrying real work and there is nothing to cut.

Censused on a live 512 aa protenix-v2 fold, 135,789 of 152,065 calls take a derived config and
83 % of their FLOPs sit in four shapes. Laddered on qb2 card 1, AICLK forced and during-sampled at
1350 MHz, arms interleaved, A/A floors 0.22-1.67 %:

    SW1  52.7 % of derived FLOPs   best out_block_h 1.0124x
    SW3  26.4 %                    1.0000x
    DIT   4.2 %                    1.0000x
    OPM   2.0 %                    1.0964x

Priced against measured call counts and the campaign's measured in-fold factor of 0.8226, the
whole surface is **0.0279 s = 1.00057x on a 48.62 s fold, against a pre-registered transfer bar of
0.35 s.** It misses its own bar by more than ten times.

**The 1.2894x is the textbook case of "an op-level ratio is a screen, never a result".** Here the
screen overstates the fold effect by roughly 500x.

Two corrections to how the brief carried this entry:
- It credits `out_block_h` with being "0.5 % MORE accurate than the shipped path against a float64
  reference". `out_block_h` is **bit-exact** at every shape and every rung. The non-bit-exact lever
  is the sibling `in0_block_w`, and against a float64 host reference at SW3 it reads mean
  **9.5456e-05 against the derived config's 9.5307e-05** — very slightly worse, not better, with
  PCC equal to nine places. The two levers were conflated and the direction inverted.
- `in0_block_w` is the one that is actually alive at those sites (1.1800x at SW3, 1.9483x at
  fixterm's key B), and priced the same way it is **0.1723 s = 1.00356x** — also under the 0.35 s
  bar. So the brief's "check key B" falsifier was run, and key B's 1.9483x does not carry the
  surface either.
